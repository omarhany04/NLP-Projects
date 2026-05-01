import math

import torch
import torch.nn as nn


class ManualEmbedding(nn.Module):
    """Learnable token embedding table implemented from parameters."""

    def __init__(self, vocab_size, embed_size, padding_idx=None):
        super().__init__()
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(vocab_size, embed_size))
        nn.init.normal_(self.weight, mean=0.0, std=embed_size ** -0.5)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()

    def forward(self, token_ids):
        embeddings = self.weight[token_ids]
        if self.padding_idx is None:
            return embeddings
        pad_mask = (token_ids == self.padding_idx).unsqueeze(-1)
        return embeddings.masked_fill(pad_mask, 0.0)


class ManualLSTMCell(nn.Module):
    """Single LSTM cell implemented directly from the gate equations."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        bound = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)

    def forward(self, x_t, h_prev, c_prev):
        gates = x_t @ self.weight_ih + h_prev @ self.weight_hh + self.bias
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)

        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, c_t


class ManualLSTMLayer(nn.Module):
    """Unidirectional LSTM layer implemented from manual cell updates."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, inputs, lengths=None, reverse=False, initial_state=None):
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        if initial_state is None:
            h_t = inputs.new_zeros(batch_size, self.hidden_size)
            c_t = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            h_t, c_t = initial_state

        outputs = inputs.new_zeros(batch_size, seq_len, self.hidden_size)
        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for t in time_steps:
            next_h, next_c = self.cell(inputs[:, t], h_t, c_t)

            if lengths is None:
                active = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            else:
                active = (lengths.to(device) > t).unsqueeze(-1)

            h_t = torch.where(active, next_h, h_t)
            c_t = torch.where(active, next_c, c_t)
            outputs[:, t] = torch.where(active, h_t, torch.zeros_like(h_t))

        return outputs, (h_t, c_t)


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder built from manual LSTM cells.

    For each source position i, the encoder returns
        h_i = [forward_h_i ; backward_h_i] in R^(2h).
    The final forward state and final backward state initialize the decoder.
    """

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout, pad_id=3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.embedding = ManualEmbedding(vocab_size, embed_size,
                                         padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            input_size = embed_size if layer_idx == 0 else 2 * hidden_size
            self.forward_layers.append(ManualLSTMLayer(input_size, hidden_size))
            self.backward_layers.append(ManualLSTMLayer(input_size, hidden_size))

        self.init_hidden = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, src_ids):
        lengths = (src_ids != self.pad_id).sum(dim=1).clamp_min(1)
        layer_input = self.dropout(self.embedding(src_ids))
        forward_last = backward_last = None

        for layer_idx, (forward_layer, backward_layer) in enumerate(
            zip(self.forward_layers, self.backward_layers)
        ):
            forward_out, (forward_last, _) = forward_layer(
                layer_input, lengths=lengths, reverse=False
            )
            backward_out, (backward_last, _) = backward_layer(
                layer_input, lengths=lengths, reverse=True
            )
            layer_output = torch.cat([forward_out, backward_out], dim=-1)
            if layer_idx < len(self.forward_layers) - 1:
                layer_output = self.dropout(layer_output)
            layer_input = layer_output

        if forward_last is None or backward_last is None:
            raise RuntimeError("BiLSTM encoder did not run any recurrent layers")

        decoder_init = torch.tanh(
            self.init_hidden(torch.cat((forward_last, backward_last), dim=-1))
        )
        return layer_input, decoder_init


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention.

    e_{t,i} = v^T tanh(W s_{t-1} + U h_i)
    c_t     = sum_i alpha_{t,i} h_i
    """

    def __init__(self, decoder_hidden_size, encoder_output_size):
        super().__init__()
        self.decoder_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size,
                                      bias=False)
        self.encoder_proj = nn.Linear(encoder_output_size, decoder_hidden_size,
                                      bias=False)
        self.energy = nn.Linear(decoder_hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, src_key_padding_mask=None):
        dec = self.decoder_proj(decoder_state).unsqueeze(1)
        enc = self.encoder_proj(encoder_outputs)
        scores = self.energy(torch.tanh(dec + enc)).squeeze(-1)

        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class AttentiveLSTMDecoder(nn.Module):
    """Unidirectional LSTM decoder built from manual LSTM cells."""

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 dropout, pad_id=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = ManualEmbedding(vocab_size, embed_size,
                                         padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(
            decoder_hidden_size=hidden_size,
            encoder_output_size=2 * hidden_size,
        )

        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            input_size = embed_size + 2 * hidden_size if layer_idx == 0 else hidden_size
            self.cells.append(ManualLSTMCell(input_size, hidden_size))

        self.output = nn.Linear(embed_size + hidden_size + 2 * hidden_size,
                                vocab_size)

    def init_state(self, decoder_init):
        hidden = decoder_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        return hidden, cell

    def forward_step(self, input_ids, state, encoder_outputs,
                     src_key_padding_mask=None):
        hidden, cell = state
        prev_state = hidden[-1]
        embedded = self.dropout(self.embedding(input_ids))

        context, attn_weights = self.attention(
            prev_state, encoder_outputs, src_key_padding_mask
        )

        layer_input = torch.cat([embedded, context], dim=-1)
        next_hidden, next_cell = [], []
        for layer_idx, lstm_cell in enumerate(self.cells):
            h_t, c_t = lstm_cell(layer_input, hidden[layer_idx], cell[layer_idx])
            next_hidden.append(h_t)
            next_cell.append(c_t)
            layer_input = self.dropout(h_t) if layer_idx < len(self.cells) - 1 else h_t

        next_state = (torch.stack(next_hidden), torch.stack(next_cell))
        decoder_state = next_state[0][-1]
        logits = self.output(torch.cat([decoder_state, context, embedded], dim=-1))
        return logits, next_state, attn_weights


class RecurrentNMT(nn.Module):
    """BiLSTM encoder + unidirectional LSTM decoder with additive attention."""

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256,
                 hidden_size=512, num_layers=1, dropout=0.3, pad_id=3):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = BiLSTMEncoder(
            src_vocab_size, embed_size, hidden_size, num_layers, dropout, pad_id
        )
        self.decoder = AttentiveLSTMDecoder(
            tgt_vocab_size, embed_size, hidden_size, num_layers, dropout, pad_id
        )

    def encode(self, src_ids):
        src_pad_mask = (src_ids == self.pad_id)
        encoder_outputs, decoder_init = self.encoder(src_ids)
        decoder_state = self.decoder.init_state(decoder_init)
        return encoder_outputs, decoder_state, src_pad_mask

    def decode_step(self, input_ids, state, encoder_outputs, src_pad_mask=None):
        return self.decoder.forward_step(
            input_ids, state, encoder_outputs, src_pad_mask
        )

    def forward(self, src_ids, tgt_ids):
        """Teacher-forced decoding.

        Returns a transformer-compatible tuple:
            logits, None, None, attention_weights
        attention_weights has shape (B, 1, T, S), treating additive attention
        as a single attention head for visualization.
        """
        encoder_outputs, state, src_pad_mask = self.encode(src_ids)
        logits, attentions = [], []

        for t in range(tgt_ids.size(1)):
            step_logits, state, attn_weights = self.decode_step(
                tgt_ids[:, t], state, encoder_outputs, src_pad_mask
            )
            logits.append(step_logits.unsqueeze(1))
            attentions.append(attn_weights.unsqueeze(1))

        logits = torch.cat(logits, dim=1)
        attention_weights = torch.cat(attentions, dim=1).unsqueeze(1)
        return logits, None, None, attention_weights
