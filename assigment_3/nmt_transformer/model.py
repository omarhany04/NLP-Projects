import torch
import torch.nn as nn
from transformer_block import TransformerEmbedding, EncoderLayer, DecoderLayer


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len,
                 num_layers, num_heads, intermediate_dim, dropout):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src_ids, src_key_padding_mask=None):
        """
        src_ids            : (B, src_len)
        src_key_padding_mask: (B, src_len)  True = pad token

        Returns:
            enc_out       : (B, src_len, d)
            attn_weights  : list of (B, H, src_len, src_len), one per layer
        """
        x = self.embedding(src_ids)
        attn_weights = []
        for layer in self.layers:
            x, w = layer(x, src_key_padding_mask=src_key_padding_mask)
            attn_weights.append(w)
        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len,
                 num_layers, num_heads, intermediate_dim, dropout):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, embed_dim, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt_ids, enc_output,
                tgt_key_padding_mask=None, src_key_padding_mask=None):
        """
        tgt_ids            : (B, tgt_len)
        enc_output         : (B, src_len, d)
        tgt_key_padding_mask: (B, tgt_len)
        src_key_padding_mask: (B, src_len)

        Returns:
            dec_out             : (B, tgt_len, d)
            self_attn_weights   : list of (B, H, tgt_len, tgt_len)
            cross_attn_weights  : list of (B, H, tgt_len, src_len)
        """
        x = self.embedding(tgt_ids)
        self_attn_weights  = []
        cross_attn_weights = []
        for layer in self.layers:
            x, self_w, cross_w = layer(
                x, enc_output,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            self_attn_weights.append(self_w)
            cross_attn_weights.append(cross_w)
        return x, self_attn_weights, cross_attn_weights


class TransformerNMT(nn.Module):
    """Full encoder-decoder Transformer for NMT.

    Output projection is weight-tied to the decoder's token embedding matrix:
        logits_i = E_dec · h_i   (E_dec shape: tgt_vocab × d)
    so logits shape is (B, tgt_len, tgt_vocab).
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, max_seq_len,
                 num_encoder_layers, num_decoder_layers, num_heads,
                 intermediate_dim, dropout, pad_id=3):
        super().__init__()
        self.pad_id = pad_id

        self.encoder = Encoder(
            src_vocab_size, embed_dim, max_seq_len,
            num_encoder_layers, num_heads, intermediate_dim, dropout,
        )
        self.decoder = Decoder(
            tgt_vocab_size, embed_dim, max_seq_len,
            num_decoder_layers, num_heads, intermediate_dim, dropout,
        )
        # Weight tying: no separate Linear — reuse decoder embedding weights.
        # logits = dec_hidden @ token_embed_weight.T

    def forward(self, src_ids, tgt_ids):
        """
        src_ids : (B, src_len)   French token IDs
        tgt_ids : (B, tgt_len)   English token IDs (teacher-forcing input)

        Returns:
            logits             : (B, tgt_len, tgt_vocab)
            enc_attn_weights   : list[layer] of (B, H, src_len, src_len)
            self_attn_weights  : list[layer] of (B, H, tgt_len, tgt_len)
            cross_attn_weights : list[layer] of (B, H, tgt_len, src_len)
        """
        src_pad_mask = (src_ids == self.pad_id)   # (B, src_len)
        tgt_pad_mask = (tgt_ids == self.pad_id)   # (B, tgt_len)

        enc_output, enc_attn = self.encoder(
            src_ids, src_key_padding_mask=src_pad_mask
        )
        dec_output, self_attn, cross_attn = self.decoder(
            tgt_ids, enc_output,
            tgt_key_padding_mask=tgt_pad_mask,
            src_key_padding_mask=src_pad_mask,
        )

        # Weight-tied projection: (B, tgt_len, d) @ (d, tgt_vocab) → (B, tgt_len, tgt_vocab)
        logits = dec_output @ self.decoder.embedding.token_embedding.weight.T

        return logits, enc_attn, self_attn, cross_attn
