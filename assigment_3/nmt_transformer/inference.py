import torch


@torch.no_grad()
def greedy_decode(model, src_text, src_tokenizer, tgt_tokenizer,
                  max_len, bos_id, eos_id, pad_id, device):
    """Simple greedy decoding (argmax at each step)."""
    model.eval()

    src_ids = src_tokenizer.encode(src_text, add_special_tokens=False)
    src_ids = src_ids[: max_len - 1] + [eos_id]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    src_pad_mask = (src_tensor == pad_id)
    enc_output, _ = model.encoder(src_tensor, src_key_padding_mask=src_pad_mask)

    generated = [bos_id]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        dec_output, _, _ = model.decoder(
            tgt_tensor, enc_output, src_key_padding_mask=src_pad_mask
        )
        logits = dec_output[:, -1, :] @ model.decoder.embedding.token_embedding.weight.T
        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)
        if next_token == eos_id:
            break

    tokens = generated[1:]   # strip BOS
    if eos_id in tokens:
        tokens = tokens[: tokens.index(eos_id)]
    return tgt_tokenizer.decode(tokens), tokens


@torch.no_grad()
def beam_search(model, src_text, src_tokenizer, tgt_tokenizer,
                beam_size, max_len, bos_id, eos_id, pad_id, device,
                length_penalty=0.6):
    """Beam search decoding.

    Scores beams with length-penalty normalization:
        score = cumulative_log_prob / (seq_len ** length_penalty)

    Returns (translation_string, token_ids, cross_attn_weights_last_layer).
    cross_attn_weights shape: (1, H, tgt_len, src_len)  — for visualization.
    """
    model.eval()

    # Encode source once
    src_ids = src_tokenizer.encode(src_text, add_special_tokens=False)
    src_ids = src_ids[: max_len - 1] + [eos_id]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    src_pad_mask = (src_tensor == pad_id)
    enc_output, _ = model.encoder(src_tensor, src_key_padding_mask=src_pad_mask)

    # Each beam: (cumulative_log_prob, token_list, cross_attn_last)
    beams     = [(0.0, [bos_id], None)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, tokens, _ in beams:
            tgt_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            dec_output, _, cross_attn = model.decoder(
                tgt_tensor, enc_output, src_key_padding_mask=src_pad_mask
            )
            # cross_attn[-1]: (1, H, tgt_len, src_len)

            logits    = dec_output[:, -1, :] @ model.decoder.embedding.token_embedding.weight.T
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)   # (V,)

            topk_vals, topk_ids = torch.topk(log_probs, beam_size)
            for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                new_tokens = tokens + [tok_id]
                new_score  = score + val
                if tok_id == eos_id:
                    # Normalize by length before storing as completed
                    norm = new_score / (len(new_tokens) ** length_penalty)
                    completed.append((norm, new_tokens, cross_attn[-1]))
                else:
                    candidates.append((new_score, new_tokens, cross_attn[-1]))

        if not candidates:
            break

        # Keep top beam_size active beams (normalized score for ranking)
        candidates.sort(
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True,
        )
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    # Flush remaining active beams
    for score, tokens, attn in beams:
        norm = score / (len(tokens) ** length_penalty)
        completed.append((norm, tokens, attn))

    completed.sort(key=lambda x: x[0], reverse=True)
    _, best_tokens, best_attn = completed[0]

    # Strip BOS / EOS
    if best_tokens and best_tokens[0] == bos_id:
        best_tokens = best_tokens[1:]
    if eos_id in best_tokens:
        best_tokens = best_tokens[: best_tokens.index(eos_id)]

    translation = tgt_tokenizer.decode(best_tokens)
    return translation, best_tokens, best_attn


@torch.no_grad()
def translate_batch(model, src_texts, src_tokenizer, tgt_tokenizer,
                    beam_size, max_len, bos_id, eos_id, pad_id, device):
    """Translate a list of French sentences; returns list of English strings."""
    return [
        beam_search(
            model, src, src_tokenizer, tgt_tokenizer,
            beam_size, max_len, bos_id, eos_id, pad_id, device,
        )[0]
        for src in src_texts
    ]