import torch


def _prepare_source(src_text, src_tokenizer, max_len, eos_id, device):
    src_ids = src_tokenizer.encode(src_text, add_special_tokens=False)
    src_ids = src_ids[: max_len - 1] + [eos_id]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    return src_ids, src_tensor


def _finish_tokens(tokens, bos_id, eos_id):
    if tokens and tokens[0] == bos_id:
        tokens = tokens[1:]
    if eos_id in tokens:
        tokens = tokens[:tokens.index(eos_id)]
    return tokens


def _stack_attention(attention_steps, keep_len):
    if keep_len == 0 or not attention_steps:
        return None
    attn = torch.stack(attention_steps[:keep_len], dim=0)
    return attn.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def lstm_greedy_decode(model, src_text, src_tokenizer, tgt_tokenizer,
                       max_len, bos_id, eos_id, pad_id, device):
    """Greedy decoding for the recurrent NMT model."""
    del pad_id
    model.eval()

    _, src_tensor = _prepare_source(src_text, src_tokenizer, max_len, eos_id,
                                    device)
    encoder_outputs, state, src_pad_mask = model.encode(src_tensor)

    generated = [bos_id]
    attentions = []
    for _ in range(max_len):
        input_id = torch.tensor([generated[-1]], dtype=torch.long).to(device)
        logits, state, attn = model.decode_step(
            input_id, state, encoder_outputs, src_pad_mask
        )
        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)
        attentions.append(attn.squeeze(0).detach().cpu())
        if next_token == eos_id:
            break

    output_tokens = _finish_tokens(generated, bos_id, eos_id)
    return tgt_tokenizer.decode(output_tokens), output_tokens


@torch.no_grad()
def lstm_beam_search(model, src_text, src_tokenizer, tgt_tokenizer,
                     beam_size, max_len, bos_id, eos_id, pad_id, device,
                     length_penalty=0.6):
    """Beam search decoding for the recurrent NMT model.

    Returns (translation_string, token_ids, attention_weights), where
    attention_weights has shape (1, 1, tgt_len, src_len).
    """
    del pad_id
    model.eval()

    _, src_tensor = _prepare_source(src_text, src_tokenizer, max_len, eos_id,
                                    device)
    encoder_outputs, init_state, src_pad_mask = model.encode(src_tensor)

    # Each beam: (raw_log_prob, tokens_with_bos, state, attention_steps)
    beams = [(0.0, [bos_id], init_state, [])]
    completed = []

    for _ in range(max_len):
        candidates = []
        for score, tokens, state, attention_steps in beams:
            input_id = torch.tensor([tokens[-1]], dtype=torch.long).to(device)
            logits, next_state, attn = model.decode_step(
                input_id, state, encoder_outputs, src_pad_mask
            )
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk_vals, topk_ids = torch.topk(log_probs, beam_size)

            for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                new_tokens = tokens + [tok_id]
                new_score = score + val
                new_attention_steps = attention_steps + [
                    attn.squeeze(0).detach().cpu()
                ]
                new_state = (
                    next_state[0].clone(),
                    next_state[1].clone(),
                )

                if tok_id == eos_id:
                    norm = new_score / (len(new_tokens) ** length_penalty)
                    completed.append(
                        (norm, new_tokens, new_attention_steps)
                    )
                else:
                    candidates.append(
                        (new_score, new_tokens, new_state, new_attention_steps)
                    )

        if not candidates:
            break

        candidates.sort(
            key=lambda item: item[0] / (len(item[1]) ** length_penalty),
            reverse=True,
        )
        beams = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    for score, tokens, _, attention_steps in beams:
        norm = score / (len(tokens) ** length_penalty)
        completed.append((norm, tokens, attention_steps))

    completed.sort(key=lambda item: item[0], reverse=True)
    _, best_tokens_with_bos, best_attention_steps = completed[0]
    best_tokens = _finish_tokens(best_tokens_with_bos, bos_id, eos_id)
    attention = _stack_attention(best_attention_steps, keep_len=len(best_tokens))

    translation = tgt_tokenizer.decode(best_tokens)
    return translation, best_tokens, attention


@torch.no_grad()
def lstm_translate_batch(model, src_texts, src_tokenizer, tgt_tokenizer,
                         beam_size, max_len, bos_id, eos_id, pad_id, device):
    """Translate a list of French sentences with LSTM beam search."""
    return [
        lstm_beam_search(
            model, src, src_tokenizer, tgt_tokenizer, beam_size, max_len,
            bos_id, eos_id, pad_id, device,
        )[0]
        for src in src_texts
    ]
