import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sacrebleu


# --------------------------------------------------------------------------- #
# Device helpers                                                               #
# --------------------------------------------------------------------------- #

# Part 2 utilities: chooses GPU when available so LSTM training can run faster.
def get_device(preferred="cuda"):
    """Return a torch.device, preferring CUDA for training when available."""
    preferred = preferred.lower()
    if preferred not in {"cuda", "cpu", "auto"}:
        raise ValueError("preferred must be one of: 'cuda', 'cpu', 'auto'")

    if preferred == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")

    if preferred == "cuda":
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot use it. "
            "Install a CUDA-enabled PyTorch build, then restart the notebook kernel."
        )

    return torch.device("cpu")


# Part 2 utilities: prints a friendly CPU/GPU description for the notebook and live tester.
def describe_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        return f"cuda - {name} ({memory_gb:.1f} GB)"
    cuda_note = "CUDA unavailable"
    if torch.version.cuda is None:
        cuda_note = "CUDA unavailable; installed PyTorch is CPU-only"
    return f"cpu ({cuda_note})"


# --------------------------------------------------------------------------- #
# BLEU                                                                          #
# --------------------------------------------------------------------------- #

# Part 2 BLEU: removes tokenizer boundary markers before comparing translations.
def _clean(text):
    """Strip SentencePiece ▁ boundary markers and collapse whitespace."""
    return ' '.join(text.replace('▁', ' ').split())


# Part 2 BLEU: computes corpus BLEU between model translations and references.
def compute_bleu(hypotheses, references):
    """
    hypotheses : list of predicted strings
    references : list of reference strings

    Returns a sacrebleu BLEU score object.  Access .score for the float value.
    """
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result


# Part 2 BLEU: translates the test split with beam search and computes the final BLEU score.
def compute_bleu_dataset(model, loader_raw, src_tokenizer, tgt_tokenizer,
                         beam_size, max_len, bos_id, eos_id, pad_id, device,
                         max_samples=None, decode_fn=None):
    """Run beam search over a raw HuggingFace dataset split and compute BLEU."""
    if decode_fn is None:
        from inference import beam_search as decode_fn

    hypotheses, references = [], []
    for i, sample in enumerate(loader_raw):
        if max_samples and i >= max_samples:
            break
        src_text = sample["text_fr"]
        ref_text = sample["text_en"]
        hyp, _, _ = decode_fn(
            model, src_text, src_tokenizer, tgt_tokenizer,
            beam_size, max_len, bos_id, eos_id, pad_id, device,
        )
        hypotheses.append(_clean(hyp))
        references.append(ref_text)

    return compute_bleu(hypotheses, references), hypotheses, references


# --------------------------------------------------------------------------- #
# Attention Visualization                                                       #
# --------------------------------------------------------------------------- #

# Part 2 visualization: draws the additive-attention weights for one attention head.
def plot_attention(attn_weights, src_tokens, tgt_tokens,
                   title="Attention", head=0, figsize=(8, 6)):
    """
    attn_weights : (B, H, tgt_len, src_len) tensor  or  (H, tgt_len, src_len)
    src_tokens   : list of source token strings (length = src_len)
    tgt_tokens   : list of target token strings (length = tgt_len)
    head         : which attention head to visualize
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]   # take first batch item → (H, tgt, src)
    w = attn_weights[head].cpu().detach().numpy()   # (tgt_len, src_len)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(w, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=9)

    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Target tokens")
    ax.set_title(f"{title}  (head {head})")
    fig.tight_layout()
    return fig


# Part 2 visualization: draws every attention head when a model returns multi-head weights.
def plot_all_heads(attn_weights, src_tokens, tgt_tokens,
                   title="Attention", figsize=None):
    """Plot all attention heads in a single figure."""
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]   # (H, tgt, src)
    H = attn_weights.shape[0]
    cols = min(H, 4)
    rows = (H + cols - 1) // cols
    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for h in range(H):
        w = attn_weights[h].cpu().detach().numpy()
        axes[h].imshow(w, aspect="auto", cmap="Blues")
        axes[h].set_title(f"Head {h}", fontsize=9)
        axes[h].set_xticks(range(len(src_tokens)))
        axes[h].set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=7)
        axes[h].set_yticks(range(len(tgt_tokens)))
        axes[h].set_yticklabels(tgt_tokens, fontsize=7)

    for h in range(H, len(axes)):
        axes[h].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# Part 2 visualization: plots training and validation loss to show learning/overfitting.
def plot_loss_curves(train_losses, val_losses):
    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses,   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training curves")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Token helpers                                                                 #
# --------------------------------------------------------------------------- #

# Part 2 visualization: converts token IDs back to readable labels for attention plots.
def ids_to_tokens(ids, tokenizer):
    """Convert a list/tensor of token IDs to a list of string tokens."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return [tokenizer.decode([i]).strip() or f"[{i}]" for i in ids]
