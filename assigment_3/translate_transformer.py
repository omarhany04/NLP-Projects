"""
Transformer translation tester — run this file to translate French sentences interactively.

Usage:
    python translate_transformer.py
    python translate_transformer.py --show_attn
    python translate_transformer.py --sentence "je suis dure ."
"""
import sys, os, argparse
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'nmt_transformer'))

import torch
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast

from config import *
from model import TransformerNMT
from train import load_checkpoint
from inference import beam_search
from utils import plot_attention, ids_to_tokens


def load_model(device):
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_FR_PATH)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_EN_PATH)

    model = TransformerNMT(
        src_vocab_size     = SRC_VOCAB_SIZE,
        tgt_vocab_size     = TGT_VOCAB_SIZE,
        embed_dim          = HIDDEN_SIZE,
        max_seq_len        = MAX_SEQ_LEN,
        num_encoder_layers = NUM_ENCODER_LAYERS,
        num_decoder_layers = NUM_DECODER_LAYERS,
        num_heads          = NUM_HEADS,
        intermediate_dim   = INTERMEDIATE_SIZE,
        dropout            = DROPOUT,
        pad_id             = PAD_ID,
    ).to(device)

    ckpt = os.path.join(CHECKPOINT_DIR, 'best_transformer.pt')
    if not os.path.exists(ckpt):
        print(f"ERROR: No checkpoint found at {ckpt}")
        print("Train the model first by running the notebook (Section 3).")
        sys.exit(1)

    load_checkpoint(model, optimizer=None, path=ckpt, device=device)
    model.eval()
    return model, src_tokenizer, tgt_tokenizer


def translate(sentence, model, src_tokenizer, tgt_tokenizer, device,
              beam_size=4, show_attn=False):
    translation, pred_tokens, cross_attn = beam_search(
        model, sentence, src_tokenizer, tgt_tokenizer,
        beam_size, MAX_SEQ_LEN, BOS_ID, EOS_ID, PAD_ID, device,
    )
    translation = " ".join(translation.replace("▁", " ").split())
    print(f"FR : {sentence}")
    print(f"EN : {translation}")

    if show_attn and cross_attn is not None:
        matplotlib = plt.get_backend()
        src_ids   = src_tokenizer.encode(sentence, add_special_tokens=False) + [EOS_ID]
        src_labels = ids_to_tokens(src_ids, src_tokenizer) + ['</s>']
        tgt_labels = ids_to_tokens(pred_tokens, tgt_tokenizer)
        attn_slice = cross_attn[:, :, :len(tgt_labels), :len(src_labels)]
        fig = plot_attention(attn_slice, src_labels, tgt_labels,
                             title=f'Cross-Attention\n"{sentence}" → "{translation}"')
        plt.show()

    return translation


def interactive_loop(model, src_tokenizer, tgt_tokenizer, device, show_attn):
    print("\nTransformer NMT — French → English")
    print("Type a French sentence and press Enter.  Type 'quit' to exit.\n")
    while True:
        try:
            sentence = input("FR> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not sentence:
            continue
        if sentence.lower() in ("quit", "exit", "q"):
            break
        translate(sentence, model, src_tokenizer, tgt_tokenizer, device, show_attn=show_attn)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence",   type=str, default=None,
                        help="Translate a single sentence and exit.")
    parser.add_argument("--show_attn",  action="store_true",
                        help="Display cross-attention heatmap after each translation.")
    parser.add_argument("--beam_size",  type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, src_tok, tgt_tok = load_model(device)

    if args.sentence:
        translate(args.sentence, model, src_tok, tgt_tok, device,
                  beam_size=args.beam_size, show_attn=args.show_attn)
    else:
        interactive_loop(model, src_tok, tgt_tok, device, show_attn=args.show_attn)