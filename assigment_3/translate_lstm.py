# This file is for live testing from the terminal.
import argparse
import os
import sys

ASSIGNMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASSIGNMENT_DIR)
sys.path.insert(0, os.path.join(ASSIGNMENT_DIR, "nmt_transformer"))
sys.path.insert(0, os.path.join(ASSIGNMENT_DIR, "nmt_lstm"))

import torch
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast

from config import *
from recurrent_model import RecurrentNMT
from recurrent_inference import lstm_beam_search
from train import load_checkpoint
from utils import describe_device, get_device, ids_to_tokens, plot_attention


# Part 2 live test: cleans tokenizer boundary markers so the printed English sentence is readable.
def clean(text):
    return " ".join(text.replace("\u2581", " ").split())


# Part 2 live test: rebuilds the LSTM model, loads the saved checkpoint, and returns tokenizers.
def load_model(device, checkpoint_path):
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_FR_PATH)
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_EN_PATH)

    model = RecurrentNMT(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        embed_size=LSTM_EMBED_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        pad_id=PAD_ID,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        print("Train the model first with: python lstm_part2.py --train")
        sys.exit(1)

    load_checkpoint(model, optimizer=None, path=checkpoint_path, device=device)
    model.eval()
    return model, src_tokenizer, tgt_tokenizer


# Part 2 live test: translates one French sentence and optionally displays its attention heatmap.
def translate(sentence, model, src_tokenizer, tgt_tokenizer, device,
              beam_size=4, show_attn=False):
    translation, pred_tokens, attn = lstm_beam_search(
        model, sentence, src_tokenizer, tgt_tokenizer,
        beam_size, MAX_SEQ_LEN, BOS_ID, EOS_ID, PAD_ID, device,
    )
    translation = clean(translation)
    print(f"FR : {sentence}")
    print(f"EN : {translation}")

    if show_attn and attn is not None:
        src_ids = src_tokenizer.encode(sentence, add_special_tokens=False)
        src_ids = src_ids[: MAX_SEQ_LEN - 1] + [EOS_ID]
        src_labels = ids_to_tokens(src_ids, src_tokenizer)
        tgt_labels = ids_to_tokens(pred_tokens, tgt_tokenizer)
        attn_slice = attn[:, :, :len(tgt_labels), :len(src_labels)]
        fig = plot_attention(
            attn_slice, src_labels, tgt_labels,
            title=f'LSTM Additive Attention: "{sentence}" -> "{translation}"',
        )
        plt.show()

    return translation


# Part 2 live test: repeatedly accepts French input from the terminal until the user quits.
def interactive_loop(model, src_tokenizer, tgt_tokenizer, device, show_attn,
                     beam_size):
    print("\nLSTM NMT - French -> English")
    print("Type a French sentence and press Enter. Type 'quit' to exit.\n")
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
        translate(sentence, model, src_tokenizer, tgt_tokenizer, device,
                  beam_size=beam_size, show_attn=show_attn)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default=None)
    parser.add_argument("--show_attn", action="store_true")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "best_lstm.pt"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"),
                        default="cuda",
                        help="Use cuda by default; choose auto to fall back to CPU.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {describe_device(device)}")

    model, src_tok, tgt_tok = load_model(device, args.checkpoint)

    if args.sentence:
        translate(args.sentence, model, src_tok, tgt_tok, device,
                  beam_size=args.beam_size, show_attn=args.show_attn)
    else:
        interactive_loop(model, src_tok, tgt_tok, device, args.show_attn,
                         args.beam_size)
