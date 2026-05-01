"""
Assignment 3 - Part 2: BiLSTM encoder + LSTM decoder with additive attention.

Examples:
    python lstm_part2.py --smoke
    python lstm_part2.py --train --epochs 10
    python lstm_part2.py --checkpoint checkpoints/best_lstm.pt --sentence "je suis dure ."
"""
import argparse
import os
import sys

ASSIGNMENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ASSIGNMENT_DIR, "nmt_transformer"))

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk

from config import *
from data import get_dataloaders
from recurrent_model import RecurrentNMT
from recurrent_inference import (
    lstm_greedy_decode,
    lstm_beam_search,
)
from train import (
    train,
    save_checkpoint,
    load_checkpoint,
)
from utils import (
    compute_bleu_dataset,
    describe_device,
    get_device,
    ids_to_tokens,
    plot_attention,
    plot_loss_curves,
)


def clean(text):
    return " ".join(text.replace("\u2581", " ").split())


def build_model(device):
    return RecurrentNMT(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        embed_size=LSTM_EMBED_SIZE,
        hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
        pad_id=PAD_ID,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Train the LSTM model.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run one forward pass sanity check.")
    parser.add_argument("--epochs", type=int, default=LSTM_MAX_EPOCHS)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--bleu_samples", type=int, default=100,
                        help="Use 0 for the full test set.")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "best_lstm.pt"))
    parser.add_argument("--sentence", type=str, default="je suis dure .")
    parser.add_argument("--plot_attention", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"),
                        default="cuda",
                        help="Use cuda by default; choose auto to fall back to CPU.")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {describe_device(device)}")

    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = (
        get_dataloaders(
            DATA_PATH, TOKENIZER_FR_PATH, TOKENIZER_EN_PATH,
            batch_size=LSTM_BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
            bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
            pin_memory=(device.type == "cuda"),
        )
    )
    del test_loader

    model = build_model(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.smoke:
        src_ids, dec_input, targets = next(iter(train_loader))
        src_ids = src_ids.to(device)
        dec_input = dec_input.to(device)
        with torch.no_grad():
            logits, _, _, attn = model(src_ids, dec_input)
        print(f"src_ids : {tuple(src_ids.shape)}")
        print(f"dec_in  : {tuple(dec_input.shape)}")
        print(f"logits  : {tuple(logits.shape)}")
        print(f"attn    : {tuple(attn.shape)}")
        print(f"targets : {tuple(targets.shape)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)

    if args.train:
        train_losses, val_losses = train(
            model, train_loader, val_loader, optimizer,
            pad_id=PAD_ID, max_epochs=args.epochs, device=device,
            checkpoint_dir=CHECKPOINT_DIR, checkpoint_name="best_lstm.pt",
        )
        fig = plot_loss_curves(train_losses, val_losses)
        fig.savefig(os.path.join(os.path.dirname(__file__), "lstm_loss_curves.png"),
                    dpi=100, bbox_inches="tight")
        save_checkpoint(model, optimizer, args.epochs, val_losses[-1],
                        os.path.join(CHECKPOINT_DIR, "final_lstm.pt"))
    elif os.path.exists(args.checkpoint):
        load_checkpoint(model, optimizer=None, path=args.checkpoint, device=device)
    else:
        print(f"No checkpoint found at {args.checkpoint}; using current weights.")

    greedy_out, _ = lstm_greedy_decode(
        model, args.sentence, src_tokenizer, tgt_tokenizer,
        MAX_SEQ_LEN, BOS_ID, EOS_ID, PAD_ID, device,
    )
    beam_out, pred_tokens, attn = lstm_beam_search(
        model, args.sentence, src_tokenizer, tgt_tokenizer,
        args.beam_size, MAX_SEQ_LEN, BOS_ID, EOS_ID, PAD_ID, device,
    )
    print(f"FR     : {args.sentence}")
    print(f"Greedy : {clean(greedy_out)}")
    print(f"Beam   : {clean(beam_out)}")

    ds = load_from_disk(DATA_PATH)
    max_samples = None if args.bleu_samples == 0 else args.bleu_samples
    bleu, _, _ = compute_bleu_dataset(
        model, ds["test"], src_tokenizer, tgt_tokenizer,
        beam_size=args.beam_size, max_len=MAX_SEQ_LEN,
        bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, device=device,
        max_samples=max_samples, decode_fn=lstm_beam_search,
    )
    label = "full test set" if max_samples is None else f"{max_samples} samples"
    print(f"BLEU ({label}): {bleu.score:.2f}")

    if args.plot_attention and attn is not None:
        src_ids = src_tokenizer.encode(args.sentence, add_special_tokens=False)
        src_ids = src_ids[: MAX_SEQ_LEN - 1] + [EOS_ID]
        src_labels = ids_to_tokens(src_ids, src_tokenizer)
        tgt_labels = ids_to_tokens(pred_tokens, tgt_tokenizer)
        attn_slice = attn[:, :, :len(tgt_labels), :len(src_labels)]
        fig = plot_attention(
            attn_slice, src_labels, tgt_labels,
            title="LSTM Additive Attention",
        )
        out_path = os.path.join(os.path.dirname(__file__), "lstm_attention.png")
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        print(f"Attention plot saved to {out_path}")


if __name__ == "__main__":
    main()
