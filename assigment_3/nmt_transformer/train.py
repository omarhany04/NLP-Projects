import torch
import torch.nn as nn


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for src_ids, dec_input, targets in loader:
        src_ids   = src_ids.to(device)
        dec_input = dec_input.to(device)
        targets   = targets.to(device)

        logits, _, _, _ = model(src_ids, dec_input)   # (B, T, V)
        B, T, V = logits.shape

        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients during early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src_ids, dec_input, targets in loader:
        src_ids   = src_ids.to(device)
        dec_input = dec_input.to(device)
        targets   = targets.to(device)

        logits, _, _, _ = model(src_ids, dec_input)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), targets.reshape(B * T))
        total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader, optimizer, pad_id,
          max_epochs, device, checkpoint_dir=None):
    """Full training loop.  Returns (train_losses, val_losses)."""
    import os
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:2d}/{max_epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if checkpoint_dir is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(checkpoint_dir, "best_transformer.pt"),
            )

    return train_losses, val_losses


# --------------------------------------------------------------------------- #
# Checkpoint helpers                                                            #
# --------------------------------------------------------------------------- #

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":             val_loss,
    }, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path}  "
          f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")
    return checkpoint["epoch"], checkpoint["val_loss"]
