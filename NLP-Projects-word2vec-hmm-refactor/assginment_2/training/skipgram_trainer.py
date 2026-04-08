import random


def get_negative_samples(vocab_size, positive_idx, k=5):
    """
    Randomly sample k negative word indices different from positive_idx.
    """
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1 to draw negative samples.")

    negs = []
    while len(negs) < k:
        neg = random.randint(0, vocab_size - 1)
        if neg != positive_idx:
            negs.append(neg)
    return negs


def train_skipgram(
    pairs,
    vocab,
    model,
    neg_samples=5,
    lr=0.05,
    epochs=20,
    training_mode="sgd",
    batch_size=32,
    scale_mini_batch_lr=True,
    shuffle=True,
    verbose=True,
):
    """
    Train SkipGramNS model on precomputed (center_idx, context_idx) pairs.

    Parameters
    ----------
    pairs : list of tuples
        Each element is (center_idx, context_idx)
    vocab : dict
        word -> index mapping
    model : SkipGramNS
        model instance
    neg_samples : int
        number of negative samples per positive pair
    lr : float
        initial learning rate
    epochs : int
        number of training epochs
    training_mode : str
        "sgd" for one update per pair or "mini_batch" for batched updates
    batch_size : int
        number of pairs per mini-batch when training_mode="mini_batch"
    scale_mini_batch_lr : bool
        whether to scale mini-batch learning rate by sqrt(batch_size)
    shuffle : bool
        whether to shuffle pairs each epoch
    verbose : bool
        whether to print epoch losses

    Returns
    -------
    loss_history : list[float]
        average loss per epoch
    """
    if len(pairs) == 0:
        raise ValueError("pairs is empty")
    if training_mode not in {"sgd", "mini_batch"}:
        raise ValueError("training_mode must be either 'sgd' or 'mini_batch'.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    vocab_size = len(vocab)
    loss_history = []

    for epoch in range(epochs):
        # linear learning-rate decay
        lr_epoch = max(lr * (1.0 - epoch / epochs), lr * 1e-4)

        if shuffle:
            random.shuffle(pairs)

        total_loss = 0.0
        batch_lr = lr_epoch

        if training_mode == "mini_batch" and scale_mini_batch_lr:
            # We average batch gradients, so a mild sqrt(batch_size) scaling
            # is a practical default without becoming too aggressive.
            batch_lr = lr_epoch * min(batch_size ** 0.5, 4.0)

        if training_mode == "sgd":
            for center_idx, context_idx in pairs:
                neg_indices = get_negative_samples(
                    vocab_size=vocab_size,
                    positive_idx=context_idx,
                    k=neg_samples,
                )

                loss = model.train_step(
                    center_idx=center_idx,
                    context_idx=context_idx,
                    neg_indices=neg_indices,
                    lr=lr_epoch,
                )
                total_loss += loss
        else:
            for start_idx in range(0, len(pairs), batch_size):
                batch_pairs = pairs[start_idx:start_idx + batch_size]
                batch_neg_indices = [
                    get_negative_samples(
                        vocab_size=vocab_size,
                        positive_idx=context_idx,
                        k=neg_samples,
                    )
                    for _, context_idx in batch_pairs
                ]

                batch_loss = model.train_batch_step(
                    batch_pairs=batch_pairs,
                    batch_neg_indices=batch_neg_indices,
                    lr=batch_lr,
                    average=True,
                )
                total_loss += batch_loss * len(batch_pairs)

        avg_loss = total_loss / len(pairs)
        loss_history.append(avg_loss)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | mode={training_mode} | "
                f"lr={batch_lr:.5f} | loss={avg_loss:.4f}"
            )

    return loss_history
