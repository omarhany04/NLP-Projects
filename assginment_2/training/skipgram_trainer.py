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

    vocab_size = len(vocab)
    loss_history = []

    for epoch in range(epochs):
        # linear learning-rate decay
        lr_epoch = max(lr * (1.0 - epoch / epochs), lr * 1e-4)

        if shuffle:
            random.shuffle(pairs)

        total_loss = 0.0

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

        avg_loss = total_loss / len(pairs)
        loss_history.append(avg_loss)

        if verbose:
            print(f"Epoch {epoch + 1:02d}/{epochs} | lr={lr_epoch:.5f} | loss={avg_loss:.4f}")

    return loss_history
