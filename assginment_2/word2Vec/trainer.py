import random
import numpy as np
from dataset import create_skipgram_pairs, get_negative_samples
from model import SkipGramNS


def train_skipgram(
    pairs,
    vocab,
    model,
    embedding_dim=50,
    window_size=2,
    neg_samples=5,
    lr=0.05,
    epochs=20,
):
    """
    Train an existing SkipGramNS model on tokenized sentences.

    Parameters:
    - pairs: list of (center, context) tuples
    - vocab: word -> index mapping
    - model: SkipGramNS object (already created)
    - window_size: context window size
    - neg_samples: number of negative samples
    - lr: learning rate
    - epochs: number of epochs

    Returns:
    - loss_history: list of average loss per epoch
    """
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(pairs)
        for center, context in pairs:
            negs = get_negative_samples(len(vocab), context, k=neg_samples)
            loss = model.forward_backward(center, context, negs, lr)
            total_loss += loss

        loss_history.append(total_loss / len(pairs))

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(pairs):.4f}")

    return loss_history
