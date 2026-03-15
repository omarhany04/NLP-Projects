import numpy as np
from utils import get_negative_samples

def train(model, dataset, vocab_size, epochs=2, lr=0.025, neg_samples=5):

    losses = []

    for epoch in range(epochs):

        total_loss = 0

        for target, context in dataset.data:

            negatives = get_negative_samples(context, vocab_size, neg_samples)

            loss, grad_v, grads_out = model.forward(target, context, negatives)

            model.update(target, grad_v, grads_out, lr)

            total_loss += loss

        avg_loss = total_loss / len(dataset)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        losses.append(float(avg_loss))
    return losses