import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_negative_samples(target, vocab_size, k):

    negatives = []

    while len(negatives) < k:

        sample = np.random.randint(0, vocab_size)

        if sample != target:
            negatives.append(sample)

    return negatives
