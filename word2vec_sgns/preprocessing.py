from collections import Counter
import numpy as np


def build_vocab(sentences, min_count=1):

    counter = Counter()

    for sent in sentences:
        counter.update(sent)

    vocab = [w for w, c in counter.items() if c >= min_count]

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return vocab, word2idx, idx2word, counter


def sentences_to_indices(sentences, word2idx):

    data = []

    for sent in sentences:
        idxs = [word2idx[w] for w in sent if w in word2idx]
        data.append(idxs)

    return data
