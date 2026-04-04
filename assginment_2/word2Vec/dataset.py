import random


def create_skipgram_pairs(tokenized_sentences, vocab, window_size=2):
    """
    For each center word, create context pairs within the window.
    """
    pairs = []
    for sentence in tokenized_sentences:
        indices = [vocab[w] for w in sentence if w in vocab]  # Only keep words in vocab
        for center_pos, center in enumerate(indices):
            start = max(0, center_pos - window_size)
            end = min(len(indices), center_pos + window_size + 1)
            for pos in range(start, end):
                if pos != center_pos:
                    context = indices[pos]
                    pairs.append((center, context))
    return pairs


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
