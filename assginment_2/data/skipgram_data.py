def combine_split_sentences(*sentence_groups):
    """
    Flatten one or more sentence collections into a single list.
    Useful for building skip-gram pairs from train/validation/test splits.
    """
    combined = []
    for group in sentence_groups:
        if not group:
            continue
        combined.extend(group)
    return combined


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
