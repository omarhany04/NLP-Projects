import numpy as np


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def most_similar(word, vocab, idx_to_word, W_in, topk=5):
    """
    Find the top-k most similar words to the given word using cosine similarity
    based on the trained embeddings (model.W_in).

    Args:
        word (str): target word
        topk (int): number of similar words to return

    Returns:
        List of tuples: [(word, similarity), ...]
    """
    # Check if the word is in the vocabulary
    if word not in vocab:
        print(f"Word '{word}' not in vocabulary")
        return []

    # Get the index of the target word
    idx = vocab[word]

    # Get the embedding vector for the target word
    vec = W_in[idx]

    sims = []

    # Compare with all other words in the vocabulary
    for i in range(len(vocab)):
        if i == idx:
            continue  # Skip the target word itself

        other_vec = W_in[i]

        # Cosine similarity = dot(a,b) / (||a|| * ||b||)
        similarity = np.dot(vec, other_vec) / (
            np.linalg.norm(vec) * np.linalg.norm(other_vec) + 1e-9
        )

        sims.append((idx_to_word[i], similarity))

    # Sort by similarity in descending order
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    # Return top-k most similar words
    return sims[:topk]


def word_analogy(a, b, c, W_in, vocab, idx_to_word, topk=3):
    """
    Solve analogy: a : b :: c : ?
    Example: king - man + woman ≈ queen
    """

    # Check words exist
    for word in [a, b, c]:
        if word not in vocab:
            raise ValueError(f"{word} not in vocabulary")

    # Get vectors
    v_a = W_in[vocab[a]]
    v_b = W_in[vocab[b]]
    v_c = W_in[vocab[c]]

    # Vector arithmetic
    target_vec = v_b - v_a + v_c

    # Compute cosine similarities
    sims = []
    for i in range(len(W_in)):
        vec = W_in[i]
        sim = np.dot(target_vec, vec) / (
            np.linalg.norm(target_vec) * np.linalg.norm(vec)
        )
        sims.append(sim)

    sims = np.array(sims)

    # Sort by similarity (descending)
    sorted_idx = np.argsort(-sims)

    results = []
    for idx in sorted_idx:
        word = idx_to_word[idx]
        if word not in [a, b, c]:
            results.append((word, sims[idx]))
        if len(results) == topk:
            break

    return results


def save_embeddings(W_in, idx_to_word, path="embeddings.txt"):
    with open(path, "w") as f:
        for i, vec in enumerate(W_in):
            word = idx_to_word[i]
            vec_str = " ".join(map(str, vec))
            f.write(f"{word} {vec_str}\n")


def load_embeddings(path="embeddings.txt"):
    word_to_vec = {}
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vec = np.array(list(map(float, tokens[1:])))
            word_to_vec[word] = vec
    return word_to_vec
