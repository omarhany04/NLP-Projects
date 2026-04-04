from collections import Counter
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def tokenize(corpus):
    """
    Tokenize sentences for NER embeddings.
    Keeps proper nouns, numbers, dates, and stopwords.
    """
    tokenized_corpus = []
    for sentence in corpus:
        tokens = []
        for word in sentence:
        #     if word in string.punctuation:
        #         continue
            # Keep original casing, numbers, and alphanumerics
            clean_word = word.strip()
            if clean_word:
                tokens.append(clean_word)
        tokenized_corpus.append(tokens)
    return tokenized_corpus


def build_vocab(tokenized_sentences, min_count=2):
    """
    Build vocabulary and mappings from tokenized sentences.

    Steps:
    1. Flatten tokenized corpus
    2. Count word frequencies
    3. Keep only words:
       - appearing at least `min_count` times
    4. Assign unique indices to words

    Returns:
        vocab: dict word -> index
        idx_to_word: dict index -> word
    """
    all_words = [word for sent in tokenized_sentences for word in sent]
    word_counts = Counter(all_words)

    # Keep only frequent
    filtered_words = [
        w for w, c in word_counts.items() if c >= min_count
    ]

    vocab = {w: i for i, w in enumerate(filtered_words)}
    idx_to_word = {i: w for w, i in vocab.items()}

    return vocab, idx_to_word

