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

def is_valid_word(word, min_len=2):
    """
    Return True if the word:
    - contains only alphabetic characters (no numbers/dates/special chars)
    - has length >= min_len
    - is not a stopword
    """
    return (
        re.match(r'^[a-zA-Z]+$', word) is not None
        and len(word) >= min_len
        and word.lower() not in stop_words
    )

def tokenize(corpus):
    """
    Tokenize a list of sentences (corpus)
    
    Steps:
    1. Lowercase
    2. Remove punctuation, numbers, dates, special characters
    3. Remove stopwords
    4. Lemmatize words
    5. Keep only valid tokens (alphabetic, min_len >= 2)
    
    Input:
        corpus: list of sentences (each sentence is a list of tokens)
    Output:
        tokenized_corpus: list of tokenized sentences
    """
    tokenized_corpus = []
    for sentence in corpus:
        tokens = [
            lemmatizer.lemmatize(word.lower())  # lowercase + lemmatize
            for word in sentence
            if word not in string.punctuation and is_valid_word(word)
        ]
        tokenized_corpus.append(tokens)
    return tokenized_corpus

def build_vocab(tokenized_sentences, min_count=5):
    """
    Build vocabulary and mappings from tokenized sentences.
    
    Steps:
    1. Flatten tokenized corpus
    2. Count word frequencies
    3. Keep only words:
       - appearing at least `min_count` times
       - passing is_valid_word check
    4. Assign unique indices to words
    
    Returns:
        vocab: dict word -> index
        idx_to_word: dict index -> word
    """
    all_words = [word for sent in tokenized_sentences for word in sent]
    word_counts = Counter(all_words)

    # Keep only frequent and valid words
    filtered_words = [
        w for w, c in word_counts.items() if c >= min_count and is_valid_word(w)
    ]

    vocab = {w: i for i, w in enumerate(filtered_words)}
    idx_to_word = {i: w for w, i in vocab.items()}

    return vocab, idx_to_word