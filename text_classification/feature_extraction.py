# feature_extraction.py
import numpy as np
from scipy.sparse import lil_matrix

def vectorize_dataframe_sparse(df, bigram_to_idx):
    n_rows = len(df)
    n_cols = len(bigram_to_idx)
    vectors = lil_matrix((n_rows, n_cols), dtype=int)
    
    for row_idx, tokens in enumerate(df['tokens']):
        for i in range(len(tokens)-1):
            bigram = (tokens[i], tokens[i+1])
            if bigram in bigram_to_idx:
                vectors[row_idx, bigram_to_idx[bigram]] = 1
                
    return vectors.tocsr()

def build_bigram_vocab(token_lists):
    bigram_set = set()
    for tokens in token_lists:
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        bigram_set.update(bigrams)
    bigram_to_idx = {bigram: idx for idx, bigram in enumerate(sorted(bigram_set))}
    return bigram_to_idx

def vectorize_text(tokens_list, bigram_to_idx):
    vector = np.zeros(len(bigram_to_idx), dtype=int)
    for i in range(len(tokens_list)-1):
        bigram = (tokens_list[i], tokens_list[i+1])
        if bigram in bigram_to_idx:
            vector[bigram_to_idx[bigram]] = 1
    return vector

def vectorize_dataframe(df, bigram_to_idx):
    vectors = np.array([vectorize_text(tokens, bigram_to_idx) for tokens in df['tokens']])
    return vectors