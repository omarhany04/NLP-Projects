import numpy as np
from utils import sigmoid


class SkipGramNS:
    def __init__(self, vocab_size, embedding_dim=50):
        """
        Initialize input and output embedding matrices.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Input embedding (center word)
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        # Output embedding (context word)
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward_backward(self, center_idx, context_idx, neg_indices, lr=0.05):
        """
        Perform forward pass, compute loss, and update embeddings with SGD.
        """
        v_c = self.W_in[center_idx]  # Center embedding
        v_o = self.W_out[context_idx]  # Context embedding

        # Positive pair
        score_pos = sigmoid(np.dot(v_c, v_o))
        loss = -np.log(score_pos + 1e-7)
        grad_pos = score_pos - 1  # derivative of -log(sigmoid(x))

        # Update embeddings for positive pair
        self.W_in[center_idx] -= lr * grad_pos * v_o
        self.W_out[context_idx] -= lr * grad_pos * v_c

        # Negative samples
        for neg in neg_indices:
            v_n = self.W_out[neg]
            score_neg = sigmoid(np.dot(v_c, v_n))
            loss += -np.log(1 - score_neg + 1e-7)
            grad_neg = score_neg  # derivative of -log(1-sigmoid(x))
            self.W_in[center_idx] -= lr * grad_neg * v_n
            self.W_out[neg] -= lr * grad_neg * v_c

        return loss
