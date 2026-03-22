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
        # Fix 1: snapshot v_c and v_o as copies to prevent stale gradient issues
        v_c = self.W_in[center_idx].copy()
        v_o = self.W_out[context_idx].copy()

        # Positive pair
        score_pos = sigmoid(np.dot(v_c, v_o))
        loss = -np.log(score_pos + 1e-7)
        grad_pos = score_pos - 1  # derivative of -log(sigmoid(x))

        # Fix 2: accumulate center word gradient instead of applying mid-loop
        grad_center = grad_pos * v_o

        # Update output embedding for positive pair using frozen v_c
        self.W_out[context_idx] -= lr * grad_pos * v_c

        # Negative samples
        for neg in neg_indices:
            v_n = self.W_out[neg].copy()
            score_neg = sigmoid(np.dot(v_c, v_n))
            loss += -np.log(1 - score_neg + 1e-7)
            grad_neg = score_neg  # derivative of -log(1 - sigmoid(x))

            # Accumulate gradient contribution to center word
            grad_center += grad_neg * v_n

            # Update negative output embedding using frozen v_c
            self.W_out[neg] -= lr * grad_neg * v_c

        # single update for center word after accumulating all gradients
        self.W_in[center_idx] -= lr * grad_center

        return loss