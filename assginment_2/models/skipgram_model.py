import numpy as np
from utils.numerical_utils import sigmoid


class SkipGramNS:
    def __init__(self, vocab_size, embedding_dim=50, seed=None):
        """
        Skip-Gram with Negative Sampling

        W_in  : input embeddings for center words
        W_out : output embeddings for context words
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        if seed is not None:
            np.random.seed(seed)

        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01

    def forward(self, center_idx, context_idx, neg_indices):
        """
        Forward pass only.
        Computes loss and stores everything needed for backward pass.
        """
        v_c = self.W_in[center_idx].copy()              # (D,)
        v_o = self.W_out[context_idx].copy()            # (D,)
        v_negs = self.W_out[neg_indices].copy()         # (K, D)

        # Positive score
        pos_logit = np.dot(v_c, v_o)                    # scalar
        pos_prob = sigmoid(pos_logit)                   # scalar

        # Negative scores
        neg_logits = v_negs @ v_c                       # (K,)
        neg_probs = sigmoid(neg_logits)                 # (K,)

        eps = 1e-10
        loss_pos = -np.log(pos_prob + eps)
        loss_neg = -np.sum(np.log(1.0 - neg_probs + eps))
        loss = loss_pos + loss_neg

        cache = {
            "center_idx": center_idx,
            "context_idx": context_idx,
            "neg_indices": np.array(neg_indices, dtype=np.int64),
            "v_c": v_c,
            "v_o": v_o,
            "v_negs": v_negs,
            "pos_prob": pos_prob,
            "neg_probs": neg_probs,
        }

        return loss, cache

    def backward(self, cache, lr=0.05):
        """
        Backward pass + SGD update.
        Uses cached frozen embeddings from forward.
        """
        center_idx = cache["center_idx"]
        context_idx = cache["context_idx"]
        neg_indices = cache["neg_indices"]

        v_c = cache["v_c"]              # (D,)
        v_o = cache["v_o"]              # (D,)
        v_negs = cache["v_negs"]        # (K, D)

        pos_prob = cache["pos_prob"]    # scalar
        neg_probs = cache["neg_probs"]  # (K,)

        # d/dx[-log(sigmoid(x))] = sigmoid(x) - 1
        grad_pos = pos_prob - 1.0       # scalar

        # d/dx[-log(1 - sigmoid(x))] = sigmoid(x)
        grad_negs = neg_probs           # (K,)

        # Gradient wrt center embedding
        grad_center = grad_pos * v_o + np.sum(grad_negs[:, None] * v_negs, axis=0)

        # Gradient wrt positive output embedding
        grad_out_pos = grad_pos * v_c

        # Gradient wrt negative output embeddings
        grad_out_negs = grad_negs[:, None] * v_c[None, :]

        # Apply updates
        self.W_in[center_idx] -= lr * grad_center
        self.W_out[context_idx] -= lr * grad_out_pos

        # Important: repeated negative indices should accumulate
        np.add.at(self.W_out, neg_indices, -lr * grad_out_negs)

    def train_step(self, center_idx, context_idx, neg_indices, lr=0.05):
        """
        One full SGD step.
        """
        loss, cache = self.forward(center_idx, context_idx, neg_indices)
        self.backward(cache, lr)
        return loss

    def get_input_embeddings(self):
        return self.W_in

    def get_output_embeddings(self):
        return self.W_out
