import numpy as np
from utils import sigmoid


class Word2VecSGNS:

    def __init__(self, vocab_size, embed_dim):

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01

    def forward(self, target, context, negatives):

        v = self.W_in[target]
        u_pos = self.W_out[context]

        score_pos = sigmoid(np.dot(v, u_pos))
        loss = -np.log(score_pos + 1e-10)

        grad_v = (score_pos - 1) * u_pos
        grad_u_pos = (score_pos - 1) * v

        grads_out = {context: grad_u_pos}

        for neg in negatives:

            u_neg = self.W_out[neg]

            score_neg = sigmoid(np.dot(v, u_neg))
            loss -= np.log(1 - score_neg + 1e-10)

            grad = score_neg * u_neg

            grad_v += grad
            grads_out[neg] = score_neg * v

        return loss, grad_v, grads_out

    def update(self, target, grad_v, grads_out, lr):

        self.W_in[target] -= lr * grad_v

        for idx, grad in grads_out.items():
            self.W_out[idx] -= lr * grad
