# logistic_regression_scratch.py
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, input_dim, num_classes, lr=0.1):
        self.W = np.zeros((input_dim, num_classes))
        self.b = np.zeros(num_classes)
        self.lr = lr

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot(self, y, num_classes):
        oh = np.zeros((len(y), num_classes))
        oh[np.arange(len(y)), y] = 1
        return oh

    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X, y, epochs=10, batch_size=32):
        y_one_hot = self.one_hot(y, self.W.shape[1])
        n_samples = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuff = X[indices]
            y_shuff = y_one_hot[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuff[start:end]
                y_batch = y_shuff[start:end]

                # Forward
                y_pred = self.predict_proba(X_batch)

                # Gradients
                grad_W = X_batch.T @ (y_pred - y_batch) / X_batch.shape[0]
                grad_b = np.mean(y_pred - y_batch, axis=0)

                # Update
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b