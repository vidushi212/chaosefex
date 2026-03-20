import numpy as np

class ChaosNet:
    def __init__(self, lr=0.05, epochs=300):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        y = (y == 2).astype(int)  # ASD=1, TC=0 (or swap if needed)

        for _ in range(self.epochs):
            logits = X @ self.w + self.b
            preds = 1 / (1 + np.exp(-logits))

            grad_w = X.T @ (preds - y) / len(y)
            grad_b = np.mean(preds - y)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        logits = X @ self.w + self.b
        return (logits > np.median(logits)).astype(int)
