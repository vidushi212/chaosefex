import numpy as np


class ChaosNetLTS:
    """
    ChaosNet classifier (Low-dimensional Temporal Spiking model)
    Designed for chaos-derived features ONLY
    """

    def __init__(
        self,
        n_neurons=30,
        learning_rate=0.01,
        max_epochs=150,
        random_state=42,
    ):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.weights = None
        self.bias = None

    # --------------------------------------------------
    # Logistic chaotic map
    # --------------------------------------------------
    def _chaotic_map(self, x):
        r = 3.9
        return r * x * (1 - x)

    # --------------------------------------------------
    # Chaos transformation (CRITICAL FIX)
    # --------------------------------------------------
    def _transform(self, X):
        # Min–max normalize INSIDE the model (not preprocessing)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

        # Keep values in chaotic regime
        X = np.clip(X, 1e-6, 1 - 1e-6)

        # Apply chaotic iterations
        for _ in range(3):
            X = self._chaotic_map(X)

        return X

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------
    def fit(self, X, y):
        np.random.seed(self.random_state)

        # Convert ABIDE labels: ASD=1, TC=2 → {1,0}
        y_bin = (y == 1).astype(int)

        Xc = self._transform(X)

        n_samples, n_features = Xc.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0.0

        for _ in range(self.max_epochs):
            logits = np.dot(Xc, self.weights) + self.bias

            # 🔴 CRITICAL FIX: adaptive threshold
            threshold = logits.mean()
            preds = (logits > threshold).astype(int)

            error = y_bin - preds

            self.weights += self.learning_rate * np.dot(Xc.T, error) / n_samples
            self.bias += self.learning_rate * error.mean()

        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------
    def predict(self, X):
        Xc = self._transform(X)
        logits = np.dot(Xc, self.weights) + self.bias

        threshold = logits.mean()
        preds = (logits > threshold).astype(int)

        # Convert back to ABIDE labels
        return np.where(preds == 1, 1, 2)
