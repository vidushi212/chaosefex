import numpy as np

def extract_chaos_features(ts):
    """
    ts: (T × ROIs) numpy array
    returns: [firing_rate, firing_time, energy, entropy]
    """

    # transpose → ROIs × Time
    X = ts.T

    # z-score per ROI (important for chaos)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # squash to (0,1)
    X = 1 / (1 + np.exp(-X))

    # logistic map
    r = 4.0
    for _ in range(3):
        X = r * X * (1 - X)

    # firing threshold
    thresh = 0.5
    fired = (X > thresh).astype(int)

    firing_rate = fired.mean()
    firing_time = fired.sum(axis=1).mean()
    energy = np.mean(X ** 2)

    # entropy
    p = np.clip(X.mean(axis=1), 1e-8, 1)
    entropy = -np.mean(p * np.log(p))

    return np.array([firing_rate, firing_time, energy, entropy])
