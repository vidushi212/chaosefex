import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from chaos_features import extract_chaos_features
from chaosnet_model import ChaosNet

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "/Users/vidushikhandelwal/Downloads/rois"     # folder with .1D files
LABEL_MAP = {         # YOU MUST ADJUST THIS
    "CMU": 2,         # control
    "Pitt": 1,        # ASD
}

# =====================================================
# LOAD DATA
# =====================================================
X, y = [], []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".1D"):
        continue

    site = fname.split("_")[0]
    if site not in LABEL_MAP:
        continue

    path = os.path.join(DATA_DIR, fname)
    ts = np.loadtxt(path)

    if ts.ndim != 2 or ts.shape[0] < 50:
        continue

    feats = extract_chaos_features(ts)

    X.append(feats)
    y.append(LABEL_MAP[site])

X = np.array(X)
y = np.array(y)

print("Dataset loaded")
print("X shape:", X.shape)
print("Class distribution:", dict(zip(*np.unique(y, return_counts=True))))

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# =====================================================
# CHAOSNET
# =====================================================
model = ChaosNet()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n======================================")
print("ChaosNet Results")
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))
print("======================================")
