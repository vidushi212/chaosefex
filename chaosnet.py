import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from abide_chaosnet_lts import ChaosNetLTS

# =====================================
# CONFIG
# =====================================
CSV_PATH = "final_preprocessed.csv"
TARGET_COL = "DX_GROUP"

CHAOS_FEATURES = [
    "firing_rate",
    "firing_time",
    "energy",
    "entropy"
]

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv(CSV_PATH)

X = df[CHAOS_FEATURES].values.astype(np.float64)
y = df[TARGET_COL].values

print("Dataset loaded")
print("X shape:", X.shape)
print("Class distribution:", pd.Series(y).value_counts().to_dict())

# =====================================
# TRAIN / TEST SPLIT (simple, reproducible)
# =====================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================
# TRAIN CHAOSNET
# =====================================
model = ChaosNetLTS(
    n_neurons=30,
    learning_rate=0.01,
    max_epochs=200,
)

model.fit(X_train, y_train)

# =====================================
# EVALUATE
# =====================================
y_pred = model.predict(X_test)

print("\n======================================")
print("ChaosNet Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("======================================")
