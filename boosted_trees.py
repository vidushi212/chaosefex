#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from evaluation_utils import evaluate_model, print_results

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("final_clean_for_ml.csv")

print("Dataset shape:", df.shape)
print("Class distribution:", df["DX_GROUP"].value_counts().to_dict())

# Remove site column (avoid leakage)
if "SITE_ID_y" in df.columns:
    df = df.drop(columns=["SITE_ID_y"])

# =====================================================
# SPLIT FEATURES / TARGET
# =====================================================
X = df.drop(columns=["DX_GROUP"])
y = df["DX_GROUP"]

# 🔴 SAFETY CHECK — ensure no NaNs exist
if X.isna().sum().sum() > 0:
    raise RuntimeError(
        "Dataset still contains NaNs. Fix preprocessing first."
    )

X = X.values
y = y.values

print("Feature matrix shape:", X.shape)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# GRADIENT BOOSTING MODEL
# =====================================================
model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# PREDICT
# =====================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# =====================================================
# EVALUATE
# =====================================================
results = evaluate_model(y_test, y_pred, y_proba)
print_results(results)

