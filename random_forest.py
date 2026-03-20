#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from evaluation_utils import evaluate_model, print_results

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("final_clean_for_ml.csv")

print("Dataset shape:", df.shape)
print("Class distribution:", df["DX_GROUP"].value_counts().to_dict())

# =====================================================
# REMOVE SITE COLUMN (NOT USED FOR TRAINING)
# =====================================================
if "SITE_ID_y" in df.columns:
    df = df.drop(columns=["SITE_ID_y"])

# =====================================================
# SPLIT X / y
# =====================================================
X = df.drop(columns=["DX_GROUP"])
y = df["DX_GROUP"]

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
# RANDOM FOREST MODEL
# =====================================================
model = RandomForestClassifier(
    n_estimators=500,        # large forest for stability
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",     # best for tabular data
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# =====================================================
# EVALUATION
# =====================================================
results = evaluate_model(y_test, y_pred, y_proba)
print_results(results)
