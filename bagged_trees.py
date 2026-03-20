#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from evaluation_utils import evaluate_model, print_results

# =====================================================
# LOAD CLEAN DATASET
# =====================================================
df = pd.read_csv("dataset_no_combat.csv")

print("Dataset shape:", df.shape)
print("Class distribution:", df["DX_GROUP"].value_counts().to_dict())

# =====================================================
# REMOVE SITE COLUMN (NOT A FEATURE)
# =====================================================
if "SITE_ID_y" in df.columns:
    df = df.drop(columns=["SITE_ID_y"])

# =====================================================
# SPLIT X / y
# =====================================================
X = df.drop(columns=["DX_GROUP"])
y = df["DX_GROUP"]

# Convert to numpy
X = X.values
y = y.values

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
# BAGGED TREES MODEL
# =====================================================
model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=200,
    random_state=42,
    n_jobs=-1
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
