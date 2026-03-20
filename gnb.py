#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "final_preprocessed.csv"
TARGET_COL = "DX_GROUP_y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)
print("Dataset shape:", df.shape)

# =====================================================
# SEPARATE FEATURES & TARGET
# =====================================================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values

print("Feature matrix shape:", X.shape)
print("Class distribution:", dict(pd.Series(y).value_counts()))

# =====================================================
# IMPUTE MISSING VALUES (REQUIRED FOR GNB)
# =====================================================
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# =====================================================
# TRAIN / TEST SPLIT (STRATIFIED)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# =====================================================
# GAUSSIAN NAIVE BAYES
# =====================================================
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# =====================================================
# PREDICTION
# =====================================================
y_pred = gnb.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================
acc = accuracy_score(y_test, y_pred)

print("\n======================================")
print("Gaussian Naïve Bayes Results")
print("Accuracy:", round(acc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("======================================")
