#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "final_preprocessed.csv"
TARGET_COL = "DX_GROUP_y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Regularization (VERY IMPORTANT for stability)
REG_PARAM = 0.01   # try 0.05, 0.1, 0.2 if needed

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)
print("Dataset shape:", df.shape)

# =====================================================
# FEATURES / TARGET
# =====================================================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values

print("Feature matrix shape:", X.shape)
print("Class distribution:", dict(pd.Series(y).value_counts()))

# =====================================================
# HANDLE MISSING VALUES
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
# QUADRATIC DISCRIMINANT ANALYSIS
# =====================================================
qda = QuadraticDiscriminantAnalysis(reg_param=REG_PARAM)
qda.fit(X_train, y_train)

# =====================================================
# PREDICTION
# =====================================================
y_pred = qda.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================
acc = accuracy_score(y_test, y_pred)

print("\n======================================")
print("Quadratic Discriminant Analysis Results")
print("Accuracy:", round(acc, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("======================================")
