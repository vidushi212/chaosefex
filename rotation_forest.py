#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from evaluation_utils import evaluate_model, print_results

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("final_clean_for_ml.csv")

print("Dataset shape:", df.shape)
print("Class distribution:", df["DX_GROUP"].value_counts().to_dict())

# Remove site column (we don't want site leakage)
if "SITE_ID_y" in df.columns:
    df = df.drop(columns=["SITE_ID_y"])

X = df.drop(columns=["DX_GROUP"]).values
y = df["DX_GROUP"].values

print("Feature matrix shape:", X.shape)

# =====================================================
# 🔥 FIX: IMPUTE MISSING VALUES
# =====================================================
print("\nMissing values before imputation:", np.isnan(X).sum())

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

print("Missing values after imputation:", np.isnan(X).sum())

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
# ROTATION FOREST IMPLEMENTATION
# =====================================================
class RotationForest:
    def __init__(self, n_estimators=25, n_feature_subsets=3):
        self.n_estimators = n_estimators
        self.n_feature_subsets = n_feature_subsets
        self.trees = []
        self.rotation_matrices = []

    def _create_rotation_matrix(self, X):
        n_samples, n_features = X.shape
        feature_indices = np.arange(n_features)
        np.random.shuffle(feature_indices)

        subsets = np.array_split(feature_indices, self.n_feature_subsets)
        R = np.zeros((n_features, n_features))

        for subset in subsets:
            pca = PCA()
            X_subset = X[:, subset]
            pca.fit(X_subset)
            R[np.ix_(subset, subset)] = pca.components_

        return R

    def fit(self, X, y):
        self.trees = []
        self.rotation_matrices = []

        for _ in range(self.n_estimators):
            R = self._create_rotation_matrix(X)
            X_rot = X @ R

            tree = DecisionTreeClassifier()
            tree.fit(X_rot, y)

            self.trees.append(tree)
            self.rotation_matrices.append(R)

    def predict(self, X):
        preds = []
        for tree, R in zip(self.trees, self.rotation_matrices):
            preds.append(tree.predict(X @ R))

        preds = np.array(preds)
        return np.round(preds.mean(axis=0)).astype(int)

    def predict_proba(self, X):
        probas = []
        for tree, R in zip(self.trees, self.rotation_matrices):
            probas.append(tree.predict_proba(X @ R))
        return np.mean(probas, axis=0)

# =====================================================
# TRAIN MODEL
# =====================================================
model = RotationForest()
model.fit(X_train, y_train)

# =====================================================
# PREDICT + EVALUATE
# =====================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

results = evaluate_model(y_test, y_pred, y_proba)
print_results(results)
