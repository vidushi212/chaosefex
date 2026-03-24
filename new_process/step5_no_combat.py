#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 5: Preprocessing - NO COMBAT (Better performance)
StandardScaler normalization only
Output: dataset_no_combat_train.csv, dataset_no_combat_test.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

# =====================================================
# CONFIG
# =====================================================
TRAIN_FILE = "chaos_ml_train.csv"
TEST_FILE = "chaos_ml_test.csv"
RESULTS_DIR = "preprocessing_results_no_combat"

TARGET_COL = "DX_GROUP"
SITE_COL = "SITE_ID"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
print("="*80)
print("STEP 5: PREPROCESSING (NO COMBAT)")
print("="*80)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"\nTrain: {train_df.shape}")
print(f"Test: {test_df.shape}")

# =====================================================
# EXTRACT TARGET AND METADATA
# =====================================================
y_train = train_df[TARGET_COL].copy()
y_test = test_df[TARGET_COL].copy()

site_train = train_df[SITE_COL].copy()
site_test = test_df[SITE_COL].copy()

# =====================================================
# GET FEATURE COLUMNS (EXCLUDE METADATA)
# =====================================================
exclude_cols = [TARGET_COL, SITE_COL, 'subject']
exclude_cols = [c for c in exclude_cols if c in train_df.columns]

feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

print(f"\nFeature columns: {len(feature_cols)}")
print(f"Features: {feature_cols[:5]} ...")

# =====================================================
# STANDARDSCALER (Fit on train, transform both)
# =====================================================
print("\n" + "="*80)
print("STANDARDSCALER NORMALIZATION")
print("="*80)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

X_train_normalized_df = pd.DataFrame(
    X_train_normalized,
    columns=feature_cols,
    index=X_train.index
)

X_test_normalized_df = pd.DataFrame(
    X_test_normalized,
    columns=feature_cols,
    index=X_test.index
)

print(f"Train mean (should be ≈0): {X_train_normalized_df.mean().mean():.8f}")
print(f"Train std (should be ≈1): {X_train_normalized_df.std().mean():.8f}")

# =====================================================
# ADD TARGET AND SAVE
# =====================================================
train_final = X_train_normalized_df.copy()
train_final[TARGET_COL] = y_train.values

test_final = X_test_normalized_df.copy()
test_final[TARGET_COL] = y_test.values

# Reorder columns
cols_order = feature_cols + [TARGET_COL]
train_final = train_final[cols_order]
test_final = test_final[cols_order]

train_output = os.path.join(RESULTS_DIR, "dataset_no_combat_train.csv")
test_output = os.path.join(RESULTS_DIR, "dataset_no_combat_test.csv")

train_final.to_csv(train_output, index=False)
test_final.to_csv(test_output, index=False)

# Save scaler
scaler_file = os.path.join(RESULTS_DIR, "scaler.pkl")
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n✅ Saved: {train_output}")
print(f"✅ Saved: {test_output}")
print(f"✅ Saved: {scaler_file}")

# Summary
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)
print(f"Train: {train_final.shape}")
print(f"Test: {test_final.shape}")
print(f"Features: {len(feature_cols)}")
print(f"Target: {TARGET_COL}")
print(f"✅ Ready for classification!")
print("="*80 + "\n")