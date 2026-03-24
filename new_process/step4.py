#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 4: Stratified train/test split (80/20)
No data leakage - split BEFORE preprocessing
Output: chaos_ml_train.csv, chaos_ml_test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "new_process/chaos_ml_ready.csv"
TRAIN_CSV = "new_process/chaos_ml_train.csv"
TEST_CSV = "new_process/chaos_ml_test.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# =====================================================
# LOAD DATA
# =====================================================
print("="*80)
print("STEP 4: STRATIFIED TRAIN/TEST SPLIT")
print("="*80)

df = pd.read_csv(INPUT_CSV)

print(f"\nTotal samples: {len(df)}")
print(f"Class distribution:")
print(df['DX_GROUP'].value_counts())

# =====================================================
# STRATIFIED SPLIT
# =====================================================
X = df.drop(['DX_GROUP'], axis=1)
y = df['DX_GROUP']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# Reconstruct dataframes
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"\nTrain set: {train_df.shape}")
print(f"Train class distribution:")
print(train_df['DX_GROUP'].value_counts())

print(f"\nTest set: {test_df.shape}")
print(f"Test class distribution:")
print(test_df['DX_GROUP'].value_counts())

# =====================================================
# SAVE
# =====================================================
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"\n✅ Saved: {TRAIN_CSV}")
print(f"✅ Saved: {TEST_CSV}")
print("="*80 + "\n")