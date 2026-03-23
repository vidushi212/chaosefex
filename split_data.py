#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2: Split data into train/test BEFORE any preprocessing
This prevents data leakage
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "chaos_ml_ready.csv"
TRAIN_OUTPUT = "chaos_ml_train.csv"
TEST_OUTPUT = "chaos_ml_test.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)

print("Original dataset shape:", df.shape)
print("Class distribution before split:")
print(df["DX_GROUP"].value_counts())

# =====================================================
# TRAIN/TEST SPLIT
# =====================================================
# Stratify by diagnosis to maintain class balance
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["DX_GROUP"]
)

# =====================================================
# SAVE SPLITS
# =====================================================
train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print("\n======================================")
print("✅ Train/test split completed")
print("======================================")
print(f"Training set: {TRAIN_OUTPUT}")
print(f"  Shape: {train_df.shape}")
print(f"  Class distribution:\n{train_df['DX_GROUP'].value_counts()}")

print(f"\nTest set: {TEST_OUTPUT}")
print(f"  Shape: {test_df.shape}")
print(f"  Class distribution:\n{test_df['DX_GROUP'].value_counts()}")

print("\n======================================")
print(f"Random state: {RANDOM_STATE}")
print(f"Test size: {TEST_SIZE * 100}%")
print("======================================")