#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 3b: Preprocess without ComBat harmonization
- Standardize ONLY demographic/clinical features
- Keep chaos features RAW (unscaled)
- Fit StandardScaler ONLY on training data
- NO ComBat applied
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# =====================================================
# CONFIG
# =====================================================
TRAIN_INPUT = "chaos_ml_train.csv"
TEST_INPUT = "chaos_ml_test.csv"

TRAIN_OUTPUT = "dataset_no_combat_train.csv"
TEST_OUTPUT = "dataset_no_combat_test.csv"

SCALER_SAVE = "scaler_no_combat.pkl"

TARGET_COL = "DX_GROUP"

# =====================================================
# FEATURES
# =====================================================
# Only demographic/clinical features get scaled
# Chaos features stay RAW
DEMOGRAPHIC_FEATURES = []  # Adjust based on your data (AGE, SEX, IQ, etc)
CHAOS_FEATURES = [
    "firing_rate",
    "firing_time",
    "energy",
    "entropy"
]

# =====================================================
# LOAD DATA
# =====================================================
train_df = pd.read_csv(TRAIN_INPUT)
test_df = pd.read_csv(TEST_INPUT)

print("Training set shape:", train_df.shape)
print("Test set shape:", test_df.shape)

# =====================================================
# PROCESSING FUNCTION
# =====================================================
def process_data_no_combat(df, scaler=None, is_train=True):
    """
    Process data WITHOUT ComBat harmonization
    - Scale demographic/clinical features ONLY
    - Keep chaos features RAW (unscaled)
    """
    
    # Separate target
    y = df[TARGET_COL].copy()
    
    # Get all feature columns (demographic + chaos)
    X = df.drop(columns=[TARGET_COL]).copy()
    
    # =====================================================
    # SEPARATE DEMOGRAPHIC vs CHAOS FEATURES
    # =====================================================
    demographic_cols = [c for c in DEMOGRAPHIC_FEATURES if c in X.columns]
    chaos_df = X[CHAOS_FEATURES].copy()
    
    # =====================================================
    # SCALE DEMOGRAPHIC FEATURES ONLY
    # =====================================================
    if demographic_cols:
        if is_train:
            print("Fitting StandardScaler on DEMOGRAPHIC features (training data)...")
            scaler = StandardScaler()
            demographic_scaled = scaler.fit_transform(X[demographic_cols])
            print("✅ StandardScaler fitted")
        else:
            print("Transforming DEMOGRAPHIC features with fitted StandardScaler...")
            demographic_scaled = scaler.transform(X[demographic_cols])
            print("✅ Demographic features transformed")
        
        demographic_df = pd.DataFrame(demographic_scaled, columns=demographic_cols, index=X.index)
    else:
        demographic_df = pd.DataFrame(index=X.index)
        print("⚠️  No demographic features to scale")
    
    # =====================================================
    # COMBINE DEMOGRAPHIC (SCALED) + CHAOS (RAW)
    # =====================================================
    feature_df = pd.concat([demographic_df, chaos_df], axis=1)
    
    # =====================================================
    # NO ComBat applied - features ready as-is
    # =====================================================
    if is_train:
        print("\n✅ No ComBat harmonization applied")
        print("   Chaos features kept raw (unscaled)")
    
    # =====================================================
    # COMBINE WITH TARGET
    # =====================================================
    final_df = feature_df.copy()
    final_df[TARGET_COL] = y.reset_index(drop=True)
    
    return final_df, scaler

# =====================================================
# PROCESS TRAINING DATA
# =====================================================
print("\n" + "="*60)
print("PROCESSING TRAINING DATA WITHOUT COMBAT")
print("="*60)
train_processed, scaler_fit = process_data_no_combat(train_df, is_train=True)

# Save scaler for test data transformation
with open(SCALER_SAVE, 'wb') as f:
    pickle.dump(scaler_fit, f)

# =====================================================
# PROCESS TEST DATA
# =====================================================
print("\n" + "="*60)
print("PROCESSING TEST DATA")
print("="*60)
# Load the fitted scaler
with open(SCALER_SAVE, 'rb') as f:
    scaler_loaded = pickle.load(f)

test_processed, _ = process_data_no_combat(test_df, scaler=scaler_loaded, is_train=False)

# =====================================================
# SAVE PROCESSED DATA
# =====================================================
train_processed.to_csv(TRAIN_OUTPUT, index=False)
test_processed.to_csv(TEST_OUTPUT, index=False)

print("\n" + "="*60)
print("✅ PREPROCESSING WITHOUT COMBAT COMPLETED")
print("="*60)
print(f"\n📊 Training set: {TRAIN_OUTPUT}")
print(f"   Shape: {train_processed.shape}")
print(f"   Class distribution:\n{train_processed[TARGET_COL].value_counts()}")

print(f"\n📊 Test set: {TEST_OUTPUT}")
print(f"   Shape: {test_processed.shape}")
print(f"   Class distribution:\n{test_processed[TARGET_COL].value_counts()}")

print(f"\n💾 Scaler saved: {SCALER_SAVE}")
print("="*60)

print("\n📈 Chaos feature statistics (should be in [0,1] range):")
print(train_processed[CHAOS_FEATURES].describe())