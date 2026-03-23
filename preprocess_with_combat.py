#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 3a: Preprocess with ComBat harmonization
- Standardize ONLY demographic/clinical features
- Keep chaos features RAW (unscaled)
- Fit StandardScaler ONLY on training data
- Fit ComBat ONLY on training data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from neuroCombat import neuroCombat
import pickle

# =====================================================
# CONFIG
# =====================================================
TRAIN_INPUT = "chaos_ml_train.csv"
TEST_INPUT = "chaos_ml_test.csv"

TRAIN_OUTPUT = "dataset_combat_train.csv"
TEST_OUTPUT = "dataset_combat_test.csv"

SCALER_SAVE = "scaler_with_combat.pkl"

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

SITE_COL = "SITE_ID"
SUBJECT_COL = "subject"

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
def process_data_with_combat(df, scaler=None, is_train=True):
    """
    Process data with ComBat harmonization
    - Scale demographic/clinical features ONLY
    - Keep chaos features RAW (unscaled)
    """
    
    # Separate target
    y = df[TARGET_COL].copy()
    site = df[SITE_COL].copy()
    
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
    # APPLY COMBAT (ONLY if training)
    # =====================================================
    if is_train:
        print("\nApplying ComBat harmonization to TRAINING data...")
        print("  - Harmonizes: ALL features")
        print("  - Preserves: DX_GROUP signal")
        print("  - Removes: Site/batch effects")
        
        # Prepare for neuroCombat (features × samples)
        combat_input = feature_df.T
        
        covars = pd.DataFrame({
            "batch": site,
            "DX": y
        }, index=feature_df.index)
        
        # Apply ComBat
        combat_data = neuroCombat(
            dat=combat_input.values,
            covars=covars,
            batch_col="batch"
        )
        
        X_harmonized = combat_data["data"].T
        X_harmonized_df = pd.DataFrame(
            X_harmonized, 
            columns=feature_df.columns, 
            index=feature_df.index
        )
        
        print("✅ ComBat applied to training set")
    else:
        # For test set: just use the already-processed feature_df
        # (ComBat was fit on training, not reapplied to test)
        print("\nUsing feature-transformed TEST data")
        print("  Note: ComBat was fit on training data only")
        X_harmonized_df = feature_df
    
    # =====================================================
    # COMBINE WITH TARGET
    # =====================================================
    final_df = X_harmonized_df.copy()
    final_df[TARGET_COL] = y.reset_index(drop=True)
    
    return final_df, scaler

# =====================================================
# PROCESS TRAINING DATA
# =====================================================
print("\n" + "="*60)
print("PROCESSING TRAINING DATA WITH COMBAT")
print("="*60)
train_processed, scaler_fit = process_data_with_combat(train_df, is_train=True)

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

test_processed, _ = process_data_with_combat(test_df, scaler=scaler_loaded, is_train=False)

# =====================================================
# SAVE PROCESSED DATA
# =====================================================
train_processed.to_csv(TRAIN_OUTPUT, index=False)
test_processed.to_csv(TEST_OUTPUT, index=False)

print("\n" + "="*60)
print("✅ PREPROCESSING WITH COMBAT COMPLETED")
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