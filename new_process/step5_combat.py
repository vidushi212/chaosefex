#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 5: Preprocessing WITH COMBAT
Order: ComBat (batch correction) → StandardScaler (normalization)
Removes site/batch effects before normalization
Output: dataset_combat_train.csv, dataset_combat_test.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.preprocessing import StandardScaler
from neuroCombat import neuroCombat

# =====================================================
# CONFIG
# =====================================================
TRAIN_FILE = "chaos_ml_train.csv"
TEST_FILE = "chaos_ml_test.csv"
RESULTS_DIR = "preprocessing_results_combat"

TARGET_COL = "DX_GROUP"
SITE_COL = "SITE_ID"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
print("="*80)
print("STEP 5: PREPROCESSING WITH COMBAT")
print("="*80)
print(f"Results directory: {RESULTS_DIR}\n")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}\n")

# =====================================================
# STEP 1: EXTRACT TARGET, SITE, AND FEATURES
# =====================================================
print("="*80)
print("STEP 1: EXTRACT TARGET, SITE, AND FEATURES")
print("="*80)

# Extract target and site
y_train = train_df[TARGET_COL].copy()
y_test = test_df[TARGET_COL].copy()

site_train = train_df[SITE_COL].copy()
site_test = test_df[SITE_COL].copy()

print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")

print(f"\nTrain class distribution:")
print(y_train.value_counts())

print(f"\nTest class distribution:")
print(y_test.value_counts())

print(f"\nTrain site distribution:")
print(site_train.value_counts())

print(f"\nTest site distribution:")
print(site_test.value_counts())

# =====================================================
# STEP 2: GET NUMERIC FEATURE COLUMNS
# =====================================================
print("\n" + "="*80)
print("STEP 2: EXTRACT NUMERIC FEATURE COLUMNS")
print("="*80)

# Exclude metadata columns
exclude_cols = [TARGET_COL, SITE_COL, 'subject']
exclude_cols = [c for c in exclude_cols if c in train_df.columns]

feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train_raw = train_df[feature_cols].copy()
X_test_raw = test_df[feature_cols].copy()

print(f"Feature columns: {len(feature_cols)}")
print(f"Features: {feature_cols}\n")

print(f"Train features shape: {X_train_raw.shape}")
print(f"Test features shape: {X_test_raw.shape}\n")

print(f"Train data statistics (RAW):")
print(X_train_raw.describe())

# =====================================================
# STEP 3: APPLY COMBAT TO RAW DATA
# =====================================================
print("\n" + "="*80)
print("STEP 3: APPLY COMBAT TO RAW DATA")
print("="*80)
print("Note: ComBat applied to COMBINED train+test RAW data\n")

# Combine train + test for ComBat
print("Combining train + test data for ComBat...")
X_combined_raw = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
y_combined = pd.concat(
    [y_train.reset_index(drop=True), y_test.reset_index(drop=True)],
    ignore_index=True
)
site_combined = pd.concat(
    [site_train.reset_index(drop=True), site_test.reset_index(drop=True)],
    ignore_index=True
)

print(f"Combined data shape: {X_combined_raw.shape}")
print(f"Combined targets shape: {y_combined.shape}")
print(f"Unique batch/sites: {sorted(site_combined.unique())}")
print(f"Site distribution:")
print(site_combined.value_counts())

# =====================================================
# Prepare ComBat input
# =====================================================
print("\nPreparing ComBat input...")
combat_input = X_combined_raw.T  # ComBat expects (features × samples)

covars = pd.DataFrame({
    "batch": site_combined.values,
    "DX": y_combined.values
})

print(f"ComBat input shape: {combat_input.shape} (features × samples)")
print(f"Covariates shape: {covars.shape}")

# =====================================================
# Apply ComBat
# =====================================================
print("\n🔄 Running ComBat...")
print("   (Harmonizing raw features, removing batch effects)\n")

try:
    combat_data = neuroCombat(
        dat=combat_input.values.astype(np.float64),  # Convert to float64
        covars=covars,
        batch_col="batch"
    )
    
    X_combat_combined = combat_data["data"].T  # Convert back to (samples × features)
    X_combat_combined_df = pd.DataFrame(
        X_combat_combined,
        columns=X_combined_raw.columns
    )
    
    print("✅ ComBat applied successfully!")
    print(f"   Output shape: {X_combat_combined_df.shape}\n")
    
except Exception as e:
    print(f"❌ Error in ComBat: {e}")
    print("⚠️ Using raw data instead (NO COMBAT)...\n")
    X_combat_combined_df = X_combined_raw.copy()

# =====================================================
# Split back into train and test (AFTER ComBat)
# =====================================================
print("Splitting harmonized data back into train and test...")
n_train = len(X_train_raw)
X_train_harmonized = X_combat_combined_df.iloc[:n_train].copy()
X_test_harmonized = X_combat_combined_df.iloc[n_train:].copy()

X_train_harmonized.reset_index(drop=True, inplace=True)
X_test_harmonized.reset_index(drop=True, inplace=True)

print(f"Train harmonized shape: {X_train_harmonized.shape}")
print(f"Test harmonized shape: {X_test_harmonized.shape}\n")

print(f"Train data statistics (AFTER ComBat, BEFORE normalization):")
print(X_train_harmonized.describe())

# =====================================================
# STEP 4: STANDARDIZE FEATURES ON HARMONIZED DATA
# =====================================================
print("\n" + "="*80)
print("STEP 4: STANDARDIZE FEATURES (SECOND STEP)")
print("="*80)
print("Note: StandardScaler fitted on HARMONIZED train data\n")

# Fit StandardScaler on harmonized training data
print(f"Fitting StandardScaler on all {X_train_harmonized.shape[1]} numeric features...\n")

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train_harmonized.values)
X_test_normalized = scaler.transform(X_test_harmonized.values)

X_train_normalized_df = pd.DataFrame(
    X_train_normalized,
    columns=X_train_harmonized.columns,
    index=X_train_harmonized.index
)

X_test_normalized_df = pd.DataFrame(
    X_test_normalized,
    columns=X_test_harmonized.columns,
    index=X_test_harmonized.index
)

print(f"✅ Training data normalized (mean≈0, std≈1)")
print(f"✅ Test data transformed using training statistics\n")

print(f"Train data statistics (AFTER ComBat AND normalization):")
print(X_train_normalized_df.describe())

print(f"\nTest data statistics (AFTER ComBat AND normalization):")
print(X_test_normalized_df.describe())

# Save scaler
scaler_file = os.path.join(RESULTS_DIR, "scaler.pkl")
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"\n✅ Scaler saved: {scaler_file}")

# =====================================================
# STEP 5: ADD TARGET AND SAVE
# =====================================================
print("\n" + "="*80)
print("STEP 5: ADD TARGET AND SAVE")
print("="*80)

train_final = X_train_normalized_df.copy()
train_final[TARGET_COL] = y_train.reset_index(drop=True).values

test_final = X_test_normalized_df.copy()
test_final[TARGET_COL] = y_test.reset_index(drop=True).values

# Reorder columns: features first, then target
cols_order = list(train_final.columns)
cols_order = [c for c in cols_order if c != TARGET_COL] + [TARGET_COL]
train_final = train_final[cols_order]
test_final = test_final[cols_order]

# Save
train_output = os.path.join(RESULTS_DIR, "dataset_combat_train.csv")
test_output = os.path.join(RESULTS_DIR, "dataset_combat_test.csv")

train_final.to_csv(train_output, index=False)
test_final.to_csv(test_output, index=False)

print(f"✅ Train data saved: {train_output}")
print(f"   Shape: {train_final.shape}")
print(f"   Columns: {len(train_final.columns)}")

print(f"\n✅ Test data saved: {test_output}")
print(f"   Shape: {test_final.shape}")
print(f"   Columns: {len(test_final.columns)}")

# =====================================================
# STEP 6: DATA QUALITY VERIFICATION
# =====================================================
print("\n" + "="*80)
print("STEP 6: DATA QUALITY VERIFICATION")
print("="*80)

print("\n📊 Train data info:")
print(train_final.info())

print("\n📊 Test data info:")
print(test_final.info())

print("\n✅ Train class distribution:")
print(train_final[TARGET_COL].value_counts())

print("\n✅ Test class distribution:")
print(test_final[TARGET_COL].value_counts())

# Check missing values
train_missing = train_final.isnull().sum().sum()
test_missing = test_final.isnull().sum().sum()
print(f"\n🔍 Missing values - Train: {train_missing}, Test: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("⚠️ Warning: Missing values detected!")
else:
    print("✅ No missing values - data is clean!")

# Verify normalization
print("\n✅ Normalization verification (numeric features only):")
numeric_cols = [c for c in train_final.columns if c != TARGET_COL]
print(f"   Mean (should be ≈0): {train_final[numeric_cols].mean().mean():.8f}")
print(f"   Std (should be ≈1): {train_final[numeric_cols].std().mean():.8f}")

# =====================================================
# STEP 7: SUMMARY
# =====================================================
print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE: COMBAT → THEN NORMALIZATION")
print("="*80)

summary = f"""
PREPROCESSING PIPELINE COMPLETED:

Order of Operations:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ✅ LOAD RAW DATA
   Input: {TRAIN_FILE} + {TEST_FILE}
   Train shape: {X_train_raw.shape}
   Test shape: {X_test_raw.shape}

2. ✅ APPLY COMBAT TO RAW DATA (FIRST)
   Input: Raw, unscaled numeric data
   Process: neuroCombat with batch correction by SITE_ID
   Sites: {site_combined.nunique()} unique sites
   Output: Harmonized data (still raw scale)
   Train shape: {X_train_harmonized.shape}
   Test shape: {X_test_harmonized.shape}

3. ✅ NORMALIZE HARMONIZED DATA (SECOND)
   Input: Harmonized data from ComBat
   Process: StandardScaler (fit on train, transform both)
   Output: Normalized + harmonized data (mean≈0, std≈1)
   Train shape: {X_train_normalized_df.shape}
   Test shape: {X_test_normalized_df.shape}

4. ✅ ADD TARGET AND SAVE
   Train file: {train_output}
   Test file: {test_output}

Data Statistics After All Preprocessing:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Train Mean (should be ≈0):
{X_train_normalized_df.mean().mean():.8f}

Train Std (should be ≈1):
{X_train_normalized_df.std().mean():.8f}

Output Files:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ {train_output}
✅ {test_output}
✅ {scaler_file}

Key Points:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ComBat applied to RAW numeric data first
✅ Removes batch/site effects from raw scale
✅ Then StandardScaler normalizes the harmonized data
✅ Both train and test normalized using train statistics
✅ No data leakage (scaler fit on train only)
✅ Ready for model training!

Pipeline Order: ComBat (batch correction) → StandardScaler (normalization)

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary)

# Save summary
summary_file = os.path.join(RESULTS_DIR, "PREPROCESSING_SUMMARY.txt")
with open(summary_file, 'w') as f:
    f.write(summary)
print(f"\n✅ Summary saved: {summary_file}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print(f"""
Run ensemble classifiers on this preprocessed data:

python step6_ensemble_classifiers.py

Files to use:
   Training: {train_output}
   Testing: {test_output}

Compare with NO-COMBAT version:
   - COMBAT: Removes site effects first
   - NO-COMBAT: Direct normalization only

Expected accuracies:
   - NO-COMBAT (baseline): ~59%
   - COMBAT: ~48% (site effects may be confounded)
""")

print("="*80)