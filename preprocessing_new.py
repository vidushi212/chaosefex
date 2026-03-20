
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =====================================================
# FILES
# =====================================================
INPUT_CSV = "final_ml_dataset.csv"
OUTPUT_CSV = "final_clean_for_ml.csv"
TARGET_COL = "DX_GROUP_y"

# =====================================================
# FEATURES WE WANT TO KEEP
# =====================================================

DEMOGRAPHIC_FEATURES = [
    "AGE_AT_SCAN",
    "SEX",
    "HANDEDNESS_CATEGORY",
    "HANDEDNESS_SCORES",
    "BMI"
]

IQ_FEATURES = [
    "FIQ",
    "VIQ",
    "PIQ"
]

CHAOS_FEATURES = [
    "firing_rate",
    "firing_time",
    "energy",
    "entropy"
]

SITE_FEATURE = ["SITE_ID_y"]   # only for ComBat later

KEEP_COLS = DEMOGRAPHIC_FEATURES + IQ_FEATURES + CHAOS_FEATURES + SITE_FEATURE + [TARGET_COL]

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)
print("Original dataset:", df.shape)

# Keep only approved columns
df = df[[c for c in KEEP_COLS if c in df.columns]]
print("After column filtering:", df.shape)

# Rename target for sanity
df = df.rename(columns={"DX_GROUP_y": "DX_GROUP"})

# =====================================================
# SPLIT TARGET
# =====================================================
y = df["DX_GROUP"]
X = df.drop(columns=["DX_GROUP"])

# =====================================================
# CHAOS FEATURES (NO PREPROCESSING)
# =====================================================
chaos_df = X[CHAOS_FEATURES].reset_index(drop=True)

# =====================================================
# SITE COLUMN (NO PREPROCESSING)
# =====================================================
site_df = X[SITE_FEATURE].reset_index(drop=True)

# =====================================================
# PHENOTYPIC FEATURES TO PROCESS
# =====================================================
pheno_df = X.drop(columns=CHAOS_FEATURES + SITE_FEATURE)

# Identify types
numeric_cols = pheno_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = pheno_df.select_dtypes(include=["object"]).columns.tolist()

print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# =====================================================
# PREPROCESS PHENOTYPIC FEATURES
# =====================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop"
)

pheno_processed = preprocessor.fit_transform(pheno_df)

# Convert sparse → dense if needed
if hasattr(pheno_processed, "toarray"):
    pheno_processed = pheno_processed.toarray()

# Feature names after encoding
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
processed_feature_names = numeric_cols + list(cat_feature_names)

pheno_processed_df = pd.DataFrame(pheno_processed, columns=processed_feature_names)

# =====================================================
# FINAL DATASET
# =====================================================
final_df = pd.concat(
    [
        pheno_processed_df,
        chaos_df,
        site_df,
        y.reset_index(drop=True)
    ],
    axis=1
)

# =====================================================
# SAVE
# =====================================================
final_df.to_csv(OUTPUT_CSV, index=False)

print("\n======================================")
print("✅ CLEAN DATASET CREATED")
print("Final shape:", final_df.shape)
print("Saved to:", OUTPUT_CSV)
print("======================================")
