#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =====================================================
# FILES
# =====================================================
INPUT_CSV = "final_ml_dataset.csv"
OUTPUT_CSV = "dataset_no_combat.csv"

TARGET_COL = "DX_GROUP_y"

# =====================================================
# FEATURES TO KEEP
# =====================================================

DEMOGRAPHIC_FEATURES = [
    "AGE_AT_SCAN",
    "SEX",
    "HANDEDNESS_CATEGORY"
]

CHAOS_FEATURES = [
    "firing_rate",
    "firing_time",
    "energy",
    "entropy"
]

SITE_COL = ["SITE_ID_y"]

KEEP_COLS = DEMOGRAPHIC_FEATURES + CHAOS_FEATURES + SITE_COL + [TARGET_COL]

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_CSV)
df = df[[c for c in KEEP_COLS if c in df.columns]]

df = df.rename(columns={"DX_GROUP_y": "DX_GROUP"})

# =====================================================
# SPLIT TARGET
# =====================================================

y = df["DX_GROUP"]
X = df.drop(columns=["DX_GROUP"])

# =====================================================
# SEPARATE FEATURES
# =====================================================

chaos_df = X[CHAOS_FEATURES].reset_index(drop=True)
site_df = X[SITE_COL].reset_index(drop=True)

pheno_df = X.drop(columns=CHAOS_FEATURES + SITE_COL)

# =====================================================
# IDENTIFY TYPES
# =====================================================

numeric_cols = pheno_df.select_dtypes(include=["int64","float64"]).columns
categorical_cols = pheno_df.select_dtypes(include=["object"]).columns

# =====================================================
# PREPROCESS
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

pheno_processed = preprocessor.fit_transform(pheno_df)

if hasattr(pheno_processed,"toarray"):
    pheno_processed = pheno_processed.toarray()

cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)

feature_names = list(numeric_cols) + list(cat_names)

pheno_processed_df = pd.DataFrame(pheno_processed,columns=feature_names)

# =====================================================
# FINAL DATASET
# =====================================================

final_df = pd.concat(
    [pheno_processed_df, chaos_df, y.reset_index(drop=True)],
    axis=1
)

final_df.to_csv(OUTPUT_CSV,index=False)

print("Dataset saved:",OUTPUT_CSV)
print("Shape:",final_df.shape)