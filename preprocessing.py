#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "final_ml_dataset.csv"
OUTPUT_CSV = "final_preprocessed.csv"
TARGET_COL = "DX_GROUP_y"

# ---- HARD BLOCK (never allowed anywhere)
ID_COLS = [
    "subject_x",
    "SITE_ID_x",
    "subject_y",
    "SUB_ID",
    "FILE_ID",
    "SITE_ID",
    "X",
    "Unnamed: 0",
    "Unnamed: 0.1",
    "COMORBIDITY",
    "qc_func_notes_rater_3",
    "qc_func_notes_rater_2",
    "qc_anat_notes_rater_2",
    "qc_notes_rater_1",
    "MEDICATION_NAME",
    "DX_GROUP_x",
    "SITE_ID_y"
]

# ---- Keep AS-IS (NO preprocessing)
PASSTHROUGH_COLS = [
    "SEX",
    "ADI_R_RSRCH_RELIABLE",
    "ADOS_RSRCH_RELIABLE",
    "SRS_VERSION",
    "OFF_STIMULANTS_AT_SCAN",
    "VINELAND_INFORMANT",
    "EYE_STATUS_AT_SCAN",
    "firing_rate",
    "firing_time",
    "energy",
    "entropy",
    "SUB_IN_SMP",
    "DSM_IV_TR"
]

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)
print("Original dataset shape:", df.shape)

# =====================================================
# DROP ALL ID COLUMNS (ONCE AND FOR ALL)
# =====================================================
df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
print("After dropping ID columns:", df.shape)

# =====================================================
# SEPARATE TARGET
# =====================================================
y = df[TARGET_COL].reset_index(drop=True)
X = df.drop(columns=[TARGET_COL])

# =====================================================
# PASSTHROUGH FEATURES
# =====================================================
passthrough_df = X[PASSTHROUGH_COLS].reset_index(drop=True)

# =====================================================
# PHENOTYPIC FEATURES (ONLY THESE CAN BE PROCESSED)
# =====================================================
pheno_df = X.drop(columns=PASSTHROUGH_COLS)

# 🔴 SAFETY CHECK — STOP IF ANY SUBJECT COLUMN LEAKS
bad_cols = [c for c in pheno_df.columns if "subject" in c.lower() or "id" in c.lower()]
if bad_cols:
    raise RuntimeError(
        f"FATAL: Subject/ID columns leaked into preprocessing: {bad_cols}"
    )

# =====================================================
# IDENTIFY TYPES
# =====================================================
numeric_cols = pheno_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = pheno_df.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric phenotypic features to scale: {len(numeric_cols)}")
print(f"Categorical phenotypic features to encode: {len(categorical_cols)}")

# =====================================================
# PREPROCESS
# =====================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop"
)

pheno_processed = preprocessor.fit_transform(pheno_df)

if hasattr(pheno_processed, "toarray"):
    pheno_processed = pheno_processed.toarray()

# =====================================================
# FEATURE NAMES
# =====================================================
cat_feature_names = (
    preprocessor.named_transformers_["cat"]
    .get_feature_names_out(categorical_cols)
)

processed_feature_names = numeric_cols + list(cat_feature_names)

pheno_processed_df = pd.DataFrame(
    pheno_processed,
    columns=processed_feature_names
)

# =====================================================
# FINAL DATASET
# =====================================================
final_df = pd.concat(
    [
        passthrough_df,
        pheno_processed_df,
        y
    ],
    axis=1
)

# =====================================================
# SAVE
# =====================================================
final_df.to_csv(OUTPUT_CSV, index=False)

print("\n======================================")
print("✅ PREPROCESSING COMPLETED CORRECTLY")
print("Final shape:", final_df.shape)
print("Saved to:", OUTPUT_CSV)
print("======================================")
