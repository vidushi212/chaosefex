#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# =====================================================
# CONFIG
# =====================================================
BASELINE_1D_CSV = "baseline_1d_with_labels.csv"
PHENO_CSV = "pheno.csv"
OUTPUT_CSV = "baseline_1d_with_labels.csv"

# =====================================================
# LOAD
# =====================================================
baseline_df = pd.read_csv(BASELINE_1D_CSV)
pheno_df = pd.read_csv(PHENO_CSV)

print("Baseline shape:", baseline_df.shape)
print("Phenotype shape:", pheno_df.shape)

# =====================================================
# FIND SUBJECT COLUMN IN BASELINE
# =====================================================
possible_subject_cols = ["subject", "SUBJECT", "filename", "file", "id"]

baseline_subject_col = None
for c in possible_subject_cols:
    if c in baseline_df.columns:
        baseline_subject_col = c
        break

if baseline_subject_col is None:
    raise RuntimeError("❌ No subject column found in baseline CSV")

# =====================================================
# NORMALIZE SUBJECT IDS
# =====================================================
baseline_df["FILE_ID"] = (
    baseline_df[baseline_subject_col]
    .astype(str)
    .str.replace(".1D", "", regex=False)
)

pheno_df["FILE_ID"] = pheno_df["FILE_ID"].astype(str)

# =====================================================
# DROP DUPLICATE ID COLUMNS
# =====================================================
baseline_df = baseline_df.drop(columns=[baseline_subject_col], errors="ignore")

# =====================================================
# MERGE
# =====================================================
df = baseline_df.merge(
    pheno_df,
    on="FILE_ID",
    how="inner",
    validate="one_to_one"
)

# =====================================================
# REPORT
# =====================================================
print("\n======================================")
print("Subjects in baseline:", baseline_df.shape[0])
print("Subjects in phenotype:", pheno_df.shape[0])
print("Subjects after merge:", df.shape[0])
print("======================================")

# =====================================================
# SAVE
# =====================================================
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
