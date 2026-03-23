#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1: Merge chaos features with phenotype data
NO ComBat application here - just merging
"""

import pandas as pd

# =====================================================
# CONFIG
# =====================================================
CHAOS_CSV = "chaos_features_subject.csv"
PHENO_CSV = "pheno.csv"
OUTPUT_CSV = "chaos_ml_ready.csv"

# Chaos feature columns
CHAOS_FEATURES = [
    "firing_rate",
    "firing_time",
    "energy",
    "entropy"
]

# =====================================================
# LOAD DATA
# =====================================================
chaos_df = pd.read_csv(CHAOS_CSV)
pheno_df = pd.read_csv(PHENO_CSV)

print("Chaos features shape:", chaos_df.shape)
print("Phenotypic shape:", pheno_df.shape)

# =====================================================
# CLEAN SUBJECT IDS
# =====================================================
chaos_df["subject_id"] = chaos_df["subject"].str.replace(".1D", "", regex=False)
pheno_df["subject_id"] = pheno_df["FILE_ID"]

# =====================================================
# SELECT REQUIRED COLUMNS ONLY
# =====================================================
chaos_keep = ["subject_id"] + CHAOS_FEATURES
pheno_keep = ["subject_id", "SITE_ID", "DX_GROUP"]

chaos_df = chaos_df[chaos_keep]
pheno_df = pheno_df[pheno_keep]

# =====================================================
# MERGE
# =====================================================
merged_df = pd.merge(
    chaos_df,
    pheno_df,
    on="subject_id",
    how="inner"
)

# =====================================================
# FINAL CLEANUP
# =====================================================
merged_df.rename(columns={"subject_id": "subject"}, inplace=True)
merged_df = merged_df[
    ["subject", "SITE_ID", "DX_GROUP"] + CHAOS_FEATURES
]

# =====================================================
# SAVE
# =====================================================
merged_df.to_csv(OUTPUT_CSV, index=False)

print("\n======================================")
print("✅ Merged data created (NO ComBat yet)")
print("Output CSV:", OUTPUT_CSV)
print("Final shape:", merged_df.shape)
print("======================================")

print("\nClass distribution:")
print(merged_df["DX_GROUP"].value_counts())

print("\nSite distribution (top 10):")
print(merged_df["SITE_ID"].value_counts().head(10))