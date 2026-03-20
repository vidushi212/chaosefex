#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# =====================================================
# CONFIG
# =====================================================
ROI_FEATURES = "roi_firing_features.csv"   # unaggregated ROI features
PHENO_CSV = "pheno.csv"                  # phenotypic data
OUTPUT_CSV = "roi_dataset_with_pheno.csv"

ROI_SUBJECT_COL = "subject"
PHENO_SUBJECT_COL = "FILE_ID"

# =====================================================
# LOAD FILES
# =====================================================
roi_df = pd.read_csv(ROI_FEATURES)
pheno_df = pd.read_csv(PHENO_CSV)

print("ROI dataset shape:", roi_df.shape)
print("Phenotypic dataset shape:", pheno_df.shape)

# =====================================================
# STANDARDIZE SUBJECT IDs
# =====================================================
# remove .1D if present
roi_df["subject_id"] = roi_df[ROI_SUBJECT_COL].str.replace(".1D", "", regex=False)

# ensure phenotypic IDs are strings
pheno_df["subject_id"] = pheno_df[PHENO_SUBJECT_COL].astype(str)

# =====================================================
# CHECK MATCHING SUBJECTS
# =====================================================
roi_subjects = set(roi_df["subject_id"])
pheno_subjects = set(pheno_df["subject_id"])

print("\nSubjects in ROI:", len(roi_subjects))
print("Subjects in pheno:", len(pheno_subjects))
print("Common subjects:", len(roi_subjects.intersection(pheno_subjects)))

# =====================================================
# MERGE (IMPORTANT STEP)
# =====================================================
merged_df = roi_df.merge(
    pheno_df,
    on="subject_id",
    how="inner"
)

print("\nMerged dataset shape:", merged_df.shape)

# =====================================================
# CLEANUP USELESS COLUMNS
# =====================================================
DROP_COLS = [
    "subject_id",
    "FILE_ID",
    "SUB_ID",
    "SITE_ID_x",
    "SITE_ID_y",
    "Unnamed: 0",
    "Unnamed: 0.1"
]

merged_df = merged_df.drop(columns=[c for c in DROP_COLS if c in merged_df.columns],
                           errors="ignore")

# =====================================================
# SAVE FINAL DATASET
# =====================================================
merged_df.to_csv(OUTPUT_CSV, index=False)

print("\n======================================")
print("✅ ROI + Phenotypic dataset created")
print("Saved to:", OUTPUT_CSV)
print("Final shape:", merged_df.shape)
print("======================================")
