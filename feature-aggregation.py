#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

# =====================================================
# PATHS
# =====================================================
INPUT_CSV = "roi_firing_features.csv"
OUTPUT_CSV = "chaos_features_subject.csv"

# =====================================================
# LOAD ROI-LEVEL FEATURES
# =====================================================
df = pd.read_csv(INPUT_CSV)

print("Input shape (ROI-level):", df.shape)
print("Columns:", df.columns.tolist())

# =====================================================
# SAFETY CHECK
# =====================================================
required_cols = {
    "subject",
    "firing_time",
    "firing_rate",
    "energy",
    "entropy"
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =====================================================
# AGGREGATE ROIs → SUBJECT
# Mean pooling (recommended & stable)
# =====================================================
agg_df = (
    df
    .groupby("subject", as_index=False)
    .agg({
        "firing_time": "mean",
        "firing_rate": "mean",
        "energy": "mean",
        "entropy": "mean"
    })
)

# =====================================================
# REPORTING
# =====================================================
num_subjects = agg_df.shape[0]
num_rois_per_subject = df.groupby("subject").size()

print(f"Subjects aggregated: {num_subjects}")
print("ROI count per subject (min / max):",
      num_rois_per_subject.min(),
      num_rois_per_subject.max())

# =====================================================
# SAVE SUBJECT-LEVEL FEATURES
# =====================================================
agg_df.to_csv(OUTPUT_CSV, index=False)

print("\n======================================")
print(f"Saved subject-level features to: {OUTPUT_CSV}")
print("Each subject has exactly 4 features")
print("======================================")
