#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2: Aggregate ROI-level chaos features to SUBJECT level
ONLY 4 FEATURES: firing_rate, firing_time, energy, entropy
Computes: mean across ROIs (simpler, no std/min/max)
Output: chaos_features_subject.csv (one row per subject)
"""

import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "new_process/roi_chaos_features.csv"
OUTPUT_CSV = "new_process/chaos_features_subject.csv"

CHAOS_FEATURES = ['firing_rate', 'firing_time', 'energy', 'entropy']

# =====================================================
# LOAD ROI-LEVEL FEATURES
# =====================================================
print("="*80)
print("STEP 2: AGGREGATING ROI FEATURES TO SUBJECT LEVEL")
print("="*80)

df_roi = pd.read_csv(INPUT_CSV)

print(f"\nInput shape: {df_roi.shape}")
print(f"Subjects: {df_roi['subject'].nunique()}")
print(f"ROIs per subject: {df_roi.groupby('subject')['roi'].count().mean():.1f} avg")
print(f"\nFeatures: {CHAOS_FEATURES}\n")

# =====================================================
# AGGREGATE PER SUBJECT (ONLY MEAN)
# =====================================================
subject_features = []

for subject_id in df_roi['subject'].unique():
    subject_data = df_roi[df_roi['subject'] == subject_id]
    
    row = {'subject': subject_id}
    
    # For each chaos feature - ONLY take MEAN
    for feat in CHAOS_FEATURES:
        feat_values = subject_data[feat].values
        row[feat] = feat_values.mean()
    
    subject_features.append(row)

df_subject = pd.DataFrame(subject_features)

print(f"Output shape: {df_subject.shape}")
print(f"Subjects: {len(df_subject)}\n")

print("First few rows:")
print(df_subject.head())

print("\nFeature statistics:")
print(df_subject[CHAOS_FEATURES].describe())

# =====================================================
# SAVE
# =====================================================
df_subject.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved: {OUTPUT_CSV}")
print(f"✅ {len(df_subject)} subjects with 4 features each")
print("="*80 + "\n")