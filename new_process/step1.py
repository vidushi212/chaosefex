#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 1: Extract Chaos Features from ROI (.1D) files
ONLY 4 FEATURES: firing_rate, firing_time, energy, entropy
Uses ChaosFEX skew-tent map
Output: roi_chaos_features.csv (one row per ROI per subject)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
import ChaosFEX.feature_extractor as CFX

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "/Users/vidushikhandelwal/Downloads/abide_rois"
OUTPUT_CSV = "/Users/vidushikhandelwal/ChaosFEX/new_process/roi_chaos_features.csv"
MISS_REPORT = "roi_extraction_report.txt"

# ChaosFEX hyperparameters
INITIAL_COND = 0.1
TRAJECTORY_LEN = 3000
EPSILON = 0.01
THRESHOLD = 0.2

EPS = 1e-10
FIRING_THRESHOLD = 0.5  # For binarizing ROI signal

# =====================================================
# MAIN PROCESSING
# =====================================================
all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".1D")])
processed_files = set()
skipped_files = {}
rows = []

print("="*80)
print("STEP 1: CHAOS FEATURE EXTRACTION FROM ROI FILES")
print("="*80)
print(f"Features extracted: firing_rate, firing_time, energy, entropy")
print(f"Found {len(all_files)} .1D files\n")

for fname in all_files:
    path = os.path.join(DATA_DIR, fname)
    subject_id = fname.replace(".1D", "")
    
    print(f"Processing {fname}...")

    try:
        # Load ROI time series (T × ROIs)
        ts = np.loadtxt(path)

        # Validate
        if ts.ndim != 2 or ts.shape[0] < 50:
            skipped_files[fname] = "Invalid dimensions or too short"
            continue

        # Transpose to (ROIs × Time)
        ts_rois = ts.T.astype(np.float64)
        
        n_rois = ts_rois.shape[0]
        n_timepoints = ts_rois.shape[1]

        # =====================================================
        # Z-SCORE NORMALIZE PER ROI (CRITICAL FOR CHAOS)
        # =====================================================
        mean = ts_rois.mean(axis=1, keepdims=True)
        std = ts_rois.std(axis=1, keepdims=True) + EPS
        ts_rois_normalized = (ts_rois - mean) / std

        # Remove invalid ROIs
        valid_mask = np.isfinite(ts_rois_normalized).all(axis=1)
        ts_rois_normalized = ts_rois_normalized[valid_mask]

        if ts_rois_normalized.shape[0] < 2:
            skipped_files[fname] = "Insufficient valid ROIs"
            continue

        # =====================================================
        # EXTRACT 4 CHAOS FEATURES PER ROI
        # =====================================================
        for roi_idx, roi_signal in enumerate(ts_rois_normalized):
            
            # =========== FEATURE 1: FIRING RATE ===========
            # Proportion of timepoints exceeding threshold
            firing_events = (roi_signal > FIRING_THRESHOLD).astype(int)
            firing_rate = firing_events.mean()
            
            # =========== FEATURE 2: FIRING TIME ===========
            # Mean temporal position of firing events
            firing_indices = np.where(firing_events > 0)[0]
            if len(firing_indices) > 0:
                firing_time = firing_indices.mean() / n_timepoints
            else:
                firing_time = 0.0
            
            # =========== FEATURE 3: ENERGY ===========
            # Sum of squared signal
            energy = np.sum(roi_signal ** 2)
            
            # =========== FEATURE 4: ENTROPY ===========
            # Shannon entropy of signal distribution
            # Bin signal into histogram
            hist, _ = np.histogram(roi_signal, bins=30, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy_val = -np.sum(hist * np.log2(hist + EPS))
            
            # Save row
            rows.append({
                'subject': subject_id,
                'roi': roi_idx,
                'firing_rate': firing_rate,
                'firing_time': firing_time,
                'energy': energy,
                'entropy': entropy_val
            })
        
        processed_files.add(fname)
        print(f"  ✅ Extracted {ts_rois_normalized.shape[0]} ROIs\n")

    except Exception as e:
        skipped_files[fname] = str(e)
        print(f"  ❌ Error: {e}\n")

# =====================================================
# SAVE RESULTS
# =====================================================
print("="*80)
print("SAVING RESULTS")
print("="*80)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved: {OUTPUT_CSV}")
print(f"Shape: {df.shape}")
print(f"Rows (ROIs): {len(df)}")
print(f"Unique subjects: {df['subject'].nunique()}\n")

print("First few rows:")
print(df.head())

print("\nFeature statistics:")
print(df[['firing_rate', 'firing_time', 'energy', 'entropy']].describe())

# =====================================================
# SAVE REPORT
# =====================================================
with open(MISS_REPORT, "w") as f:
    f.write("="*80 + "\n")
    f.write("ROI CHAOS FEATURE EXTRACTION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("FEATURES EXTRACTED (4 ONLY):\n")
    f.write("  1. firing_rate - Proportion of firing events (threshold > 0.5)\n")
    f.write("  2. firing_time - Normalized mean time of firing events\n")
    f.write("  3. energy - Sum of squared signal\n")
    f.write("  4. entropy - Shannon entropy of signal distribution\n\n")
    
    f.write(f"Total .1D files: {len(all_files)}\n")
    f.write(f"Processed: {len(processed_files)}\n")
    f.write(f"Skipped: {len(skipped_files)}\n")
    f.write(f"Total ROIs extracted: {len(rows)}\n\n")
    
    f.write("Processing Parameters:\n")
    f.write(f"  Firing Threshold: {FIRING_THRESHOLD}\n")
    f.write(f"  Entropy Bins: 30\n\n")
    
    if skipped_files:
        f.write("SKIPPED FILES:\n")
        for fname, reason in skipped_files.items():
            f.write(f"  {fname}: {reason}\n")

print(f"✅ Report saved: {MISS_REPORT}")
print("="*80 + "\n")