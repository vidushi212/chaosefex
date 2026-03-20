#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "/Users/vidushikhandelwal/Downloads/rois"
OUTPUT_CSV = "roi_firing_features.csv"
MISS_REPORT = "missed_files.txt"

THRESHOLD_Z = 1.0
EPS = 1e-10

# =====================================================
# FEATURE FUNCTION
# =====================================================
def extract_features(signal):
    T = len(signal)
    firing_idx = np.where(signal > THRESHOLD_Z)[0]

    firing_time = firing_idx.mean() if firing_idx.size > 0 else -1
    firing_rate = firing_idx.size / T
    energy = np.sum(signal ** 2)

    hist, _ = np.histogram(signal, bins=30, density=True)
    hist += EPS
    ent = shannon_entropy(hist)

    return [firing_time, firing_rate, energy, ent]

# =====================================================
# FILE TRACKING
# =====================================================
all_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".1D"))
processed_files = set()
skipped_files = {}

rows = []

# =====================================================
# MAIN LOOP
# =====================================================
for fname in all_files:
    path = os.path.join(DATA_DIR, fname)
    print(f"Processing {fname}")

    try:
        ts = np.loadtxt(path)

        if ts.ndim != 2:
            skipped_files[fname] = "Not a T×ROI matrix"
            continue

        ts = ts.T

        mean = ts.mean(axis=1, keepdims=True)
        std = ts.std(axis=1, keepdims=True) + EPS
        ts = (ts - mean) / std

        for roi_idx, roi_signal in enumerate(ts):
            features = extract_features(roi_signal)
            rows.append([fname, roi_idx, *features])

        processed_files.add(fname)

    except Exception as e:
        skipped_files[fname] = str(e)

# =====================================================
# SAVE CSV
# =====================================================
df = pd.DataFrame(
    rows,
    columns=["subject", "roi", "firing_time", "firing_rate", "energy", "entropy"]
)
df.to_csv(OUTPUT_CSV, index=False)

# =====================================================
# MISS REPORT
# =====================================================
missed_files = set(all_files) - processed_files

with open(MISS_REPORT, "w") as f:
    f.write("=== FILE PROCESSING REPORT ===\n\n")

    f.write(f"Total .1D files found: {len(all_files)}\n")
    f.write(f"Files processed successfully: {len(processed_files)}\n")
    f.write(f"Files skipped: {len(skipped_files)}\n")
    f.write(f"Files missed: {len(missed_files)}\n\n")

    if skipped_files:
        f.write("---- SKIPPED FILES ----\n")
        for k, v in skipped_files.items():
            f.write(f"{k} :: {v}\n")

    if missed_files:
        f.write("\n---- MISSED FILES ----\n")
        for fmiss in sorted(missed_files):
            f.write(f"{fmiss}\n")

# =====================================================
# SUMMARY
# =====================================================
print("\n======================================")
print(f"CSV saved to: {OUTPUT_CSV}")
print(f"Total .1D files found: {len(all_files)}")
print(f"Processed files: {len(processed_files)}")
print(f"Skipped files: {len(skipped_files)}")
print(f"Missed files: {len(missed_files)}")
print(f"Detailed report: {MISS_REPORT}")
print("======================================")
