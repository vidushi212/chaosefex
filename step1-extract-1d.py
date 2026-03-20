#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = "/Users/vidushikhandelwal/Downloads/rois"
OUTPUT_CSV = "baseline_1d_features.csv"

# =====================================================
# HELPER
# =====================================================
def signal_entropy(x, bins=50):
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

# =====================================================
# MAIN
# =====================================================
rows = []

for fname in sorted(os.listdir(DATA_DIR)):

    if not fname.endswith(".1D"):
        continue

    subject_id = fname.replace(".1D", "")
    file_path = os.path.join(DATA_DIR, fname)

    ts = np.loadtxt(file_path)

    if ts.ndim != 2:
        continue

    flat = ts.flatten()

    rows.append({
        "FILE_ID": subject_id,
        "mean": np.mean(flat),
        "std": np.std(flat),
        "energy": np.sum(flat ** 2),
        "entropy": signal_entropy(flat)
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved baseline features:", OUTPUT_CSV)
print("Shape:", df.shape)
