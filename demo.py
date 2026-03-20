#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import ChaosFEX.feature_extractor as CFX

# =====================================================
# CONFIGURATION
# =====================================================
DATA_DIR = "/Users/vidushikhandelwal/Downloads/rois"   # folder with .1D files
OUT_FILE = "chaos_features_variable.npz"

INITIAL_COND = 0.1
TRAJECTORY_LEN = 3000
EPSILON = 0.01
THRESHOLD = 0.2

# =====================================================
# STORAGE (dictionary for variable-length vectors)
# =====================================================
chaos_features = {}
skipped = []

# =====================================================
# MAIN LOOP
# =====================================================
for fname in sorted(os.listdir(DATA_DIR)):

    if not fname.endswith(".1D"):
        continue

    subject_id = fname.replace(".1D", "")
    file_path = os.path.join(DATA_DIR, fname)

    print(f"Processing {subject_id}")

    try:
        # ---------------------------------------------
        # Load ROI time series (T × ROIs)
        # ---------------------------------------------
        ts = np.loadtxt(file_path)

        if ts.ndim != 2 or ts.shape[0] < 10:
            print(f"  Skipped: invalid time series")
            skipped.append(subject_id)
            continue

        # ---------------------------------------------
        # Transpose → (ROIs × Time)
        # ---------------------------------------------
        feat_mat = ts.T.astype(np.float64)

        # ---------------------------------------------
        # Z-score normalization PER ROI (REQUIRED)
        # ---------------------------------------------
        mean = feat_mat.mean(axis=1, keepdims=True)
        std = feat_mat.std(axis=1, keepdims=True) + 1e-8
        feat_mat = (feat_mat - mean) / std

        # ---------------------------------------------
        # Remove invalid ROIs
        # ---------------------------------------------
        feat_mat = feat_mat[np.isfinite(feat_mat).all(axis=1)]

        if feat_mat.shape[0] < 2:
            print(f"  Skipped: insufficient valid ROIs")
            skipped.append(subject_id)
            continue

        # ---------------------------------------------
        # Chaos Feature Extraction (FULL VECTOR)
        # ---------------------------------------------
        chaos_vec = CFX.transform(
            feat_mat,
            initial_cond=INITIAL_COND,
            trajectory_len=TRAJECTORY_LEN,
            epsilon=EPSILON,
            threshold=THRESHOLD
        )

        chaos_vec = np.asarray(chaos_vec).ravel()

        if chaos_vec.size == 0:
            print(f"  Skipped: empty chaos output")
            skipped.append(subject_id)
            continue

        # ---------------------------------------------
        # STORE VARIABLE-LENGTH VECTOR
        # ---------------------------------------------
        chaos_features[subject_id] = chaos_vec
        print(f"  ✔ Stored ({chaos_vec.size} features)")

    except Exception as e:
        print(f"  Skipped: error -> {e}")
        skipped.append(subject_id)

# =====================================================
# SAVE OUTPUT
# =====================================================
if not chaos_features:
    raise RuntimeError("No valid subjects processed.")

np.savez_compressed(OUT_FILE, **chaos_features)

print("\n======================================")
print(f"Saved chaos features to: {OUT_FILE}")
print(f"Subjects processed: {len(chaos_features)}")
print(f"Subjects skipped: {len(skipped)}")
print("======================================")
