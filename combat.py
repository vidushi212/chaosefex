#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from neuroCombat import neuroCombat

# =====================================================
# CONFIG
# =====================================================
INPUT_CSV = "chaos_ml_ready.csv"
OUTPUT_CSV = "chaos_ml_ready_combat.csv"

FEATURE_COLS = [
    "firing_rate",
    "firing_time",
    "energy",
    "entropy"
]

BATCH_COL = "SITE_ID"
BIO_COL = "DX_GROUP"   # preserve diagnosis signal

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_CSV)
print("Original shape:", df.shape)

# =====================================================
# PREPARE MATRICES
# =====================================================
# neuroCombat expects (features × samples)
X = df[FEATURE_COLS].T.values

# Covariates dataframe
covars = df[[BATCH_COL, BIO_COL]].copy()

# =====================================================
# APPLY COMBAT
# =====================================================
combat_out = neuroCombat(
    dat=X,
    covars=covars,
    batch_col=BATCH_COL
)

X_combat = combat_out["data"].T

# =====================================================
# REASSEMBLE DATAFRAME
# =====================================================
combat_df = df.copy()
combat_df[FEATURE_COLS] = X_combat

# =====================================================
# SAVE
# =====================================================
combat_df.to_csv(OUTPUT_CSV, index=False)

print("\n======================================")
print("✅ ComBat harmonization completed")
print("Saved to:", OUTPUT_CSV)
print("======================================")

# =====================================================
# QUICK CHECK
# =====================================================
print("\nMean before vs after (per feature):")
print(pd.DataFrame({
    "before": df[FEATURE_COLS].mean(),
    "after": combat_df[FEATURE_COLS].mean()
}))
