#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 3: Merge chaos features with phenotype data
ONLY 4 CHAOS FEATURES: firing_rate, firing_time, energy, entropy
Requires: pheno.csv with [FILE_ID, SITE_ID, DX_GROUP, AGE_AT_SCAN, SEX]
Output: chaos_ml_ready.csv (ready for ML)
"""

import pandas as pd

# =====================================================
# CONFIG
# =====================================================
CHAOS_CSV = "new_process/chaos_features_subject.csv"
PHENO_CSV = "pheno.csv"
OUTPUT_CSV = "new_process/chaos_ml_ready.csv"

CHAOS_FEATURES = ['firing_rate', 'firing_time', 'energy', 'entropy']

# =====================================================
# LOAD DATA
# =====================================================
print("="*80)
print("STEP 3: MERGE CHAOS FEATURES WITH PHENOTYPE")
print("="*80)

df_chaos = pd.read_csv(CHAOS_CSV)
df_pheno = pd.read_csv(PHENO_CSV)

print(f"\nChaos features: {df_chaos.shape}")
print(f"Features: {CHAOS_FEATURES}")
print(f"\nPhenotype data: {df_pheno.shape}")

# =====================================================
# CLEAN SUBJECT IDS
# =====================================================
df_chaos['subject_id'] = df_chaos['subject'].str.replace(".1D", "", regex=False)
df_pheno['subject_id'] = df_pheno['FILE_ID'].astype(str)

print(f"\nChaos unique subjects: {df_chaos['subject_id'].nunique()}")
print(f"Phenotype unique subjects: {df_pheno['subject_id'].nunique()}")

# =====================================================
# MERGE
# =====================================================
merged_df = pd.merge(
    df_chaos[['subject_id'] + CHAOS_FEATURES],
    df_pheno[['subject_id', 'SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']],
    on='subject_id',
    how='inner'
)

print(f"\nMerged shape: {merged_df.shape}")

# =====================================================
# RENAME AND REORDER
# =====================================================
merged_df.rename(columns={'subject_id': 'subject'}, inplace=True)

final_cols = ['subject', 'SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX'] + CHAOS_FEATURES
merged_df = merged_df[final_cols]

print(f"\nFinal columns: {list(merged_df.columns)}")
print(f"\nClass distribution:")
print(merged_df['DX_GROUP'].value_counts())

print(f"\nData summary:")
print(merged_df.head())

# =====================================================
# SAVE
# =====================================================
merged_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved: {OUTPUT_CSV}")
print(f"✅ Shape: {merged_df.shape}")
print(f"✅ Ready for ML pipeline!")
print("="*80 + "\n")