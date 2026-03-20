import pandas as pd

# =====================================
# CONFIG
# =====================================
FEATURE_FILE = "chaos_ml_ready_combat.csv"
PHENO_FILE = "pheno.csv"
OUTPUT_FILE = "final_ml_dataset.csv"

PHENO_ID_COL = "FILE_ID"   # confirmed from your phenotype file

# =====================================
# LOAD DATA
# =====================================
feat_df = pd.read_csv(FEATURE_FILE)
pheno_df = pd.read_csv(PHENO_FILE)

print("Features shape:", feat_df.shape)
print("Phenotypic shape:", pheno_df.shape)

# =====================================
# AUTO-DETECT SUBJECT COLUMN IN FEATURES
# =====================================
# Assume first column is subject identifier
feature_subject_col = feat_df.columns[0]

print("Detected feature subject column:", feature_subject_col)

# =====================================
# NORMALIZE SUBJECT IDS
# =====================================
feat_df["subject_id"] = (
    feat_df[feature_subject_col]
    .astype(str)
    .str.replace(".1D", "", regex=False)
)

pheno_df[PHENO_ID_COL] = pheno_df[PHENO_ID_COL].astype(str)

# =====================================
# DIAGNOSTICS
# =====================================
feat_subjects = set(feat_df["subject_id"])
pheno_subjects = set(pheno_df[PHENO_ID_COL])

print("\n======================================")
print(f"Subjects in features: {len(feat_subjects)}")
print(f"Subjects in phenotype: {len(pheno_subjects)}")
print(f"Subjects common: {len(feat_subjects & pheno_subjects)}")
print("======================================")

# =====================================
# MERGE
# =====================================
merged_df = pd.merge(
    feat_df,
    pheno_df,
    left_on="subject_id",
    right_on=PHENO_ID_COL,
    how="inner"
)

print("Subjects after merge:", merged_df.shape[0])

# =====================================
# CLEANUP
# =====================================
cols_to_drop = [feature_subject_col, "subject_id", PHENO_ID_COL]
cols_to_drop = [c for c in cols_to_drop if c in merged_df.columns]

merged_df.drop(columns=cols_to_drop, inplace=True)

# =====================================
# SAVE
# =====================================
merged_df.to_csv(OUTPUT_FILE, index=False)

print("\n======================================")
print(f"Final merged dataset saved as: {OUTPUT_FILE}")
print("Final shape:", merged_df.shape)
print("======================================")
