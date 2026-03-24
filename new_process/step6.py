#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 6: Train ensemble classifiers on preprocessed data
Compares: Random Forest, Bagged Trees, Boosted Trees, Rotation Forest
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)

# =====================================================
# CONFIG
# =====================================================
TRAIN_FILE = "preprocessing_results_no_combat/dataset_no_combat_train.csv"
TEST_FILE = "preprocessing_results_no_combat/dataset_no_combat_test.csv"
RESULTS_DIR = "ensemble_results"

TARGET_COL = "DX_GROUP"

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
print("="*80)
print("STEP 6: ENSEMBLE CLASSIFIERS")
print("="*80)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop([TARGET_COL], axis=1).values
y_train = train_df[TARGET_COL].values

X_test = test_df.drop([TARGET_COL], axis=1).values
y_test = test_df[TARGET_COL].values

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Classes: {np.unique(y_train)}")

# =====================================================
# DEFINE CLASSIFIERS
# =====================================================
classifiers = {
    'Random Forest (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42),
    'Random Forest (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42),
    'Bagged Trees (10)': BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=42
    ),
    'Bagged Trees (50)': BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=50,
        random_state=42
    ),
    'AdaBoost (10)': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=10,
        random_state=42
    ),
    'AdaBoost (50)': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    ),
}

# =====================================================
# TRAIN AND EVALUATE
# =====================================================
results = []

for name, clf in classifiers.items():
    print(f"\n{name}...")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
    
    # Train on full training set
    clf.fit(X_train, y_train)
    
    # Test predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    results.append({
        'Classifier': name,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Test_Accuracy': accuracy,
        'Test_Precision': precision,
        'Test_Recall': recall,
        'Test_F1': f1,
        'Test_ROC_AUC': roc_auc
    })
    
    print(f"  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test F1: {f1:.4f}")

# =====================================================
# SAVE RESULTS
# =====================================================
results_df = pd.DataFrame(results)
results_file = os.path.join(RESULTS_DIR, "ensemble_comparison.csv")
results_df.to_csv(results_file, index=False)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

print(f"\n✅ Saved: {results_file}")

# Best classifier
best_idx = results_df['Test_Accuracy'].idxmax()
print(f"\n🏆 Best: {results_df.loc[best_idx, 'Classifier']}")
print(f"   Accuracy: {results_df.loc[best_idx, 'Test_Accuracy']:.4f}")
print("="*80 + "\n")