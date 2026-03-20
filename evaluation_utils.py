import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

def evaluate_model(y_true, y_pred, y_proba=None):
    results = {}

    # =====================
    # Overall metrics
    # =====================
    acc = accuracy_score(y_true, y_pred)

    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    results["overall"] = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1
    }

    # ROC-AUC (binary only)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            results["overall"]["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            results["overall"]["roc_auc"] = None
    else:
        results["overall"]["roc_auc"] = None

    # =====================
    # Class-wise metrics
    # =====================
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    class_results = {}
    classes = np.unique(y_true)

    for i, cls in enumerate(classes):
        class_acc = np.mean(y_pred[y_true == cls] == cls)

        class_results[int(cls)] = {
            "precision": precisions[i],
            "recall": recalls[i],
            "f1": f1s[i],
            "accuracy": class_acc,
            "support": int(supports[i])
        }

    results["class_wise"] = class_results
    return results


def print_results(results):
    print("\n==============================")
    print("Overall Metrics")
    for k, v in results["overall"].items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: None")

    print("\nClass-wise Metrics")
    for cls, metrics in results["class_wise"].items():
        print(f"\nClass {cls}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
