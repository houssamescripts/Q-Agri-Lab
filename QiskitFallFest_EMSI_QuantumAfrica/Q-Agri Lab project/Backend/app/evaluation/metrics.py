"""Metrics shared by experiment execution and reports."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.core.config import CLASS_NAMES


def classification_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_NAMES, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "class_order": list(CLASS_NAMES),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_NAMES).tolist(),
    }
