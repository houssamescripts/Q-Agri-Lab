from time import perf_counter
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import SVC
import numpy as np
from app.core.config import DATA_DIR, FEATURES, CLASS_NAMES


def load_split(name):
    if name not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation or test")
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Official challenge data not found: {path}")
    data = pd.read_csv(path)
    absent = sorted(set(FEATURES) - set(data.columns))
    if absent:
        raise ValueError(f"Challenge dataset lacks feature columns: {absent}")
    return data


def dataset_summary():
    output = {"feature_names": FEATURES, "class_order": CLASS_NAMES, "splits": {}}
    for name in ("train", "validation", "test"):
        frame = load_split(name)
        item = {"rows": len(frame), "labeled": "target" in frame}
        if item["labeled"]:
            item["class_counts"] = {str(k): int(v) for k, v in frame.target.value_counts().items()}
        output["splits"][name] = item
    return output


def run_experiment(model_type="svm", seed=0, split="validation"):
    if model_type != "svm":
        raise ValueError("Only the challenge RBF-SVM baseline is currently implemented. Quantum models need a dataset-specific implementation before execution.")
    if split != "validation":
        raise ValueError("Test labels are not supplied. Evaluate on validation; reserve test for final predictions.")
    view_path = DATA_DIR / "views" / "pca4.npz"
    if not view_path.is_file():
        raise FileNotFoundError(f"Official training-fitted PCA view not found: {view_path}")
    with np.load(view_path) as view:
        X_train, y_train = view["X_train"], view["y_train"]
        X_valid, y_valid = view["X_validation"], view["y_validation"]
    # Match the supplied challenge reference.json: transformations are already
    # fit on train only; use its fixed SVC settings for a fair baseline.
    model = SVC(kernel="rbf", C=100, gamma=0.01, random_state=seed)
    started = perf_counter()
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    duration = perf_counter() - started
    precision, recall, f1, support = precision_recall_fscore_support(y_valid, pred, labels=CLASS_NAMES, zero_division=0)
    return {"model_type": model_type, "seed": seed, "split": split, "runtime_seconds": duration,
            "metrics": {"accuracy": float(accuracy_score(y_valid, pred)), "macro_f1": float(f1.mean()),
                        "class_order": CLASS_NAMES, "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i,c in enumerate(CLASS_NAMES)},
                        "confusion_matrix": confusion_matrix(y_valid, pred, labels=CLASS_NAMES).tolist()},
            "predictions": pred.tolist()}
