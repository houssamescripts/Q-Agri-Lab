from time import perf_counter
import pandas as pd
import numpy as np
import json
from functools import lru_cache

from app.core.config import DATA_DIR, FEATURES, CLASS_NAMES, REPO_DIR
from app.classical.svm import build_svm
from app.evaluation.metrics import classification_metrics


@lru_cache(maxsize=1)
def _load_official_release():
    """Validate the supplied release schema and split alignment.

    The challenge's byte-level manifest hashes were authored for LF line
    endings; Windows checkouts normalize text files. Validate schemas, counts,
    finite values, IDs and PCA alignment without rewriting or rejecting those
    unchanged checkout files solely for line-ending differences.
    """
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    tables = {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in ("train", "validation", "test")}
    seen = set()
    labels = manifest["labels"]
    for split, frame in tables.items():
        expected = ["sample_id", *FEATURES] + ([] if split == "test" else ["target"])
        if list(frame.columns) != expected or len(frame) != manifest["rows"][split]:
            raise ValueError(f"Official {split} schema or row count does not match manifest.json")
        ids = set(frame.sample_id.astype(str))
        if not frame.sample_id.is_unique or seen.intersection(ids):
            raise ValueError("Official split sample IDs must be unique and disjoint")
        seen.update(ids)
        if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
            raise ValueError(f"Official {split} contains nonfinite features")
        if split != "test" and set(frame.target) != set(labels):
            raise ValueError(f"Official {split} must contain every declared class")
    with np.load(DATA_DIR / "views" / "pca4.npz", allow_pickle=False) as stored:
        view = {key: stored[key] for key in stored.files}
    if "y_test" in view:
        raise ValueError("The challenge release must not contain test labels")
    for split, frame in tables.items():
        if not np.array_equal(view[f"ids_{split}"], frame.sample_id.to_numpy()):
            raise ValueError(f"PCA view IDs are misaligned for {split}")
        if split != "test" and not np.array_equal(view[f"y_{split}"], frame.target.to_numpy()):
            raise ValueError(f"PCA view labels are misaligned for {split}")
    starter_ids = json.loads((DATA_DIR / "starter_ids.json").read_text(encoding="utf-8"))
    train_ids = set(tables["train"].sample_id.astype(str))
    if len(starter_ids) != 100 or len(set(starter_ids)) != 100 or not set(starter_ids).issubset(train_ids):
        raise ValueError("The official 100-row starter subset is invalid")
    return tables, view, manifest, None


def load_split(name):
    if name not in {"train", "validation", "test"}:
        raise ValueError("Split must be train, validation or test")
    tables, _, _, _ = _load_official_release()
    data = tables[name]
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


def _training_indices(view, training_size):
    if training_size in (None, 914):
        return np.arange(len(view["X_train"]))
    if training_size not in (100, 600):
        raise ValueError("training_size must be one of 100, 600 or 914")
    filename = "starter_ids.json" if training_size == 100 else "training_600_ids.json"
    selected = json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))
    id_to_index = {str(value): index for index, value in enumerate(view["ids_train"])}
    try:
        return np.asarray([id_to_index[str(value)] for value in selected], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Official subset IDs do not match the training PCA view: {exc}") from exc


def run_experiment(model_type="svm", seed=0, split="validation", training_size=None, max_evals=100):
    if model_type not in {"svm", "qnn", "quantum_kernel"}:
        raise ValueError("model_type must be svm, qnn or quantum_kernel")
    if split != "validation":
        raise ValueError("Test labels are not supplied. Evaluate on validation; reserve test for final predictions.")
    _, views, _, _ = _load_official_release()
    effective_size = 100 if model_type in {"qnn", "quantum_kernel"} and training_size is None else training_size
    if model_type == "quantum_kernel" and effective_size != 100:
        raise ValueError("quantum_kernel is capped at the official 100-row starter subset")
    view = views
    all_X_train, all_y_train = view["X_train"], view["y_train"]
    X_valid, y_valid = view["X_validation"], view["y_validation"]
    indices = _training_indices(view, effective_size)
    X_train, y_train = all_X_train[indices], all_y_train[indices]
    started = perf_counter()
    resources = {}
    if model_type == "svm":
        # The challenge freezes its classical reference on all 13 supplied
        # satellite features; the PCA4 view is reserved for quantum models.
        train_table = load_split("train")
        valid_table = load_split("validation")
        if effective_size not in (None, 914):
            selected_ids = set(json.loads((DATA_DIR / ("starter_ids.json" if effective_size == 100 else "training_600_ids.json")).read_text(encoding="utf-8")))
            train_table = train_table[train_table["sample_id"].astype(str).isin(selected_ids)]
        svm_X_train, svm_y_train = train_table[FEATURES], train_table["target"]
        svm_X_valid, svm_y_valid = valid_table[FEATURES], valid_table["target"]
        model = build_svm(seed)
        model.fit(svm_X_train, svm_y_train)
        pred = model.predict(svm_X_valid)
        y_valid = svm_y_valid.to_numpy()
        resources = {"training_rows": len(svm_X_train), "features": len(FEATURES), "feature_source": "official 13-feature CSV"}
    elif model_type == "qnn":
        from app.quantum.qnn import build_qnn
        model = build_qnn(max_evals=max_evals)
        model.fit(X_train, y_train, seed=seed)
        pred = model.predict(X_valid)
        resources = model.resources()
    else:
        # Fidelity kernels have O(n²) cost; default to the fixed 100-row subset.
        from app.quantum.quantum_kernel import build_quantum_kernel_classifier
        model, kernel = build_quantum_kernel_classifier(X_train, y_train, seed=seed)
        pred = model.predict(kernel.evaluate(X_valid, X_train))
        resources = {"training_rows": len(X_train), "features": int(X_train.shape[1]), "kernel": "exact fidelity", "scaling": "quadratic"}
    duration = perf_counter() - started
    return {"model_type": model_type, "seed": seed, "split": split, "training_rows": len(X_train),
            "runtime_seconds": duration, "resources": resources,
            "metrics": classification_metrics(y_valid, pred), "predictions": pred.tolist()}
