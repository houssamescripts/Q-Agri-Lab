"""Deterministic experiment-plan helper for research conversations."""
from app.core.config import SEEDS


def build_experiment_plan(objective: str, model_types=("svm", "qnn")):
    if not objective.strip():
        raise ValueError("An experiment objective is required")
    return {
        "objective": objective.strip(),
        "models": list(model_types),
        "train_split": "official training split",
        "selection_split": "official validation split",
        "test_policy": "Do not inspect test labels; reserve test for final predictions.",
        "seeds": list(SEEDS),
        "metrics": ["macro_f1", "accuracy", "per_class_precision_recall", "confusion_matrix", "runtime_seconds", "model_resources"],
        "controls": ["keep spatial splits fixed", "fit preprocessing on training data only", "report model-specific sample counts"],
    }
