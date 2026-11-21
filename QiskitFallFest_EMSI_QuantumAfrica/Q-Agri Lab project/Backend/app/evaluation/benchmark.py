"""Seed-based paired summaries for model comparisons."""
from statistics import mean, pstdev

from app.core.config import SEEDS
from app.experiments.experiment_runner import run_experiment


def compare_models(model_types=("svm",), seeds=SEEDS, training_size=None):
    if training_size is None:
        training_size = 100 if any(model in {"qnn", "quantum_kernel"} for model in model_types) else 914
    runs = [run_experiment(model_type=model, seed=seed, training_size=training_size) for model in model_types for seed in seeds]
    summaries = {}
    for model in model_types:
        rows = [run for run in runs if run["model_type"] == model]
        summaries[model] = {
            "seeds": [row["seed"] for row in rows],
            "mean_macro_f1": mean(row["metrics"]["macro_f1"] for row in rows),
            "std_macro_f1": pstdev(row["metrics"]["macro_f1"] for row in rows),
            "mean_accuracy": mean(row["metrics"]["accuracy"] for row in rows),
            "std_accuracy": pstdev(row["metrics"]["accuracy"] for row in rows),
            "training_rows": training_size,
            "runs": rows,
        }
    return {"split": "validation", "summaries": summaries}
