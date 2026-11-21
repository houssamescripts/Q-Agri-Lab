from fastapi import APIRouter
from app.experiments.experiment_runner import dataset_summary

router = APIRouter(prefix="/results", tags=["Results"])

@router.get("/dataset")
def results_dataset():
    return dataset_summary()

@router.get("/reference")
def reference_results():
    return {"source": "challenge README starter results", "split": "validation", "seeds": [0, 1, 2], "models": [
        {"name": "RBF SVM", "mean_macro_f1": 0.8797, "mean_accuracy": 0.88},
        {"name": "Four-qubit QNN", "mean_macro_f1": 0.8557, "mean_accuracy": 0.8556}
    ], "note": "Reference values from the challenge; not produced by this API run."}
