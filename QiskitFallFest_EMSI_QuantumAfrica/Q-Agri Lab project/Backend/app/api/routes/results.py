from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.experiments.experiment_runner import dataset_summary
from app.reports.report_generator import generate_report

router = APIRouter(prefix="/results", tags=["Results"])

class ReportRequest(BaseModel):
    result: dict

@router.get("/dataset")
def results_dataset():
    return dataset_summary()

@router.get("/reference")
def reference_results():
    return {"source": "challenge README starter results", "split": "validation", "seeds": [0, 1, 2], "models": [
        {"name": "RBF SVM", "mean_macro_f1": 0.8797, "mean_accuracy": 0.88},
        {"name": "Four-qubit QNN", "mean_macro_f1": 0.8557, "mean_accuracy": 0.8556}
    ], "note": "Reference values from the challenge; not produced by this API run."}

@router.post("/report")
def create_report(request: ReportRequest):
    try:
        return {"format": "markdown", "report": generate_report(request.result)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
