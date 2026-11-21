from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.experiments.experiment_runner import run_experiment, dataset_summary

router = APIRouter(prefix="/experiments", tags=["Experiments"])

class ExperimentRequest(BaseModel):
    model_type: str = Field(default="svm", pattern="^(svm|qnn|quantum_kernel)$")
    seed: int = Field(default=0, ge=0)
    split: str = Field(default="validation", pattern="^(validation|test)$")

@router.get("/dataset")
def get_dataset():
    try:
        return dataset_summary()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@router.post("")
def create_experiment(request: ExperimentRequest):
    try:
        return run_experiment(request.model_type, request.seed, request.split)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
