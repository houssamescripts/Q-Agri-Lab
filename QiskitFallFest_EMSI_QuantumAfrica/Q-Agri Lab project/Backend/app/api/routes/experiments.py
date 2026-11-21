from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.core.config import MODEL_TYPES
from app.evaluation.benchmark import compare_models
from app.experiments.experiment_runner import run_experiment, dataset_summary

router = APIRouter(prefix="/experiments", tags=["Experiments"])

class ExperimentRequest(BaseModel):
    model_type: str = Field(default="svm", pattern="^(svm|qnn|quantum_kernel)$")
    seed: int = Field(default=0, ge=0)
    split: str = Field(default="validation", pattern="^(validation|test)$")
    training_size: Literal[100, 600, 914] | None = None
    max_evals: int = Field(default=100, ge=10, le=100)

class ComparisonRequest(BaseModel):
    model_types: list[str] = Field(default_factory=lambda: ["svm"], min_length=1, max_length=3)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1, max_length=3)
    training_size: Literal[100, 600, 914] | None = None

@router.get("/dataset")
def get_dataset():
    try:
        return dataset_summary()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@router.post("")
def create_experiment(request: ExperimentRequest):
    try:
        return run_experiment(request.model_type, request.seed, request.split, request.training_size, request.max_evals)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

@router.post("/compare")
def compare(request: ComparisonRequest):
    if any(model not in MODEL_TYPES for model in request.model_types):
        raise HTTPException(422, f"model_types must be selected from {list(MODEL_TYPES)}")
    if any(seed < 0 for seed in request.seeds):
        raise HTTPException(422, "seeds must be non-negative")
    try:
        return compare_models(request.model_types, request.seeds, request.training_size)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
