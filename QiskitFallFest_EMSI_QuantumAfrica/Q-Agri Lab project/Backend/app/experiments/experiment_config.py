from typing import Literal
from pydantic import BaseModel, Field

from app.core.config import MODEL_TYPES


class ExperimentConfig(BaseModel):
    model_type: str = Field(default="svm", pattern="^(svm|qnn|quantum_kernel)$")
    seed: int = Field(default=0, ge=0)
    split: str = Field(default="validation", pattern="^(validation|test)$")
    training_size: Literal[100, 600, 914] | None = None
    max_evals: int = Field(default=100, ge=10, le=100)
