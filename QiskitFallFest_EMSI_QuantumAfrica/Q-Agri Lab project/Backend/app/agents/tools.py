"""Safe experiment tools made available to the Responses API agent."""
import json

from app.core.config import MODEL_TYPES, SEEDS
from app.evaluation.benchmark import compare_models
from app.experiments.experiment_runner import run_experiment


AGENT_TOOLS = [
    {
        "type": "function", "name": "run_experiment",
        "description": "Run one validation experiment on the official challenge split.",
        "parameters": {
            "type": "object", "properties": {
                "model_type": {"type": "string", "enum": list(MODEL_TYPES)},
                "seed": {"type": "integer", "minimum": 0},
            }, "required": ["model_type", "seed"], "additionalProperties": False,
        }, "strict": True,
    },
    {
        "type": "function", "name": "compare_models",
        "description": "Compare selected models on the fixed validation split over reproducibility seeds.",
        "parameters": {
            "type": "object", "properties": {
                "model_types": {"type": "array", "items": {"type": "string", "enum": list(MODEL_TYPES)}, "minItems": 1, "maxItems": 3},
                "seeds": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "maxItems": 3},
            }, "required": ["model_types", "seeds"], "additionalProperties": False,
        }, "strict": True,
    },
]


def execute_tool(name: str, arguments: dict):
    if name == "run_experiment":
        result = run_experiment(arguments["model_type"], arguments["seed"])
        result.pop("predictions", None)
        return result
    if name == "compare_models":
        result = compare_models(arguments["model_types"], arguments["seeds"])
        for summary in result["summaries"].values():
            for run in summary["runs"]:
                run.pop("predictions", None)
        return result
    raise ValueError(f"Unknown agent tool: {name}")


def tool_output(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
