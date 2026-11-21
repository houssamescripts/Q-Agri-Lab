
from app.experiments.experiment_runner import run_experiment
from app.evaluation.benchmark import compare_models

AGENT_TOOLS = [
    {
        "type": "function",
        "name": "run_experiment",
        "description": "Run a classical or quantum crop classification experiment.",
        "parameters": {
            "type": "object",
            "properties": {
                "model_type": {
                    "type": "string",
                    "enum": ["svm", "qnn", "quantum_kernel"]
                }
            },
            "required": ["model_type"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "compare_models",
        "description": "Compare classical and quantum model results.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    }
]

def execute(name: str, arguments: dict):
    tools = {
        "run_experiment": run_experiment,
        "compare_models": compare_models
    }

    if name not in tools:
        raise ValueError(f"Unknown tool: {name}")

    return tools[name]
