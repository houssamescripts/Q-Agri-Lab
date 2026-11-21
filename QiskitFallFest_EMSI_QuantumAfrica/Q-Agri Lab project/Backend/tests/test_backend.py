from types import SimpleNamespace
import json

from fastapi.testclient import TestClient

from app.agents.research_agent import ResearcherAgent
from app.experiments.experiment_runner import dataset_summary, run_experiment
from app.reports.report_generator import generate_report
from main import app


def test_dataset_schema_and_splits():
    result = dataset_summary()
    assert result["splits"]["train"]["rows"] == 914
    assert result["splits"]["validation"]["rows"] == 150
    assert result["splits"]["test"] == {"rows": 221, "labeled": False}


def test_reference_svm_runs_on_official_data():
    result = run_experiment("svm", seed=0)
    assert result["training_rows"] == 914
    assert result["resources"]["features"] == 13
    assert set(result["metrics"]) >= {"accuracy", "macro_f1", "per_class", "confusion_matrix"}
    assert len(result["predictions"]) == 150


def test_quantum_models_smoke_on_official_starter_subset():
    qnn = run_experiment("qnn", seed=0, training_size=100, max_evals=10)
    assert qnn["training_rows"] == 100
    assert len(qnn["predictions"]) == 150
    kernel = run_experiment("quantum_kernel", seed=0, training_size=100)
    assert kernel["training_rows"] == 100
    assert len(kernel["predictions"]) == 150


def test_report_from_measured_result():
    result = run_experiment("svm", seed=0)
    report = generate_report(result)
    assert "Q-Agri Lab experiment report" in report
    assert "Macro-F1" in report


def test_api_routes_and_validation():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/experiments/dataset").json()["splits"]["train"]["rows"] == 914
    assert client.post("/api/experiments", json={"model_type": "svm", "seed": 0}).status_code == 200
    assert client.post("/api/experiments/compare", json={"model_types": ["svm"], "seeds": [0]}).status_code == 200
    measured = run_experiment("svm", seed=0)
    assert client.post("/api/results/report", json={"result": measured}).status_code == 200
    assert client.get("/api/quantum/status").status_code == 200
    assert client.post("/api/quantum/circuit", json={"num_qubits": 4, "layers": 1}).status_code == 200
    assert client.post("/api/quantum/experiment", json={"model_type": "qnn", "seed": 0, "training_size": 100, "max_evals": 10}).status_code == 200
    assert client.post("/api/quantum/experiment", json={"model_type": "quantum_kernel", "seed": 0, "training_size": 100}).status_code == 200
    assert client.post("/api/quantum/circuit", json={"num_qubits": 9, "layers": 1}).status_code == 422
    assert client.post("/api/experiments", json={"model_type": "svm", "split": "test"}).status_code == 422
    assert client.get("/openapi.json").status_code == 200


def test_agent_function_call_loop_without_network():
    first = SimpleNamespace(
        id="response-1",
        output=[SimpleNamespace(type="function_call", name="run_experiment", call_id="call-1", arguments=json.dumps({"model_type": "svm", "seed": 0}))],
        output_text="",
    )
    second = SimpleNamespace(id="response-2", output=[SimpleNamespace(type="message")], output_text="Validation SVM completed.")

    class Responses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return first if self.calls == 1 else second

    class FakeClient:
        def __init__(self):
            self.responses = Responses()

    fake = FakeClient()
    assert ResearcherAgent(client=fake).run("Run the baseline") == "Validation SVM completed."
    assert fake.responses.calls == 2
