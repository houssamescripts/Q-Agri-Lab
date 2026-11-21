from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from app.quantum.circuits import create_quantum_circuit, circuit_info
from app.quantum.ibm_runtime import run_on_ibm_env, runtime_service
from app.experiments.experiment_runner import run_experiment

router = APIRouter(prefix="/quantum", tags=["Quantum"])

class CircuitRequest(BaseModel):
    num_qubits: int = Field(default=4, ge=1, le=8)
    layers: int = Field(default=1, ge=1, le=5)

class QuantumMLRequest(BaseModel):
    model_type: str = Field(default="qnn", pattern="^(qnn|quantum_kernel)$")
    seed: int = Field(default=0, ge=0)
    training_size: Literal[100, 600, 914] | None = None
    max_evals: int = Field(default=100, ge=10, le=100)

@router.get("/status")
def status():
    try:
        import qiskit
        available, version = True, qiskit.__version__
    except ImportError:
        available, version = False, None
    return {"status": "available" if available else "dependency_missing", "framework": "Qiskit", "version": version, "local_simulation": available, "hardware": "optional"}

@router.get("/ibm/status")
def ibm_status():
    import os
    token = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    if not token:
        return {"status": "not_configured", "token_present": False, "instance_present": bool(instance)}
    try:
        service = runtime_service()
        backends = service.backends(operational=True)
        return {"status": "connected", "token_present": True, "instance_present": bool(instance), "operational_backend_count": len(backends)}
    except ImportError as exc:
        return {"status": "dependency_missing", "token_present": True, "instance_present": bool(instance), "detail": "Install requirements-hardware.txt."}
    except Exception as exc:
        return {"status": "connection_failed", "token_present": True, "instance_present": bool(instance), "detail": str(exc)}

@router.post("/circuit")
def circuit(request: CircuitRequest):
    try:
        qc = create_quantum_circuit(request.num_qubits, request.layers)
        return {"status": "success", **circuit_info(qc), "circuit": str(qc)}
    except ImportError as exc:
        raise HTTPException(503, "Install backend requirements to enable Qiskit circuits.") from exc

@router.post("/experiment")
def quantum_ml_experiment(request: QuantumMLRequest):
    try:
        return run_experiment(request.model_type, request.seed, "validation", request.training_size, request.max_evals)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Quantum ML experiment failed: {exc}") from exc

@router.post("/ibm/run")
def ibm_run(request: CircuitRequest, shots: int = 1024):
    if not 100 <= shots <= 10000:
        raise HTTPException(422, "shots must be between 100 and 10000")
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(request.num_qubits)
        for q in range(request.num_qubits):
            qc.h(q)
        for q in range(request.num_qubits - 1):
            qc.cx(q, q + 1)
        return run_on_ibm_env(qc, shots=shots)
    except ImportError as exc:
        raise HTTPException(503, "Install requirements-hardware.txt to enable IBM Quantum execution.") from exc
    except Exception as exc:
        raise HTTPException(502, f"IBM Quantum execution failed: {exc}") from exc
