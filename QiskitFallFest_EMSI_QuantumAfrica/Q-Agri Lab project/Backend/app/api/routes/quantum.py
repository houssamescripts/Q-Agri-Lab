from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/quantum", tags=["Quantum"])

class CircuitRequest(BaseModel):
    num_qubits: int = Field(default=4, ge=1, le=8)
    layers: int = Field(default=1, ge=1, le=5)

@router.get("/status")
def status():
    try:
        import qiskit
        available = True
        version = qiskit.__version__
    except ImportError:
        available = False
        version = None
    return {"status": "available" if available else "dependency_missing", "framework": "Qiskit", "version": version, "local_simulation": available, "hardware": "optional"}

@router.get("/ibm/status")
def ibm_status():
    import os
    token = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    if not token:
        return {"status": "not_configured", "token_present": False, "instance_present": bool(instance)}
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        kwargs = {"channel": "ibm_quantum_platform", "token": token}
        if instance:
            kwargs["instance"] = instance
        service = QiskitRuntimeService(**kwargs)
        backends = service.backends(operational=True)
        return {"status": "connected", "token_present": True, "instance_present": bool(instance), "operational_backend_count": len(backends)}
    except ImportError as exc:
        return {"status": "dependency_missing", "token_present": True, "instance_present": bool(instance), "detail": "Install requirements-hardware.txt."}
    except Exception as exc:
        return {"status": "connection_failed", "token_present": True, "instance_present": bool(instance), "detail": str(exc)}

@router.post("/circuit")
def circuit(request: CircuitRequest):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        from fastapi import HTTPException
        raise HTTPException(503, "Install the repository requirements to enable Qiskit circuits.") from exc
    qc = QuantumCircuit(request.num_qubits)
    for _ in range(request.layers):
        for q in range(request.num_qubits):
            qc.ry(0.5, q)
        for q in range(request.num_qubits - 1):
            qc.cx(q, q + 1)
    return {"qubits": qc.num_qubits, "depth": qc.depth(), "operations": {k: int(v) for k,v in qc.count_ops().items()}, "circuit": str(qc)}

@router.post("/ibm/run")
def ibm_run(request: CircuitRequest, shots: int = 1024):
    if not 100 <= shots <= 10000:
        raise HTTPException(422, "shots must be between 100 and 10000")
    try:
        from qiskit import QuantumCircuit
        from quantum.ibm_runtime import run_on_ibm_env
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
