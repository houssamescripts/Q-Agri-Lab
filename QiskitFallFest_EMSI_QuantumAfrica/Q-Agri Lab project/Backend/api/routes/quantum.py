
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.quantum.circuits import create_quantum_circuit
from app.quantum.quantum_kernel import run_quantum_kernel

router = APIRouter(
    prefix="/quantum",
    tags=["Quantum Computing"]
)


class QuantumCircuitRequest(BaseModel):
    num_qubits: int = Field(default=4, ge=1, le=8)
    layers: int = Field(default=1, ge=1, le=5)


class QuantumExperimentRequest(BaseModel):
    num_qubits: int = Field(default=4, ge=1, le=8)
    shots: int = Field(default=1024, ge=100, le=10000)


@router.get("/status")
async def quantum_status():
    """Check whether the quantum module is available."""
    return {
        "status": "available",
        "framework": "Qiskit",
        "local_simulation": True,
        "ibm_hardware": "optional"
    }


@router.post("/circuit")
async def generate_circuit(request: QuantumCircuitRequest):
    """Generate a quantum circuit."""
    try:
        circuit = create_quantum_circuit(
            num_qubits=request.num_qubits,
            layers=request.layers
        )

        return {
            "status": "success",
            "num_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "circuit": str(circuit)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Circuit generation failed: {str(error)}"
        )


@router.post("/experiment")
async def execute_quantum_experiment(
    request: QuantumExperimentRequest
):
    """Execute a quantum kernel experiment."""
    try:
        result = run_quantum_kernel(
            num_qubits=request.num_qubits,
            shots=request.shots
        )

        return {
            "status": "success",
            "result": result
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Quantum experiment failed: {str(error)}"
        )
