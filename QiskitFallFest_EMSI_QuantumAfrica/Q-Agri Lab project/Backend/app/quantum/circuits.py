from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes, zz_feature_map


def create_quantum_circuit(num_qubits: int = 4, layers: int = 1):
    if not 1 <= num_qubits <= 8 or not 1 <= layers <= 5:
        raise ValueError("num_qubits must be 1–8 and layers must be 1–5")
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=1, entanglement="linear")
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=layers, entanglement="linear")
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    return circuit


def circuit_info(circuit):
    return {"qubits": circuit.num_qubits, "depth": circuit.depth(), "parameters": circuit.num_parameters,
            "operations": {str(name): int(count) for name, count in circuit.count_ops().items()}}
