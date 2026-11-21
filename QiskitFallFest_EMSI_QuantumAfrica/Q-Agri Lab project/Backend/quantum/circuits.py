
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes


def create_quantum_circuit(num_qubits: int = 4, reps: int = 1):
    """
    Build a Variational Quantum Classifier circuit.

    ZZFeatureMap: encodes satellite features.
    RealAmplitudes: provides trainable quantum parameters.
    """
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits,
        reps=1,
        entanglement="linear"
    )

    ansatz = RealAmplitudes(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="linear"
    )

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    return circuit


def get_circuit_info(circuit):
    return {
        "qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "parameters": circuit.num_parameters,
        "gates": circuit.count_ops()
    }
