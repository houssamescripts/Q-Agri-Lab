"""Exact local fidelity quantum-kernel classifier."""
from sklearn.svm import SVC


def build_quantum_kernel_classifier(training_features, training_labels, *, seed=0):
    """Fit an SVC with a Qiskit fidelity kernel on already-prepared PCA4 data.

    Quantum kernel matrices scale quadratically, so callers should use the
    fixed 100-row starter subset. ``training_features`` must have four columns.
    """
    import numpy as np
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    X = np.asarray(training_features, dtype=float)
    if X.ndim != 2 or X.shape[1] != 4 or len(X) > 100:
        raise ValueError("Quantum-kernel experiments require 1–100 rows of four-feature PCA data")
    feature_map = zz_feature_map(feature_dimension=4, reps=1, entanglement="linear")
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    train_kernel = kernel.evaluate(X)
    model = SVC(kernel="precomputed", random_state=seed)
    model.fit(train_kernel, training_labels)
    return model, kernel
