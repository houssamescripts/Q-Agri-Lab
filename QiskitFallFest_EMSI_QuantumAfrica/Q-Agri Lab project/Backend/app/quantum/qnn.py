"""Adapter to the challenge's supplied, reproducible four-qubit QNN."""
import sys

from app.core.config import REPO_DIR


def build_qnn(max_evals: int = 100, encoding: str = "ry"):
    root = str(REPO_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    from challenge.models import QuantumClassifier

    return QuantumClassifier(labels=("alfalfa", "maize", "wheat"), max_evals=max_evals, encoding=encoding)
