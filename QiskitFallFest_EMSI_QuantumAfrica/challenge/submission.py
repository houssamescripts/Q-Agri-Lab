"""Submission interface on four training-fitted PCA features, without IDs."""
from .models import QuantumClassifier


class Submission(QuantumClassifier):
    def __init__(self):
        super().__init__(labels=('alfalfa','maize','wheat'),max_evals=100,encoding='ry')
