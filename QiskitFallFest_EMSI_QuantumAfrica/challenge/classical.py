"""Frozen classical reference, chosen with spatial training folds."""
import json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .data import ROOT


def reference(seed=0):
    config=json.loads((ROOT/'configs/reference.json').read_text())
    if config['name']!='rbf_svm':raise ValueError('This release freezes an RBF SVM')
    return make_pipeline(StandardScaler(),SVC()).set_params(**config['params'])
