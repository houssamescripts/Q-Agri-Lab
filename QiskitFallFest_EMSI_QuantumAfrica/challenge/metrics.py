import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def score(y_true,y_pred,labels):
    if not set(y_true).issubset(labels) or not set(y_pred).issubset(labels):
        raise ValueError('Unknown label')
    precision,recall,_,support=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
    return dict(macro_f1=float(f1_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)),
                accuracy=float(accuracy_score(y_true,y_pred)),precision=precision.tolist(),
                recall=recall.tolist(),support=support.tolist(),
                confusion=confusion_matrix(y_true,y_pred,labels=labels).tolist())


def probabilities(p,n,k):
    p=np.asarray(p,dtype=float)
    if p.shape!=(n,k) or not np.isfinite(p).all() or (p < -1e-12).any() or not np.allclose(p.sum(axis=1),1,atol=1e-10):
        raise ValueError('Invalid class probabilities')
    return p
