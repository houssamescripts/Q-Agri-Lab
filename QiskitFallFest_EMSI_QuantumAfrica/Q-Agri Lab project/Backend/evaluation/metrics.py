from fastapi import fastapi

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import make_scorer
from sklearn.svm import LinearSVC
rom sklearn.metrics import accuracy_score

clf = svm.SVC(random_state=0)
cross_val_score(, scoring='recall_macro')
grid = GridSearchCV(LinearSVC(),})
