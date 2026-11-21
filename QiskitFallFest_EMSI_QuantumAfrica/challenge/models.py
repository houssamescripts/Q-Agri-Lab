"""Local quantum classifier and classical control for three-crop classification."""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.primitives import QMLSampler
from .metrics import probabilities


def angles(X):
    X=np.asarray(X,dtype=float)
    if X.ndim!=2 or X.shape[1]!=4 or not np.isfinite(X).all():
        raise ValueError('Expected a finite (n,4) prepared view')
    return (np.pi/2)*(1+np.tanh(X))


def make_circuit(encoding_name='zz'):
    if encoding_name == 'ry':
        encoding=QuantumCircuit(4)
        x=ParameterVector('x',4)
        for i in range(4): encoding.ry(x[i],i)
    elif encoding_name == 'zz':
        encoding=zz_feature_map(4,reps=1,entanglement='linear',parameter_prefix='x')
    else: raise ValueError('Unknown encoding')
    ansatz=real_amplitudes(4,reps=1,entanglement='linear',parameter_prefix='theta')
    return encoding.compose(ansatz),tuple(encoding.parameters),tuple(ansatz.parameters)


class QuantumClassifier:
    def __init__(self, labels=('alfalfa','maize','wheat'), max_evals=100, encoding='ry'):
        self.labels=np.asarray(labels); self.max_evals=max_evals
        self.encoding=encoding
        self.circuit,self.inputs,self.weights=make_circuit(encoding)
        k=len(self.labels)
        self.outcomes=np.array([int(i).bit_count()%2 if encoding=='ry' and k==2 else i%k for i in range(16)])
        self.qnn=SamplerQNN(circuit=self.circuit,sampler=QMLSampler(shots=None),
                            input_params=self.inputs,weight_params=self.weights,
                            interpret=lambda outcome:int(self.outcomes[int(outcome)]),output_shape=k)
        self.history=[]

    def fit(self,X_train,y_train,seed=0):
        y=np.asarray(y_train)
        if set(y)!=set(self.labels): raise ValueError('Training must contain every declared class')
        encoded=np.array([np.flatnonzero(self.labels==v)[0] for v in y])
        start=perf_counter(); self.history=[]
        def callback(weights,objective):
            self.history.append(dict(evaluation=len(self.history)+1,objective=float(objective),seconds=perf_counter()-start))
        self.model=NeuralNetworkClassifier(self.qnn,one_hot=True,loss='cross_entropy',
            optimizer=COBYLA(maxiter=self.max_evals),callback=callback,
            initial_point=np.random.default_rng(seed).uniform(-np.pi,np.pi,len(self.weights)))
        self.model.fit(angles(X_train),np.eye(len(self.labels))[encoded])
        self.fit_seconds=perf_counter()-start
        return self

    def predict_proba(self,X):
        return probabilities(self.qnn.forward(angles(X),self.model.weights),len(X),len(self.labels))

    def predict(self,X):
        return self.labels[self.predict_proba(X).argmax(axis=1)]

    def bound_circuit(self,row,weights=None):
        if weights is None: weights=self.model.weights
        bind=dict(zip(self.inputs,angles(np.asarray(row).reshape(1,4))[0]))
        bind.update(zip(self.weights,weights))
        return self.circuit.assign_parameters(bind)

    def ideal_proba(self,row,weights=None):
        state=Statevector.from_instruction(self.bound_circuit(row,weights))
        p=np.bincount(self.outcomes,weights=state.probabilities(),minlength=len(self.labels))
        return p

    def resources(self):
        return dict(qubits=4,trainable=len(self.weights),frozen_quantum=0,depth=self.circuit.depth(),
                    two_qubit_gates=sum(len(i.qubits)==2 for i in self.circuit.data),
                    objective_evaluations=len(self.history),readout_group_sizes=np.bincount(self.outcomes).tolist(),encoding=self.encoding,
                    compilation='logical decomposed H/P/RY/CX; no backend routing')


class MatchedHead:
    """4 -> tanh(1) -> softmax(K); 5+2K trainable values; same 100 COBYLA calls."""
    def __init__(self,labels=('alfalfa','maize','wheat'),max_evals=100):
        self.labels=np.asarray(labels);self.max_evals=max_evals

    def _proba(self,X,w):
        hidden=np.tanh(X@w[:4]+w[4]); k=len(self.labels)
        return softmax(hidden[:,None]*w[5:5+k]+w[5+k:],axis=1)

    def fit(self,X_train,y_train,seed=0):
        X=np.asarray(X_train); y=np.array([np.flatnonzero(self.labels==v)[0] for v in y_train]); start=perf_counter();self.history=[]
        def objective(w):
            p=self._proba(X,w);loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()
            self.history.append(dict(evaluation=len(self.history)+1,objective=float(loss),seconds=perf_counter()-start))
            return loss
        res=minimize(objective,np.random.default_rng(seed).normal(0,.3,5+2*len(self.labels)),method='COBYLA',options={'maxiter':self.max_evals})
        self.weights=res.x;self.fit_seconds=perf_counter()-start;return self

    def predict_proba(self,X):
        return probabilities(self._proba(np.asarray(X),self.weights),len(X),len(self.labels))

    def predict(self,X): return self.labels[self.predict_proba(X).argmax(axis=1)]

    def resources(self): return dict(trainable=len(self.weights),objective_evaluations=len(self.history))
