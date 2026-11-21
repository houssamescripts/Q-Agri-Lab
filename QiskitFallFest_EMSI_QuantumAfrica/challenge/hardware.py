"""Explicitly opt-in inference only. No import or credentials on the normal path."""
import os
import numpy as np


def infer_once(trained_model,row,backend_name,*,approved=False,shots=512):
    if not approved: raise PermissionError('Obtain approval for this hardware job first')
    if not hasattr(trained_model,'model'): raise ValueError('Train the model locally first')
    if not backend_name: raise ValueError('Choose and inspect a backend first')
    if not 1<=shots<=2048: raise ValueError('Use 1–2048 inference shots')
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.executor_sampler import Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    # Credentials stay in the user's environment, never in the notebook or outputs.
    service=QiskitRuntimeService(channel='ibm_quantum_platform',token=os.environ['QISKIT_IBM_TOKEN'])
    backend=service.backend(backend_name)
    circuit=trained_model.bound_circuit(row);circuit.measure_all()
    isa=generate_preset_pass_manager(backend=backend,optimization_level=2,seed_transpiler=42).run(circuit)
    job=Sampler(mode=backend).run([isa],shots=shots)
    print('Inference-only job:',job.job_id(),'backend:',backend_name,flush=True)
    counts=job.result()[0].data.meas.get_counts()
    probabilities=np.zeros(len(trained_model.labels))
    for bits,count in counts.items():probabilities[trained_model.outcomes[int(bits.replace(' ',''),2)]]+=count
    probabilities/=sum(counts.values())
    return dict(job_id=job.job_id(),backend=backend_name,counts=counts,
                predicted_class=str(trained_model.labels[probabilities.argmax()]),
                sampled_probabilities=probabilities.tolist(),ideal_probabilities=trained_model.ideal_proba(row).tolist(),
                scope='One fixed-model inference example; not a model accuracy estimate')
