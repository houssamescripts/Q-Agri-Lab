import os

def run_on_ibm_env(circuit, shots=1024):
    """Submit a measured circuit to IBM hardware; credentials remain environment-only."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    token = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("IBM_QUANTUM_INSTANCE")
    if not token:
        raise RuntimeError("Set IBM_CLOUD_API_KEY in the project .env to use IBM hardware.")
    service_args = {"channel": "ibm_quantum_platform", "token": token}
    if instance:
        service_args["instance"] = instance
    service = QiskitRuntimeService(**service_args)

    backend = service.least_busy(operational=True, simulator=False)

    circuit = circuit.copy()
    circuit.measure_all()

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=1
    )

    isa_circuit = pass_manager.run(circuit)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)

    result = job.result()

    return {
        "backend": backend.name,
        "job_id": job.job_id(),
        "counts": result[0].data.meas.get_counts()
    }
