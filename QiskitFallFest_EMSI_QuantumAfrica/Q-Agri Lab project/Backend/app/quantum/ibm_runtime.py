"""Optional IBM Quantum Runtime support; no IBM imports needed for local use."""
import os


def runtime_service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("QISKIT_IBM_TOKEN")
    if not token:
        raise RuntimeError("IBM credentials are not configured")
    kwargs = {"channel": "ibm_quantum_platform", "token": token}
    if os.getenv("IBM_QUANTUM_INSTANCE"):
        kwargs["instance"] = os.environ["IBM_QUANTUM_INSTANCE"]
    return QiskitRuntimeService(**kwargs)


def run_on_ibm_env(circuit, shots=1024):
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    service = runtime_service()
    backend = service.least_busy(operational=True, simulator=False)
    measured = circuit.copy()
    measured.measure_all()
    isa_circuit = generate_preset_pass_manager(backend=backend, optimization_level=1).run(measured)
    job = SamplerV2(mode=backend).run([isa_circuit], shots=shots)
    result = job.result()
    return {"backend": backend.name, "job_id": job.job_id(), "counts": result[0].data.meas.get_counts()}
