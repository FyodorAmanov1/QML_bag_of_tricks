r"""Quantum natural gradient descent: native cirq implementation.

Published result scope: geometric optimization method; no general barren-plateau avoidance guarantee.
Implementation boundary: Computes the dense Fubini-Study metric from central derivatives of native SDK statevectors. Strategy uses a regularized QNG update; baseline uses Euclidean gradient descent with identical initialization, circuit, objective and learning rate. Even in sampled-loss mode, the metric is an exact-simulator resource, not a hardware-estimated metric.
"""
import numpy as np
import cirq
from qml_tricks.fixtures import haar
from qml_tricks.core import RunConfig, experiment

CHANNEL = False


def build_circuit(theta, config):
    """Build the native circuit; PennyLane operations are queued in a QNode."""
    n = config.n_qubits
    q = cirq.LineQubit.range(n)
    c = cirq.Circuit()
    for i in range(n):
        c.append(cirq.ry(theta[i])(q[i]))
        c.append(cirq.rz(theta[n+i])(q[i]))
    for i in range(n-1):
        c.append(cirq.CZ(q[i], q[i+1]))
    for i in range(n):
        c.append(cirq.H(q[i]))
    return c


def _normalize(p):
    p = np.maximum(np.real(p), 0)
    return p / p.sum()


def probabilities(theta, config):
    """Exact simulator probabilities in wire-0-first bit order."""
    c = build_circuit(theta, config)
    order = cirq.LineQubit.range(config.n_qubits)
    if CHANNEL:
        result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(c, qubit_order=order)
        p = np.real(np.diag(result.final_density_matrix))
    else:
        result = cirq.Simulator(dtype=np.complex128).simulate(c, qubit_order=order)
        p = np.abs(result.final_state_vector)**2
    return _normalize(p)



def statevector(theta, config):
    """Native pure-state amplitudes; never a hardware-access assumption."""
    if CHANNEL:
        raise ValueError("Pure-state metric is unavailable for channel recipes")
    c = build_circuit(theta, config)
    return cirq.Simulator(dtype=np.complex128).simulate(c, qubit_order=cirq.LineQubit.range(config.n_qubits)).final_state_vector



def resources(theta, config):
    """Native logical counts, including preparation and channel operations."""
    c = build_circuit(theta, config)
    ops = list(c.all_operations())
    return {"qubits": config.n_qubits, "ancillas": 0, "native_operations": len(ops),
            "two_qubit_operations": sum(len(op.qubits) == 2 for op in ops),
            "native_depth": len(c), "gate_count_stage": "logical, before hardware synthesis"}



def run(config: RunConfig = RunConfig()):
    return experiment("natural_gradient", "cirq", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
