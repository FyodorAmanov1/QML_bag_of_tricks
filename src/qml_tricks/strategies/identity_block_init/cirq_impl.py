r"""Identity-block initialization: native cirq implementation.

Published result scope: initialization only; shallow-block conditions.
Implementation boundary: Uses B=RY(c) CZ RX(a) per shallow block, with independently trainable inverse copies, matched only at initialization. Prepares |+> on wire zero and reads Z zero. Exact unitary cancellation is tested. A nonzero gradient is task dependent; no universal lower bound is asserted.
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
    c.append(cirq.H(q[0]))
    for l in range(config.depth):
        base = 4*l*n
        for i in range(n):
            c.append(cirq.rx(theta[base+i])(q[i]))
        for i in range(n-1):
            c.append(cirq.CZ(q[i], q[i+1]))
        for i in range(n):
            c.append(cirq.ry(theta[base+n+i])(q[i]))
        for i in reversed(range(n)):
            c.append(cirq.ry(-theta[base+2*n+i])(q[i]))
        for i in reversed(range(n-1)):
            c.append(cirq.CZ(q[i], q[i+1]))
        for i in reversed(range(n)):
            c.append(cirq.rx(-theta[base+3*n+i])(q[i]))
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
    return experiment("identity_block_init", "cirq", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
