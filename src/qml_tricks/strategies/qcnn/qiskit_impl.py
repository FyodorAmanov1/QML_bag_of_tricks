r"""Quantum convolutional neural networks: native qiskit implementation.

Published result scope: initialization ensemble.
Implementation boundary: Haar-dressed convolution and coherent controlled-Y pooling followed by discarding wires. The selected tangent is illustrative; the code does not reproduce the complete parameter ensemble in the theorem.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, Kraus
from qml_tricks.fixtures import haar
from qml_tricks.core import RunConfig, experiment

CHANNEL = False


def build_circuit(theta, config):
    """Build the native circuit; PennyLane operations are queued in a QNode."""
    n = config.n_qubits
    c = QuantumCircuit(n)
    if config.variant == "baseline":
        for i in range(n):
            c.ry(0.2 + 0.1*i, i)
        for l in range(config.depth):
            for i in range(n-1):
                c.unitary(haar(config.seed, l*n+i), [i+1, i])
                c.ry(theta[(l*n+i) % len(theta)], i)
        return c
    for i in range(n):
        c.ry(0.2 + 0.1*i, i)
    active = list(range(n))
    block = 0
    while len(active) > 1:
        for j in range(0, len(active), 2):
            a, b = active[j], active[j+1]
            c.unitary(haar(config.seed, block), [b, a])
            c.ry(theta[block % len(theta)], a)
            c.cry(0.7 + theta[(block+1) % len(theta)], b, a)
            block += 1
        active = active[::2]
    return c


def _normalize(p):
    p = np.maximum(np.real(p), 0)
    return p / p.sum()


def probabilities(theta, config):
    """Exact simulator probabilities in wire-0-first bit order."""
    c = build_circuit(theta, config)
    state = DensityMatrix.from_instruction(c) if CHANNEL else Statevector.from_instruction(c)
    p = state.probabilities()
    # Qiskit indexes qubit zero as the least significant bit.
    order = [int(format(i, f"0{config.n_qubits}b")[::-1], 2) for i in range(len(p))]
    return _normalize(p[order])



def statevector(theta, config):
    """Native pure-state amplitudes; never a hardware-access assumption."""
    if CHANNEL:
        raise ValueError("Pure-state metric is unavailable for channel recipes")
    state = np.asarray(Statevector.from_instruction(build_circuit(theta, config)))
    order = [int(format(i, f"0{config.n_qubits}b")[::-1], 2) for i in range(len(state))]
    return state[order]



def resources(theta, config):
    """Native logical counts, including preparation and channel operations."""
    c = build_circuit(theta, config)
    return {"qubits": c.num_qubits, "ancillas": 0, "native_operations": len(c.data),
            "two_qubit_operations": sum(len(x.qubits) == 2 for x in c.data),
            "native_depth": c.depth(), "gate_count_stage": "logical, before hardware synthesis"}



def run(config: RunConfig = RunConfig()):
    return experiment("qcnn", "qiskit", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
