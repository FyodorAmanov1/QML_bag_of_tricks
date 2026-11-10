r"""Finite local-depth circuits: native qiskit implementation.

Published result scope: local 2-design ensemble.
Implementation boundary: The executable is a one-dimensional bounded-incidence Haar sweep. Its bond dimension is small and it is efficiently simulable; higher-dimensional generalizations require a separate geometry implementation.
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
        for l in range(config.depth):
            for i in range(n-1):
                c.unitary(haar(config.seed, l*n+i), [i+1, i])
                c.ry(theta[(l*n+i) % len(theta)], i)
        return c
    pairs = [(i, i+1) for i in range(n-1)]
    for block, (a,b) in enumerate(pairs):
        c.unitary(haar(config.seed, block), [b, a])
        c.ry(theta[block % len(theta)], a)
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
    return experiment("finite_local_depth", "qiskit", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
