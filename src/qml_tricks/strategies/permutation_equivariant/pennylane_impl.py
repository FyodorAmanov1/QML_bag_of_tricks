r"""Permutation-equivariant quantum models: native pennylane implementation.

Published result scope: specified equivariant ensemble.
Implementation boundary: Collective Y rotations and equal-angle all-pairs ZZ interactions commute with permutations. The short circuit verifies symmetry but does not certify convergence to the ensemble used in the theorem.
"""
import numpy as np
import pennylane as qml
from qml_tricks.fixtures import haar
from qml_tricks.core import RunConfig, experiment

CHANNEL = False


def build_circuit(theta, config):
    """Build the native circuit; PennyLane operations are queued in a QNode."""
    n = config.n_qubits
    c = None
    for l in range(config.depth):
        for i in range(n):
            qml.RY(theta[2*l*n] if config.variant == "strategy" else theta[2*l*n+i], wires=i)
        for i in range(n):
            for j in range(i+1,n):
                qml.IsingZZ(theta[2*l*n+1] if config.variant == "strategy" else theta[(2*l*n+n+i+j) % len(theta)], wires=[i, j])
    return c


def _normalize(p):
    p = np.maximum(np.real(p), 0)
    return p / p.sum()


def probabilities(theta, config):
    """Exact simulator probabilities in wire-0-first bit order."""
    dev = qml.device("default.mixed" if CHANNEL else "default.qubit", wires=config.n_qubits)
    @qml.qnode(dev)
    def circuit():
        build_circuit(theta, config)
        return qml.probs(wires=range(config.n_qubits))
    return _normalize(np.asarray(circuit(), dtype=float))



def statevector(theta, config):
    """Native pure-state amplitudes; never a hardware-access assumption."""
    if CHANNEL:
        raise ValueError("Pure-state metric is unavailable for channel recipes")
    dev = qml.device("default.qubit", wires=config.n_qubits)
    @qml.qnode(dev)
    def circuit():
        build_circuit(theta, config)
        return qml.state()
    return np.asarray(circuit(), dtype=complex)



def resources(theta, config):
    """Native logical counts, including preparation and channel operations."""
    with qml.queuing.AnnotatedQueue() as queue:
        build_circuit(theta, config)
    tape = qml.tape.QuantumScript.from_queue(queue)
    return {"qubits": config.n_qubits, "ancillas": 0, "native_operations": len(tape.operations),
            "two_qubit_operations": sum(len(op.wires) == 2 for op in tape.operations),
            "native_depth": tape.graph.get_depth(), "gate_count_stage": "logical, before hardware synthesis"}



def run(config: RunConfig = RunConfig()):
    return experiment("permutation_equivariant", "pennylane", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
