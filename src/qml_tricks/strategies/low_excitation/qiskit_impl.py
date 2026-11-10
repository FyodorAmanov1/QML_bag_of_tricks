r"""Fixed low-excitation sectors: native qiskit implementation.

Published result scope: specified sector ensemble.
Implementation boundary: Prepares one excitation and uses matched XX/YY rotations. Two layers need not mix to the theorem ensemble; exact excitation conservation is tested separately from trainability.
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
    c.x(0)
    for l in range(config.depth):
        if config.variant == "baseline":
            for i in range(n):
                c.ry(theta[2*l*n+n+i], i)
        for i in range(n-1):
            c.rxx(theta[2*l*n+i], i, i+1)
            c.ryy(theta[2*l*n+i], i, i+1)
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
    return experiment("low_excitation", "qiskit", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
