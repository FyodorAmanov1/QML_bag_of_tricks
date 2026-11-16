"""Native Qiskit full-coordinate sequential angle uploads."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from .portable_contract import validate


def state(x, theta, scale=1., entangle=True):
    x, theta, n = validate(x, theta, scale, entangle)
    circuit = QuantumCircuit(n)
    for layer in theta:
        for start in range(0, len(x), n):
            for q, value in enumerate(x[start:start+n]):
                circuit.ry(scale*value, n-1-q)
                circuit.rz(scale*value, n-1-q)
            if entangle:
                for q in range(n-1):
                    circuit.cx(n-1-q, n-2-q)
        for q in range(n):
            circuit.ry(layer[q,0], n-1-q)
            circuit.rz(layer[q,1], n-1-q)
    return np.asarray(Statevector.from_instruction(circuit).data)
