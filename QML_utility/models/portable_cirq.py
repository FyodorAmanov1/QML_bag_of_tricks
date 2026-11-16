"""Native Cirq full-coordinate sequential angle uploads."""
import numpy as np
import cirq
from .portable_contract import validate


def state(x, theta, scale=1., entangle=True):
    x, theta, n = validate(x, theta, scale, entangle)
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit(cirq.I(q) for q in qubits)
    for layer in theta:
        for start in range(0, len(x), n):
            for q, value in enumerate(x[start:start+n]):
                circuit.append(cirq.ry(scale*value)(qubits[q]))
                circuit.append(cirq.rz(scale*value)(qubits[q]))
            if entangle:
                for q in range(n-1):
                    circuit.append(cirq.CNOT(qubits[q],qubits[q+1]))
        for q in range(n):
            circuit.append(cirq.ry(layer[q,0])(qubits[q]))
            circuit.append(cirq.rz(layer[q,1])(qubits[q]))
    return cirq.Simulator(dtype=np.complex128).simulate(circuit,qubit_order=qubits).final_state_vector
