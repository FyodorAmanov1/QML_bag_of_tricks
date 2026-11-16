"""Native PennyLane full-coordinate sequential angle uploads."""
import numpy as np
import pennylane as qml
from .portable_contract import validate


def state(x, theta, scale=1., entangle=True):
    x, theta, n = validate(x, theta, scale, entangle)
    device = qml.device('default.qubit', wires=n)

    @qml.qnode(device)
    def circuit():
        for layer in theta:
            for start in range(0, len(x), n):
                for q, value in enumerate(x[start:start+n]):
                    qml.RY(scale*value, wires=q)
                    qml.RZ(scale*value, wires=q)
                if entangle:
                    for q in range(n-1):
                        qml.CNOT(wires=[q,q+1])
            for q in range(n):
                qml.RY(layer[q,0], wires=q)
                qml.RZ(layer[q,1], wires=q)
        return qml.state()

    return np.asarray(circuit())
