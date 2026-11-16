"""Independent PennyLane construction, logical wire zero most significant."""
import numpy as np
import pennylane as qml
from .numerics import su4_gate, validate_state_angles, validate_rho


def continual_state(amplitudes, angles):
    x, theta, n = validate_state_angles(amplitudes, angles)
    dev = qml.device('default.qubit', wires=n)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(x, wires=range(n))
        for layer in theta:
            k = 0
            for offset in (0, 1):
                for i in range(offset, n-1, 2):
                    qml.QubitUnitary(su4_gate(layer[k]), wires=[i, i+1])
                    k += 1
        return qml.state()

    return np.asarray(circuit())


def bell_probabilities(rho):
    rho, n = validate_rho(rho)
    dev = qml.device('default.mixed', wires=2*n)

    @qml.qnode(dev)
    def circuit():
        qml.QubitDensityMatrix(np.kron(rho, rho), wires=range(2*n))
        for i in range(n):
            qml.CNOT(wires=[i, n+i])
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(2*n))

    return np.asarray(circuit())


def pauli_channel_probabilities(probabilities):
    """Ideal Pauli channel on system halves of Bell pairs; n <= 3."""
    from .pauli_channel import channel_spec
    p, n, _, operators = channel_spec(probabilities)
    dev = qml.device('default.mixed', wires=2*n)

    @qml.qnode(dev)
    def circuit():
        for i in range(n):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, n+i])
        qml.QubitChannel([np.sqrt(weight)*operator for weight, operator in zip(p, operators)],
                         wires=range(n))
        for i in range(n):
            qml.CNOT(wires=[i, n+i])
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(2*n))

    return np.asarray(circuit())
