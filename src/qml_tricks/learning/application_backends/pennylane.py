"""Native verification circuits for the application benchmarks."""
import numpy as np
import pennylane as qml


def ramsey_features(phases, frequencies=(1, 2, 3)):
    dev = qml.device('default.qubit', wires=1)
    @qml.qnode(dev)
    def circuit(angle):
        qml.Hadamard(wires=0)
        qml.RZ(angle, wires=0)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0))
    return np.array([[v for frequency in frequencies
                      for v in circuit(float(frequency*phase))]
                     for phase in np.asarray(phases).reshape(-1)], dtype=float)


def bell_probabilities(rho):
    n = int(np.log2(len(rho)))
    if n > 4 or np.shape(rho) != (2**n, 2**n):
        raise ValueError('Small dense verification fixture only')
    dev = qml.device('default.mixed', wires=2*n)
    @qml.qnode(dev)
    def circuit():
        qml.QubitDensityMatrix(np.kron(rho, rho), wires=range(2*n))
        for j in range(n):
            qml.CNOT(wires=[j, n+j])
            qml.Hadamard(wires=j)
        return qml.probs(wires=range(2*n))
    return np.asarray(circuit(), dtype=float)
