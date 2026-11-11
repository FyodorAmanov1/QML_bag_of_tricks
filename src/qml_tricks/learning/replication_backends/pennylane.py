"""Independent native circuits used to verify paper-reproduction models."""
import numpy as np
import pennylane as qml

def zz_state(x, repetitions=1):
    n = len(x); dev = qml.device('default.qubit', wires=n)
    edges = [(0,1)] if n==2 else [(n-1,0)]+[(i,i+1) for i in range(n-1)]
    @qml.qnode(dev)
    def circuit():
        for _ in range(repetitions):
            for i in range(n): qml.Hadamard(wires=i)
            for i in range(n): qml.PhaseShift(float(2*x[i]), wires=i)
            for i,j in edges:
                qml.CNOT(wires=[i,j]); qml.PhaseShift(float(2*(np.pi-x[i])*(np.pi-x[j])), wires=j); qml.CNOT(wires=[i,j])
        return qml.state()
    return np.asarray(circuit(),dtype=complex)

def evolve(rho, unitary):
    n = int(np.log2(len(rho))); dev = qml.device('default.mixed', wires=n)
    @qml.qnode(dev)
    def circuit():
        qml.QubitDensityMatrix(rho,wires=range(n))
        qml.QubitUnitary(unitary,wires=range(n))
        return qml.density_matrix(wires=range(n))
    return np.asarray(circuit(),dtype=complex)
