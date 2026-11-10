import numpy as np
import pennylane as qml

def state(theta, x, n, depth):
    dev = qml.device('default.qubit', wires=n)
    @qml.qnode(dev)
    def circuit():
        for l in range(depth):
            for i in range(n):
                qml.RY(float(x[i]), wires=i)
                qml.RZ(float(x[i]/2), wires=i)
                qml.RY(float(theta[l*n+i]), wires=i)
            for i in range(n-1): qml.CZ(wires=[i,i+1])
        return qml.state()
    return np.asarray(circuit(),dtype=complex)
