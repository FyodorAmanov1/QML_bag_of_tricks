"""Independent PennyLane circuits for reduced application-model adaptations."""
import numpy as np
import pennylane as qml


def kernel_state(x, topology='full'):
    n = len(x)
    dev = qml.device('default.qubit', wires=n)
    @qml.qnode(dev)
    def circuit():
        if topology in ('x', 'y', 'z'):
            for i in range(n):
                if topology == 'z': qml.Hadamard(wires=i)
                {'x':qml.RX,'y':qml.RY,'z':qml.RZ}[topology](float(x[i]),wires=i)
        else:
            for i in range(n): qml.Hadamard(wires=i); qml.RZ(float(x[i]),wires=i)
            if topology == 'full': edges = [(i,j) for i in range(n) for j in range(i+1,n)]
            else:
                edges = [(i,i+1) for i in range(n-1)]
                if topology == 'circular' and n > 2: edges.append((n-1,0))
            for i,j in edges: qml.IsingZZ(float(x[i]*x[j]),wires=[i,j])
        return qml.state()
    return np.asarray(circuit(),dtype=complex)


def qvc_state(amplitudes, theta):
    n = theta.shape[1]
    dev = qml.device('default.qubit',wires=n)
    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(amplitudes,wires=range(n))
        for layer in theta:
            for i in range(n):
                qml.RZ(float(layer[i,0]),wires=i); qml.RY(float(layer[i,1]),wires=i); qml.RZ(float(layer[i,2]),wires=i)
            for i in range(n-1): qml.CZ(wires=[i,i+1])
        return qml.state()
    return np.asarray(circuit(),dtype=complex)


def entity_state(angles, mixing, strong):
    n = len(angles)
    dev = qml.device('default.qubit',wires=n)
    @qml.qnode(dev)
    def circuit():
        edges = [(0,1)] if n == 2 else [(i,(i+1)%n) for i in range(n)]
        for layer in mixing:
            for i in range(n): qml.RX(float(angles[i]),wires=i)
            for k,(i,j) in enumerate(edges): qml.IsingZZ(float(layer[k]),wires=[i,j])
            for i in range(n): qml.RY(float(layer[len(edges)+i]),wires=i)
        for i in range(n): qml.RX(float(angles[i]),wires=i)
        for layer in strong:
            for i in range(n):
                qml.RZ(float(layer[i,0]),wires=i); qml.RY(float(layer[i,1]),wires=i); qml.RZ(float(layer[i,2]),wires=i)
            for i in range(n): qml.CNOT(wires=[i,(i+1)%n])
        return qml.state()
    return np.asarray(circuit(),dtype=complex)
