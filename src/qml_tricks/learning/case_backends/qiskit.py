"""Independent Qiskit circuits for reduced application-model adaptations."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def _logical_state(circuit):
    n = circuit.num_qubits
    order = [int(format(i, f'0{n}b')[::-1], 2) for i in range(2**n)]
    return np.asarray(Statevector.from_instruction(circuit))[order]


def kernel_state(x, topology='full'):
    n = len(x)
    c = QuantumCircuit(n)
    if topology in ('x', 'y', 'z'):
        for i in range(n):
            if topology == 'z': c.h(i)
            getattr(c, 'r' + topology)(float(x[i]), i)
    else:
        for i in range(n): c.h(i); c.rz(float(x[i]), i)
        if topology == 'full': edges = [(i,j) for i in range(n) for j in range(i+1,n)]
        else:
            edges = [(i,i+1) for i in range(n-1)]
            if topology == 'circular' and n > 2: edges.append((n-1,0))
        for i,j in edges: c.rzz(float(x[i]*x[j]), i, j)
    return _logical_state(c)


def qvc_state(amplitudes, theta):
    n = theta.shape[1]
    order = [int(format(i, f'0{n}b')[::-1], 2) for i in range(2**n)]
    c = QuantumCircuit(n)
    c.initialize(np.asarray(amplitudes)[order], range(n))
    for layer in theta:
        for i in range(n):
            c.rz(float(layer[i,0]), i); c.ry(float(layer[i,1]), i); c.rz(float(layer[i,2]), i)
        for i in range(n-1): c.cz(i,i+1)
    return _logical_state(c)


def entity_state(angles, mixing, strong):
    n = len(angles)
    c = QuantumCircuit(n)
    edges = [(0,1)] if n == 2 else [(i,(i+1)%n) for i in range(n)]
    for layer in mixing:
        for i in range(n): c.rx(float(angles[i]), i)
        for k,(i,j) in enumerate(edges): c.rzz(float(layer[k]), i,j)
        for i in range(n): c.ry(float(layer[len(edges)+i]), i)
    for i in range(n): c.rx(float(angles[i]), i)
    for layer in strong:
        for i in range(n):
            c.rz(float(layer[i,0]),i); c.ry(float(layer[i,1]),i); c.rz(float(layer[i,2]),i)
        for i in range(n): c.cx(i,(i+1)%n)
    return _logical_state(c)
