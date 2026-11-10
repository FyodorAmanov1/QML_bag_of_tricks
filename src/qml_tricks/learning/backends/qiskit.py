import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def state(theta, x, n, depth):
    c = QuantumCircuit(n)
    for l in range(depth):
        for i in range(n):
            c.ry(float(x[i]), i)
            c.rz(float(x[i]/2), i)
            c.ry(float(theta[l*n+i]), i)
        for i in range(n-1): c.cz(i,i+1)
    v = np.asarray(Statevector.from_instruction(c))
    order = [int(format(i, f'0{n}b')[::-1],2) for i in range(2**n)]
    return v[order]
