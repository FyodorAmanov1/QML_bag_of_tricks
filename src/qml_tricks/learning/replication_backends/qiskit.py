"""Independent native circuits used to verify paper-reproduction models."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, Operator

def zz_state(x, repetitions=1):
    n = len(x); c = QuantumCircuit(n)
    edges = [(0,1)] if n==2 else [(n-1,0)]+[(i,i+1) for i in range(n-1)]
    for _ in range(repetitions):
        for i in range(n): c.h(i)
        for i in range(n): c.p(float(2*x[i]), i)
        for i,j in edges:
            c.cx(i,j); c.p(float(2*(np.pi-x[i])*(np.pi-x[j])), j); c.cx(i,j)
    order = [int(format(i,f'0{n}b')[::-1],2) for i in range(2**n)]
    return np.asarray(Statevector.from_instruction(c))[order]

def evolve(rho, unitary):
    n = int(np.log2(len(rho)))
    order = [int(format(i,f'0{n}b')[::-1],2) for i in range(2**n)]
    c = QuantumCircuit(n)
    c.unitary(Operator(unitary[np.ix_(order,order)]), range(n))
    out = np.asarray(DensityMatrix(rho[np.ix_(order,order)]).evolve(c))
    return out[np.ix_(order,order)]
