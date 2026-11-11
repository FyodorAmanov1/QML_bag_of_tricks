"""Independent native circuits used to verify paper-reproduction models."""
import numpy as np
import cirq

def zz_state(x, repetitions=1):
    n = len(x); q = cirq.LineQubit.range(n); c = cirq.Circuit()
    edges = [(0,1)] if n==2 else [(n-1,0)]+[(i,i+1) for i in range(n-1)]
    for _ in range(repetitions):
        c.append(cirq.H.on_each(*q))
        for i in range(n): c.append(cirq.ZPowGate(exponent=float(2*x[i]/np.pi))(q[i]))
        for i,j in edges:
            c.append(cirq.CNOT(q[i],q[j])); c.append(cirq.ZPowGate(exponent=float(2*(np.pi-x[i])*(np.pi-x[j])/np.pi))(q[j])); c.append(cirq.CNOT(q[i],q[j]))
    return cirq.Simulator(dtype=np.complex128).simulate(c,qubit_order=q).final_state_vector

def evolve(rho, unitary):
    n = int(np.log2(len(rho))); q = cirq.LineQubit.range(n)
    c = cirq.Circuit(cirq.MatrixGate(unitary).on(*q))
    return cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(c,initial_state=rho,qubit_order=q).final_density_matrix
