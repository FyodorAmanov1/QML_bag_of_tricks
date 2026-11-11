"""Native verification circuits for the application benchmarks."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix


def ramsey_features(phases, frequencies=(1, 2, 3)):
    out = []
    for phase in np.asarray(phases).reshape(-1):
        row = []
        for frequency in frequencies:
            circuit = QuantumCircuit(1)
            circuit.h(0)
            circuit.rz(float(frequency*phase), 0)
            state = np.asarray(Statevector.from_instruction(circuit))
            coherence = np.conj(state[0])*state[1]
            row.extend([2*coherence.real, 2*coherence.imag])
        out.append(row)
    return np.asarray(out)


def bell_probabilities(rho):
    n = int(np.log2(len(rho)))
    if n > 4 or np.shape(rho) != (2**n, 2**n):
        raise ValueError('Small dense verification fixture only')
    circuit = QuantumCircuit(2*n)
    for j in range(n):
        circuit.cx(j, n+j)
        circuit.h(j)
    order = [int(format(i, f'0{2*n}b')[::-1], 2) for i in range(4**n)]
    both = np.kron(rho, rho)
    native = both[np.ix_(order, order)]
    probabilities = DensityMatrix(native).evolve(circuit).probabilities()
    return np.asarray(probabilities)[order]
