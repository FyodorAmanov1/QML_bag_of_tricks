"""Native verification circuits for the application benchmarks."""
import numpy as np
import cirq


def ramsey_features(phases, frequencies=(1, 2, 3)):
    q = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=np.complex128)
    out = []
    for phase in np.asarray(phases).reshape(-1):
        row = []
        for frequency in frequencies:
            circuit = cirq.Circuit(cirq.H(q), cirq.rz(float(frequency*phase))(q))
            state = simulator.simulate(circuit, qubit_order=[q]).final_state_vector
            coherence = np.conj(state[0])*state[1]
            row.extend([2*coherence.real, 2*coherence.imag])
        out.append(row)
    return np.asarray(out)


def bell_probabilities(rho):
    n = int(np.log2(len(rho)))
    if n > 4 or np.shape(rho) != (2**n, 2**n):
        raise ValueError('Small dense verification fixture only')
    qubits = cirq.LineQubit.range(2*n)
    circuit = cirq.Circuit()
    for j in range(n):
        circuit.append(cirq.CNOT(qubits[j], qubits[n+j]))
        circuit.append(cirq.H(qubits[j]))
    result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
        circuit, initial_state=np.kron(rho, rho), qubit_order=qubits)
    return np.diag(result.final_density_matrix).real
