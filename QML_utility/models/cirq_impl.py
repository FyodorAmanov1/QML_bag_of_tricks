"""Independent Cirq construction with an explicit logical qubit order."""
import numpy as np
import cirq
from .numerics import su4_gate, validate_state_angles, validate_rho


def continual_state(amplitudes, angles):
    x, theta, n = validate_state_angles(amplitudes, angles)
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit(cirq.I(q) for q in qubits)
    for layer in theta:
        k = 0
        for offset in (0, 1):
            for i in range(offset, n-1, 2):
                circuit.append(cirq.MatrixGate(su4_gate(layer[k])).on(qubits[i], qubits[i+1]))
                k += 1
    return cirq.Simulator(dtype=np.complex128).simulate(
        circuit, initial_state=x, qubit_order=qubits).final_state_vector


def bell_probabilities(rho):
    rho, n = validate_rho(rho)
    qubits = cirq.LineQubit.range(2*n)
    circuit = cirq.Circuit(cirq.I(q) for q in qubits)
    for i in range(n):
        circuit.append(cirq.CNOT(qubits[i], qubits[n+i]))
        circuit.append(cirq.H(qubits[i]))
    density = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
        circuit, initial_state=np.kron(rho, rho), qubit_order=qubits).final_density_matrix
    return np.diag(density).real


def pauli_channel_probabilities(probabilities):
    """Ideal mixed-unitary system channel, retaining a reference; n <= 3."""
    from .pauli_channel import channel_spec
    p, n, _, operators = channel_spec(probabilities)
    qubits = cirq.LineQubit.range(2*n)
    circuit = cirq.Circuit(cirq.I(q) for q in qubits)
    for i in range(n):
        circuit.append(cirq.H(qubits[i]))
        circuit.append(cirq.CNOT(qubits[i], qubits[n+i]))
    circuit.append(cirq.MixedUnitaryChannel(list(zip(p, operators))).on(*qubits[:n]))
    for i in range(n):
        circuit.append(cirq.CNOT(qubits[i], qubits[n+i]))
        circuit.append(cirq.H(qubits[i]))
    density = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
        circuit, qubit_order=qubits).final_density_matrix
    return np.diag(density).real
