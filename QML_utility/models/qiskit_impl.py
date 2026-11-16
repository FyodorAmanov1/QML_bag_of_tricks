"""Independent Qiskit construction; matrices use logical big-endian ordering."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from .numerics import su4_gate, validate_state_angles, validate_rho


def continual_state(amplitudes, angles):
    x, theta, n = validate_state_angles(amplitudes, angles)
    circuit = QuantumCircuit(n)
    for layer in theta:
        k = 0
        for offset in (0, 1):
            for i in range(offset, n-1, 2):
                # Qiskit's first qarg is the least-significant matrix subsystem.
                circuit.unitary(su4_gate(layer[k]), [n-2-i, n-1-i])
                k += 1
    return np.asarray(Statevector(x).evolve(circuit).data)


def bell_probabilities(rho):
    rho, n = validate_rho(rho)
    circuit = QuantumCircuit(2*n)
    for i in range(n):
        a, b = 2*n-1-i, n-1-i
        circuit.cx(a, b)
        circuit.h(a)
    return np.asarray(DensityMatrix(np.kron(rho, rho)).evolve(circuit).probabilities())


def pauli_channel_probabilities(probabilities):
    """Ideal system-reference acquisition, not a device EMEEL implementation.

    Errors are IXYZ lexicographic; output bits are system then reference,
    both in logical big-endian order. Explicit enumeration caps n at three.
    """
    from .pauli_channel import channel_spec
    p, n, words, _ = channel_spec(probabilities)
    prepare = QuantumCircuit(2*n)
    for i in range(n):
        a, b = 2*n-1-i, n-1-i
        prepare.h(a)
        prepare.cx(a, b)
    out = np.zeros(4**n)
    for weight, word in zip(p, words):
        if weight == 0:
            continue
        circuit = prepare.copy()
        for i, symbol in enumerate(word):
            if symbol != 'I':
                getattr(circuit, symbol.lower())(2*n-1-i)
        for i in range(n):
            a, b = 2*n-1-i, n-1-i
            circuit.cx(a, b)
            circuit.h(a)
        out += weight*Statevector.from_instruction(circuit).probabilities()
    return out
