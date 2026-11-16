"""SDK-independent matrix contracts and fixtures; no native circuit conversion."""
import numpy as np
from scipy.linalg import expm

PAULIS = (
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], complex),
    np.array([[0, -1j], [1j, 0]], complex),
    np.diag([1, -1]).astype(complex),
)
GENERATORS = np.array([np.kron(a, b) for i, a in enumerate(PAULIS)
                       for j, b in enumerate(PAULIS) if i or j])


def su4_gate(theta):
    theta = np.asarray(theta, float)
    if theta.shape != (15,) or not np.isfinite(theta).all():
        raise ValueError('Require 15 finite generator angles')
    return expm(-0.5j * np.einsum('i,ijk->jk', theta, GENERATORS))


def amplitude_input(features):
    x = np.asarray(features, complex)
    if x.ndim != 1 or len(x) < 3 or len(x) > 1024 or not np.isfinite(x).all() or np.linalg.norm(x) == 0:
        raise ValueError('Require 3..1024 finite, nonzero features')
    n = (len(x) - 1).bit_length()
    return np.pad(x / np.linalg.norm(x), (0, 2**n - len(x)))


def validate_state_angles(amplitudes, angles):
    x, theta = np.asarray(amplitudes, complex), np.asarray(angles, float)
    if x.ndim != 1 or len(x) < 4 or len(x) > 1024 or len(x) & (len(x) - 1):
        raise ValueError('Require a 2..10 qubit input')
    n = (len(x) - 1).bit_length()
    if not np.isfinite(x).all() or not np.isclose(np.linalg.norm(x), 1):
        raise ValueError('Input state must be finite and normalized')
    if theta.ndim != 3 or theta.shape[1:] != (n - 1, 15) or not 1 <= len(theta) <= 30 or not np.isfinite(theta).all():
        raise ValueError('Expected angles[depth=1..30, n-1, 15]')
    return x, theta, n


def last_z(state):
    p = np.abs(np.asarray(state))**2
    return float(p @ (1 - 2 * (np.arange(len(p)) & 1)))


def pauli_state(word, amplitude):
    if not 1 <= len(word) <= 3 or any(p not in 'XYZ' for p in word) or abs(amplitude) > 1:
        raise ValueError('Explicit density fixture supports 1..3 full-weight Paulis')
    operator = np.ones((1, 1), complex)
    for symbol in word:
        operator = np.kron(operator, PAULIS['IXYZ'.index(symbol)])
    return (np.eye(len(operator)) + amplitude * operator) / len(operator)


def validate_rho(rho):
    rho = np.asarray(rho, complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or len(rho) not in (2, 4, 8):
        raise ValueError('Native dense Bell reference supports 1..3 qubits per copy')
    if not np.isfinite(rho).all() or not np.allclose(rho, rho.conj().T) or not np.isclose(np.trace(rho), 1) or np.linalg.eigvalsh(rho).min() < -1e-9:
        raise ValueError('Require a physical density matrix')
    return rho, (len(rho) - 1).bit_length()


def bell_signs(word):
    """Group output bits as A0..An-1, B0..Bn-1; YY has a minus sign."""
    n = len(word)
    bits = (np.arange(4**n)[:, None] >> np.arange(2*n-1, -1, -1)) & 1
    values = np.ones(4**n)
    for j, symbol in enumerate(word):
        u, v = bits[:, j], bits[:, n+j]
        values *= {'X': 1-2*u, 'Y': -(1-2*u)*(1-2*v), 'Z': 1-2*v}[symbol]
    return values
