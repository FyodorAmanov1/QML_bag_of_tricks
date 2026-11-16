"""Small-register engineering model contract; no paper-replication claim."""
import numpy as np


def validate(x, theta, scale, entangle):
    x, theta = np.asarray(x, float), np.asarray(theta, float)
    if x.ndim != 1 or not 1 <= len(x) <= 64 or not np.isfinite(x).all():
        raise ValueError('Require 1..64 finite features; features are never truncated')
    if theta.ndim != 3 or theta.shape[2] != 2 or not 1 <= theta.shape[0] <= 4 or not 1 <= theta.shape[1] <= 6:
        raise ValueError('theta must have shape (1..4 layers, 1..6 qubits, 2 rotations)')
    if not np.isfinite(theta).all() or not np.isfinite(scale) or not 0 < scale <= 4 or type(entangle) is not bool:
        raise ValueError('Require finite angles, 0 < scale <= 4, boolean entangle')
    return x, theta, theta.shape[1]


def local_features(state):
    """Z_0..Z_n-1, X_0..X_n-1, Y_0..Y_n-1 in logical big-endian order."""
    state = np.asarray(state, complex)
    n = int(np.log2(len(state)))
    if state.ndim != 1 or len(state) != 2**n or not np.isclose(np.linalg.norm(state), 1):
        raise ValueError('Require a normalized statevector')
    indices = np.arange(len(state))
    values = [[], [], []]
    for q in range(n):
        bit = 1 << (n-1-q)
        zero = indices[(indices & bit) == 0]
        product = np.vdot(state[zero], state[zero | bit])
        values[0].append(np.sum(np.abs(state[zero])**2)-np.sum(np.abs(state[zero | bit])**2))
        values[1].append(2*product.real)
        values[2].append(2*product.imag)
    return np.asarray(values).reshape(-1)
