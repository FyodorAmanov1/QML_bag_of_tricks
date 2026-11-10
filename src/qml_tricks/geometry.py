"""Pure-state geometry shared across native SDK statevector evaluators."""
import numpy as np


def fubini_study_metric(statevector, theta, step=1e-5):
    """Dense FS metric via central derivatives, with the phase direction removed.

    Costs 2P+1 exact state evaluations and O(2**n P + P**2) host storage.
    This simulator implementation does not estimate the metric from shots.
    """
    psi = np.asarray(statevector(theta), dtype=complex)
    derivatives = np.empty((len(psi), len(theta)), dtype=complex)
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = step
        derivatives[:, j] = (statevector(theta + delta) - statevector(theta - delta)) / (2*step)
    overlaps = derivatives.conj().T @ psi
    metric = np.real(derivatives.conj().T @ derivatives - np.outer(overlaps, overlaps.conj()))
    return (metric + metric.T)/2


def natural_gradient_direction(metric, cost_gradient, regularization=1e-3):
    """Solve the regularized FS system; never explicitly form its inverse."""
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    return np.linalg.solve(metric + regularization*np.eye(len(cost_gradient)), cost_gradient)
