"""Shared numerical fixtures; circuit construction belongs to each SDK module."""
import numpy as np


def haar(seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 813]))
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(z)
    return q @ np.diag(np.diag(r) / np.abs(np.diag(r)))


def initial_parameters(strategy, n, depth, seed, variant):
    rng = np.random.default_rng(seed)
    count = 4*n*depth if strategy == "identity_block_init" else 2*n if strategy == "natural_gradient" else n if strategy == "near_identity_encoding" else n * depth * 2
    theta = rng.uniform(-np.pi, np.pi, count)
    if strategy == "near_identity_encoding":
        return np.full(count, np.pi/2)
    if variant == "baseline":
        return theta
    if strategy == "identity_block_init":
        for l in range(depth):
            base = 4*l*n
            theta[base+2*n:base+3*n] = theta[base+n:base+2*n]
            theta[base+3*n:base+4*n] = theta[base:base+n]
    if strategy == "gaussian_init":
        theta = rng.normal(0, 1 / np.sqrt(8 * depth * n), count)
    if strategy == "gaussian_mixture":
        # Observable Z^n: G1 is an allowed last-layer choice for I/Z sites.
        theta = rng.normal(0, 1 / np.sqrt(2 * depth * n), count)
    if strategy == "constrained_hva":
        theta = rng.uniform(0, 0.01 / (n * count), count)
    if strategy in {"qcnn", "mps", "ttn", "mera", "finite_local_depth"}:
        theta = np.zeros(count)
    return theta


def terms(strategy, n, variant):
    """Diagonal observable: (coefficient, measured wires), big-endian fixtures."""
    if strategy == "local_loss" and variant == "baseline":
        return "projector"
    if strategy == "gaussian_mixture":
        return [(1.0, tuple(range(n)))]
    if strategy in {"qcnn", "constrained_hva", "nonunital_noise", "identity_block_init"}:
        return [(1.0, (0,))]
    if strategy == "low_excitation":
        return [(float(i + 1) / n, (i,)) for i in range(n)]
    if strategy == "mmd_loss":
        # Degree-one polynomial kernel: mean squared one-bit moment error.
        return "mmd_global" if variant == "baseline" else "mmd"
    if strategy == "engineered_dissipation":
        return [(-1 / n, (i,)) for i in range(n)]
    if strategy == "permutation_equivariant":
        return [(1 / n, (i,)) for i in range(n)]
    return [(1 / n, (i,)) for i in range(n)]


def values(strategy, n, variant):
    bits = ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    z = 1 - 2 * bits
    observable = terms(strategy, n, variant)
    if observable == "projector":
        return 1.0 - np.all(bits == 0, axis=1).astype(float)
    if isinstance(observable, str) and observable.startswith("mmd"):
        return z
    result = np.zeros(2**n)
    for coefficient, wires in observable:
        result += coefficient * np.prod(z[:, wires], axis=1)
    if strategy == "local_loss":
        result = (1 - result) / 2
    return result


def loss_from_probabilities(probabilities, strategy, n, variant):
    if strategy == "mmd_loss" and variant == "baseline":
        weights = np.array([i.bit_count() for i in range(2**n)])
        target = 0.25**weights * 0.75**(n - weights)
        return float(np.sum((probabilities - target)**2))
    v = values(strategy, n, variant)
    moments = probabilities @ v
    return float(np.mean((moments - 0.5)**2)) if strategy == "mmd_loss" else float(moments)
