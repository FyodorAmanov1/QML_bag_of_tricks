"""Measurement records for a restricted published quantum-learning protocol.

Huang et al., Science 376, 1182 (2022), DOI:10.1126/science.abn7293.
The learner gets records and a late Pauli query, never the generator amplitude.
Analytic sampling is efficient because the simulator knows the hidden state.
"""
import numpy as np
from scipy.special import logsumexp

PAULIS = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], complex),
    'Y': np.array([[0, -1j], [1j, 0]], complex),
    'Z': np.diag([1, -1]).astype(complex),
}


def validate_word(word):
    if not word or any(p not in PAULIS for p in word) or set(word) == {'I'}:
        raise ValueError('Need a nonidentity Pauli word')


def pauli_state(word, amplitude):
    """Small dense verification fixture, not a scalable hardware input loader."""
    validate_word(word)
    if abs(amplitude) > 1:
        raise ValueError('State amplitude must lie in [-1, 1]')
    p = np.array([[1.]], complex)
    for letter in word:
        p = np.kron(p, PAULIS[letter])
    return (np.eye(len(p)) + amplitude * p) / len(p)


def bell_signs(bits, query):
    """Canonical bit order: all first-copy bits, then all second-copy bits.

    XX=(-1)^a, ZZ=(-1)^b, YY=-(-1)^(a+b) for each Bell outcome (a,b).
    """
    validate_word(query)
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] != 2*len(query):
        raise ValueError('Wrong Bell record width')
    n = len(query)
    exponent = np.zeros(len(bits), dtype=int)
    for j, letter in enumerate(query):
        if letter in ('X', 'Y'):
            exponent += bits[:, j]
        if letter in ('Z', 'Y'):
            exponent += bits[:, n+j]
        if letter == 'Y':
            exponent += 1
    return 1 - 2*(exponent % 2)


def bell_distribution(word, amplitude):
    """Exact small-system Born probabilities in the canonical output order."""
    validate_word(word)
    if abs(amplitude) > 1:
        raise ValueError('Invalid amplitude')
    n = len(word)
    if n > 4:
        raise ValueError('Dense diagnostic is capped at four input qubits')
    bits = ((np.arange(4**n)[:, None] >> np.arange(2*n-1, -1, -1)) & 1)
    return (1 + amplitude**2 * bell_signs(bits, word)) / 4**n


def acquire_bell(word, amplitude, copies, rng, readout_error=0.):
    """Query-independent Bell acquisition on pairs of independent state copies.

    For rho=(I+aP)/2**n the Born law is (1+a*a*g_P)/4**n.
    Sample that law directly; no classical learner receives a or this sampler.
    Per-bit readout flips are an explicit toy noise model, not device calibration.
    """
    validate_word(word)
    if copies < 2 or copies % 2 or not 0 <= readout_error < .5 or abs(amplitude) > 1:
        raise ValueError('Need even copy count, physical amplitude and readout error')
    n = len(word)
    bits = rng.integers(0, 2, (copies//2, 2*n), dtype=np.int8)
    target = 2*rng.binomial(1, (1+amplitude**2)/2, copies//2)-1
    j = next(i for i, letter in enumerate(word) if letter != 'I')
    pivot = j if word[j] in ('X', 'Y') else n+j
    bits[:, pivot] ^= (bell_signs(bits, word) != target).astype(np.int8)
    bits ^= (rng.random(bits.shape) < readout_error).astype(np.int8)
    return bits


def acquire_local(word, amplitude, copies, rng, readout_error=0.):
    """Uniform single-copy local Pauli measurements, chosen BEFORE the query.

    The state simulator uses the hidden word to generate measurement outcomes;
    it never uses it to choose the measurement bases.
    """
    validate_word(word)
    if copies < 1 or abs(amplitude) > 1 or not 0 <= readout_error < .5:
        raise ValueError('Invalid acquisition configuration')
    n = len(word)
    bases = rng.integers(0, 3, (copies, n), dtype=np.int8)
    outcomes = 2*rng.integers(0, 2, (copies, n), dtype=np.int8)-1
    active = [j for j, p in enumerate(word) if p != 'I']
    axes = np.array(['XYZ'.index(word[j]) for j in active])
    matched = np.all(bases[:, active] == axes, axis=1)
    rows = np.flatnonzero(matched)
    sign = 2*rng.binomial(1, (1+amplitude)/2, len(rows))-1
    other = np.prod(outcomes[rows][:, active[:-1]], axis=1, dtype=int)
    outcomes[rows, active[-1]] = sign*other
    outcomes *= 1-2*(rng.random(outcomes.shape) < readout_error).astype(np.int8)
    return bases, outcomes


def local_counts(bases, outcomes, query):
    validate_word(query)
    active = [j for j, p in enumerate(query) if p != 'I']
    axes = np.array(['XYZ'.index(query[j]) for j in active])
    matched = np.all(bases[:, active] == axes, axis=1)
    values = np.prod(outcomes[matched][:, active], axis=1, dtype=int)
    return int(np.sum(values == 1)), int(np.sum(values == -1))


def local_features(plus, minus, weight, copies):
    """All informative counts plus convenient derived features; no labels."""
    hits = plus + minus
    return [plus, minus, hits, abs(plus-minus)/max(hits, 1),
            abs(plus-minus)*3.**weight/copies]


def local_log_likelihood_ratio(plus, minus, magnitude):
    """Optimal equal-prior classifier for THESE random local measurements.

    H0: a=0; H1: a=+/-magnitude with equal sign probability. Unmatched rows
    have identical likelihood under both hypotheses and cancel exactly.
    """
    if not 0 < magnitude < 1:
        raise ValueError('Need a magnitude strictly between zero and one')
    plus, minus = np.asarray(plus), np.asarray(minus)
    a = plus*np.log1p(magnitude) + minus*np.log1p(-magnitude)
    b = minus*np.log1p(magnitude) + plus*np.log1p(-magnitude)
    return logsumexp(np.stack([a, b]), axis=0)-np.log(2.)
