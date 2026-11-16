"""Pauli-channel and Choi conventions; no claim of full device EMEEL replay."""
from itertools import product
import numpy as np
from .numerics import PAULIS


def channel_spec(probabilities):
    p = np.asarray(probabilities, float)
    if p.ndim != 1 or len(p) not in (4,16,64) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-12,rtol=0):
        raise ValueError('Reference requires normalized Pauli probabilities for 1..3 system qubits')
    n = {4:1,16:2,64:3}[len(p)]
    words = [''.join(w) for w in product('IXYZ', repeat=n)]
    matrices = []
    for word in words:
        operator = np.ones((1,1), complex)
        for symbol in word:
            operator = np.kron(operator, PAULIS['IXYZ'.index(symbol)])
        matrices.append(operator)
    return p,n,words,matrices


def eigenvalues_from_error_probabilities(probabilities):
    p,n,words,_ = channel_spec(probabilities)
    signs = np.array([[(-1)**sum(a!='I' and b!='I' and a!=b for a,b in zip(w,e)) for e in words] for w in words])
    return words,signs@p


def eigenvalues_from_bell_probabilities(probabilities):
    p,n,words,_ = channel_spec(probabilities)
    bits = (np.arange(4**n)[:,None] >> np.arange(2*n-1,-1,-1)) & 1
    out = []
    for word in words:
        sign = np.ones(len(p))
        for j,symbol in enumerate(word):
            u,v = bits[:,j],bits[:,n+j]
            # Choi observable is P tensor P^T: no extra YY minus sign.
            if symbol == 'X':sign *= 1-2*u
            if symbol == 'Z':sign *= 1-2*v
            if symbol == 'Y':sign *= (1-2*u)*(1-2*v)
        out.append(p@sign)
    return words,np.array(out)


def remove_known_memory_attenuation(observed, auxiliary, min_abs_auxiliary=.2):
    observed,auxiliary = np.asarray(observed,float),np.asarray(auxiliary,float)
    if observed.shape != auxiliary.shape or not np.isfinite(observed).all() or not np.isfinite(auxiliary).all():
        raise ValueError('Require matching finite coefficient arrays')
    if not 0 < min_abs_auxiliary <= 1 or np.any(np.abs(auxiliary)<min_abs_auxiliary):
        raise ValueError('Unstable auxiliary correction; acquire a reliable calibration')
    return observed/auxiliary  # no silent clipping; estimate uncertainty separately
