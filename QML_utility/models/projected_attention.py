"""Classical Gaussian aggregation of measured scalar q/k and value features.

This is the postmeasurement QSANN component, not a complete trained NLP model.
"""
import numpy as np


def gaussian_projected_attention(query, key, values, key_mask=None):
    q, k, v = np.asarray(query, float), np.asarray(key, float), np.asarray(values, float)
    if q.ndim != 1 or k.ndim != 1 or not len(q) or not len(k) or v.ndim != 2 or v.shape[0] != len(k) or not v.shape[1]:
        raise ValueError('Require q[queries], k[keys], values[keys, features]')
    if not all(np.isfinite(x).all() for x in (q,k,v)) or np.max(np.abs(q)) > 1+1e-9 or np.max(np.abs(k)) > 1+1e-9:
        raise ValueError('Require finite features and q/k Pauli expectations within [-1,1]')
    mask = np.ones(len(k), dtype=bool) if key_mask is None else np.asarray(key_mask)
    if mask.dtype != bool or mask.shape != k.shape or not mask.any():
        raise ValueError('Require a boolean key mask with at least one valid token')
    weights = np.exp(-(q[:,None]-k[None,:])**2)*mask[None,:]
    weights /= weights.sum(axis=1, keepdims=True)
    return weights@v, weights
