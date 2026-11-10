"""Mathematical primitives with explicit conditions and independent tests."""
import numpy as np


def parameter_shift(expectation, theta):
    """Exact identity when each coordinate occurs in one exp(-itP/2) gate.

    Shift the expectation, then apply the classical loss chain rule. Shared
    coordinates require summing occurrence-wise derivatives instead.
    """
    result = []
    for i in range(len(theta)):
        d = np.zeros_like(theta); d[i] = np.pi/2
        result.append((expectation(theta+d)-expectation(theta-d))/2)
    return np.asarray(result)


def spsa(objective, initial, steps=10, seed=7, a=.1, c=.1):
    """Two loss evaluations per update; conditional asymptotic SA guarantees."""
    rng = np.random.default_rng(seed); theta = np.array(initial, float)
    for k in range(steps):
        delta = rng.choice([-1., 1.], size=len(theta))
        ck = c/(k+1)**.101; ak = a/(k+1)**.602
        estimate = (objective(theta+ck*delta)-objective(theta-ck*delta))/(2*ck)*delta
        theta -= ak*estimate
    return theta


def allocate_shots(coefficients, variances, budget):
    """Rounded Neyman allocation for independent terms with known variances.

    At least one shot per term. Pilot-shot costs are additional; estimated
    variances do not inherit the known-variance optimum automatically.
    """
    c, v = np.asarray(coefficients), np.asarray(variances)
    if len(c) != len(v) or budget < len(c) or np.any(v < 0):
        raise ValueError("Invalid independent-term allocation")
    weights = np.abs(c)*np.sqrt(v)
    if not weights.sum(): weights = np.ones(len(c))
    raw = (budget-len(c))*weights/weights.sum()
    count = np.floor(raw).astype(int)+1
    remainder = budget-count.sum()
    count[np.argsort(-(raw-np.floor(raw)), kind="stable")[:remainder]] += 1
    return count


def bit_kernel(x, y, gamma=1.):
    """Strictly positive definite Gaussian kernel on distinct binary vectors."""
    x, y = np.asarray(x), np.asarray(y)
    return np.exp(-gamma*np.sum((x[:, None, :]-y[None, :, :])**2, axis=-1))


def mmd_unbiased(x, y, gamma=1.):
    """Unbiased MMD squared U-statistic; a finite estimate can be negative."""
    if len(x) < 2 or len(y) < 2: raise ValueError("Need two samples per distribution")
    xx, yy, xy = bit_kernel(x,x,gamma), bit_kernel(y,y,gamma), bit_kernel(x,y,gamma)
    return float((xx.sum()-np.trace(xx))/(len(x)*(len(x)-1)) +
                 (yy.sum()-np.trace(yy))/(len(y)*(len(y)-1))-2*xy.mean())


def born_mmd_gradient(sampler, theta, target, count=32, gamma=1.):
    """Unbiased parameter-shift gradient of population MMD squared.

    sampler(t,count) must draw independent samples on EVERY call; every t
    coordinate must occur once in a Pauli rotation. Target is an empirical
    distribution, held fixed during training.
    """
    base = sampler(theta, count); out = []
    for j in range(len(theta)):
        d = np.zeros_like(theta); d[j] = np.pi/2
        plus, minus = sampler(theta+d,count), sampler(theta-d,count)
        out.append(bit_kernel(plus,base,gamma).mean()-bit_kernel(minus,base,gamma).mean()
                   -bit_kernel(plus,target,gamma).mean()+bit_kernel(minus,target,gamma).mean())
    return np.array(out)


def pauli_shadows(state, n, shots, seed=7):
    """Host simulation of random local Pauli measurements of a native state.

    Returns basis indices (X,Y,Z) and eigenvalues. Access to full amplitudes is
    a SIMULATOR convenience, not a hardware data-access promise.
    """
    rng = np.random.default_rng(seed)
    h = np.array([[1,1],[1,-1]])/np.sqrt(2)
    rotations = [h, h@np.diag([1,-1j]), np.eye(2)]
    bases = rng.integers(0,3,(shots,n)); signs = np.empty((shots,n),int)
    for k, basis in enumerate(bases):
        psi = np.array(state).reshape([2]*n)
        for i, b in enumerate(basis):
            psi = np.moveaxis(np.tensordot(rotations[b],psi,axes=(1,i)),0,i)
        p = np.abs(psi.reshape(-1))**2; p /= p.sum()
        index = rng.choice(2**n,p=p)
        signs[k] = 1-2*((index >> np.arange(n-1,-1,-1)) & 1)
    return bases, signs


def shadow_expectation(bases, signs, pauli, groups=1):
    """Median of group means; use theory-sized groups for confidence guarantees."""
    estimate = np.ones(len(bases)); axes = {"X":0,"Y":1,"Z":2}
    if not 1 <= groups <= len(bases): raise ValueError("Invalid number of groups")
    for i, symbol in enumerate(pauli):
        if symbol != "I": estimate *= 3*signs[:,i]*(bases[:,i] == axes[symbol])
    return float(np.median([part.mean() for part in np.array_split(estimate,groups)]))
