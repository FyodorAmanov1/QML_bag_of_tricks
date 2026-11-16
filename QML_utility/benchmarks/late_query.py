"""Restricted late-query audit using exact sufficient statistics, not QPU shots."""
import argparse
import json
from pathlib import Path
import platform
import hashlib
import gzip
import numpy as np


def mixture_llr(plus, minus, signal):
    a = plus*np.log1p(signal) + minus*np.log1p(-signal)
    b = minus*np.log1p(signal) + plus*np.log1p(-signal)
    return np.logaddexp(a, b) - np.log(2.)


def balanced_accuracy(labels, predictions):
    return float(np.mean([np.mean(predictions[labels == y] == y) for y in (0, 1)]))


def audit(n=8, copies=512, states=2000, magnitude=.8, readout_error=0., seed=0):
    if type(n) is not int or not 1 <= n <= 30 or copies < 2 or copies % 2 or states < 20 or states % 2:
        raise ValueError('Require n=1..30, even copies >=2 and even states >=20')
    if not 0 < magnitude < 1 or not 0 <= readout_error < .5:
        raise ValueError('Invalid magnitude/readout error')
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat([0, 1], states//2))
    query_axes = rng.integers(0, 3, (states, n))  # X, Y, Z; revealed only after acquisition
    amplitude = labels * rng.choice([-magnitude, magnitude], states)
    local_signal = amplitude * (1-2*readout_error)**n
    # XX uses the A bit, ZZ the B bit, YY both; independent measured-bit errors.
    bell_signal = magnitude**2 * (1-2*readout_error)**(n + np.sum(query_axes == 1, axis=1))
    rounds = copies//2
    bell_plus = rng.binomial(rounds, (1 + labels*bell_signal)/2)
    bell_llr = bell_plus*np.log1p(bell_signal) + (rounds-bell_plus)*np.log1p(-bell_signal)
    hits = rng.binomial(copies, 3.**(-n), states)
    local_plus = rng.binomial(hits, (1 + local_signal)/2)
    local_llr = mixture_llr(local_plus, hits-local_plus, magnitude*(1-2*readout_error)**n)
    early_plus = rng.binomial(copies, (1 + local_signal)/2)
    early_llr = mixture_llr(early_plus, copies-early_plus, magnitude*(1-2*readout_error)**n)
    # A single local hit is uninformative for the symmetric +/- hypothesis.
    # Keep floating-point roundoff from deciding analytically tied likelihoods.
    predictions = {'two_copy_bell': (bell_llr > 1e-12).astype(int),
                   'optimal_random_local_records': (local_llr > 1e-12).astype(int),
                   'query_known_early': (early_llr > 1e-12).astype(int)}
    return dict(configuration=dict(n=n, state_copies_per_example=copies, bell_rounds=rounds,
                                    states=states, magnitude=magnitude, readout_error=readout_error, seed=seed),
                scores={name: balanced_accuracy(labels, pred) for name, pred in predictions.items()},
                expected_local_matches=copies*3.**(-n),
                local_any_hit_accuracy_upper_bound=.5-.5*np.expm1(copies*np.log1p(-3.**(-n))),
                records=dict(labels=labels.tolist(), queries=[''.join('XYZ'[j] for j in row) for row in query_axes],
                             bell_plus=bell_plus.tolist(), local_hits=hits.tolist(),
                             local_plus=local_plus.tolist(), early_plus=early_plus.tolist(),
                             predictions={k: v.tolist() for k, v in predictions.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[2, 4, 8, 12])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--readout-error', type=float, default=0.)
    p.add_argument('--copies', type=int, default=512)
    p.add_argument('--states', type=int, default=2000)
    p.add_argument('--magnitude', type=float, default=.8)
    args = p.parse_args()
    runs = [audit(n=n, seed=seed, copies=args.copies, states=args.states, magnitude=args.magnitude,
                  readout_error=args.readout_error) for n in args.sizes for seed in args.seeds]
    summary = []
    for n in args.sizes:
        selected = [r for r in runs if r['configuration']['n'] == n]
        summary.append(dict(n=n, balanced_accuracy_mean={
            name: float(np.mean([r['scores'][name] for r in selected])) for name in selected[0]['scores']}))
    result = dict(scope='Restricted analytic protocol audit, not a full paper or hardware reproduction',
                  data_access='Quantum state family; pre-query random local acquisition versus two-copy Bell acquisition',
                  inference='Both rules know magnitude and priors; optimal for their specified sufficient statistics',
                  resources='Equal state copies; Bell needs 2n simultaneous coherent qubits. No loading CSV data or QPU timing measured.',
                  simulation='Exact binomial sufficient-statistic sampling; simulator knows hidden family, inference receives counts only',
                  python=platform.python_version(), numpy=np.__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=summary, runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2)+'\n').encode('utf-8')
    if args.output.suffix == '.gz':
        args.output.write_bytes(gzip.compress(payload, mtime=0))
    else:
        args.output.write_bytes(payload)
    print(json.dumps(dict(output=str(args.output), summary=summary), indent=2))


if __name__ == '__main__':
    main()
