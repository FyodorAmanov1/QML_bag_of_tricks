# Foundations and evidence boundaries

## What is a barren plateau?

Specify a family of objectives $C_n(\theta)$, a parameter distribution
$\mathcal P_n$, a derivative coordinate or gradient metric, and normalization.
A commonly studied signature is
$\operatorname{Var}_{\theta\sim\mathcal P_n}[\partial_j C_n]\leq A e^{-an}$,
with mean zero and constants independent of $n$. Other papers bound loss
variance or an expected squared gradient norm. These statements are related
only under their respective hypotheses; keep their metrics distinct.

A useful avoidance bound has inverse-polynomial scale and non-negligible state,
observable, and data prefactors. An average lower bound alone need not imply a
high-probability lower bound for every individual run. It also says nothing by
itself about spurious minima, expressivity, generalization, or total runtime.

Primary sources: [original concentration result](https://doi.org/10.1038/s41467-018-07090-4),
[local-cost results](https://doi.org/10.1038/s41467-021-21728-w), and
[algebraic variance theory](https://doi.org/10.1038/s41467-024-49909-3).

## Why measurements matter

For a bounded measurement with variance $v$, estimating a mean to additive
standard error $\epsilon$ requires order $v/\epsilon^2$ independent shots.
A finite difference with spacing $h$ amplifies independent measurement noise
by $1/(4h^2)$ per side and introduces truncation bias. This project uses $h=10^{-5}$
on exact simulators and $h=0.1$ for sampled experiments. The latter is a disclosed
biased estimator; reducing $h$ at fixed shots can make it worse.

The sampled mode draws computational-basis outcomes from exact SDK output
probabilities using NumPy. It is an end-measurement simulation. It does not
include hardware compilation, readout error, or provider execution. All bundled
observables are diagonal, so one measurement setting suffices. Channel examples
include amplitude damping inside the native evolution.

For independent repetitions $\hat g_{kr}$ at initialization $k$, report within-k
variance $s_w^2$ and variance of the repetition means $s_b^2$. Then
$s_b^2-s_w^2/R$ estimates across-initialization signal variance, subject to
finite-difference bias. Negative estimates are retained. They indicate sampling
uncertainty, not negative physical variance. Exact simulator values provide a
reference unavailable on a large quantum processor.

## Data access and space

An $n$-qubit register carries a vector in dimension $2^n$. Arbitrary classical
amplitude loading can itself require exponential work. Extracting every amplitude
also requires extensive resources. Compare algorithms on the same required
outputs, precision, state-copy access, loading assumptions, and success criterion.

The reported $16\,2^n$ and $16\,4^n$ byte estimates describe one dense complex128
vector/matrix, excluding temporary storage. They are not lower bounds on the
best classical algorithm. Tensor contraction, covariance matrices, excitation
sectors, and causal cones often bypass those dense representations.

[Trainability and classical simulability](https://doi.org/10.1038/s41467-025-63099-6)
explains why these comparisons are essential, including classical surrogates
constructed from quantum experimental data.

## Composing remedies

Re-derive the guarantee after changing the ensemble, objective, input distribution,
or channels. Shared parameters violate independence assumptions in some proofs.
Nonlinear learning losses can suppress or cancel otherwise visible per-example
gradients. Symmetry can help or erase the relevant signal. Local objectives can
change what is learned. Noise can preserve a late-layer gradient while destroying
information about the input.
