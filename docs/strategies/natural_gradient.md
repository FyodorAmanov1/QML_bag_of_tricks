# Quantum natural gradient descent

**Family:** Optimization companions · **Result scope:** geometric optimization method; no general barren-plateau avoidance guarantee · **Code:** optimizer companion; exact-simulator metric; not a proven general plateau remedy

## When to use it

Ill-conditioned quantum-state parameterizations with resolvable gradients. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

$g_{ij}=\operatorname{Re}[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle]$, and $\theta^+=\theta-\eta(g+\lambda I)^{-1}\nabla C$. For pure states the QFIM is $F=4g$; switching conventions requires adjusting the learning rate and regularization. Preconditioning alone does not ensure the information needed to estimate these quantities is available with polynomially many shots.

**Required assumptions:** Differentiable normalized pure state; consistently defined Fubini-Study metric; regularization or a pseudoinverse for singular directions; gradient and metric estimation budgets must be affordable.

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## When QNG is useful, and what it does not establish

Use QNG when gradients are statistically resolvable but Euclidean coordinates
give poorly scaled or redundant directions. It is an optimizer companion, not
an independently proven general barren-plateau remedy. A regularized inverse
can amplify gradient estimation noise; a small eigenvalue is not a source of
missing measurement information. Compare wall time, shots and metric work,
not just the number of iterations.

The circuit prepares product RY/RZ states, applies fixed CZ gates, then fixed H
gates to measure mean X in the premeasurement state using computational-basis
probabilities. The fixed gates preserve FS geometry. With parameters ordered
(alpha_0,...,alpha_n-1,phi_0,...,phi_n-1), the analytic metric is
$g=\frac14\operatorname{diag}(1,...,1,\sin^2\alpha_0,...,\sin^2\alpha_{n-1})$.
The implementations compute this through native statevectors rather than
hardcoding the formula; tests compare against the formula and verify invariance
under parameter-dependent global phase.

The default uses lambda=0.001 and solves the regularized linear system without
forming an explicit inverse. It performs 2P+1 native state evaluations per step,
stores a dense Jacobian and metric, and has dense O(P^3) solve cost. Metric
regularization and condition numbers are reported. At alpha=0 a phase direction
is redundant; regularization handles this singularity without inventing a signal.

On hardware, replace the simulator metric with a measured full, block-diagonal,
or diagonal estimator and count its shots and errors. Computational-basis output
probabilities alone do not determine the full pure-state FS metric. Mixed states
require a suitable mixed-state information geometry. The sampled-loss mode here
retains an exact-simulator metric and is explicitly labeled as such.

PennyLane also supplies [QNGOptimizer](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNGOptimizer.html)
and metric transforms. Its lambda and metric-approximation settings must be
matched when comparing with this dense reference implementation. Classical
natural gradient uses a classical Fisher metric; the pure-state quantum metric
used here is FS, with QFIM=4g.

```bash
python -m qml_tricks run natural_gradient --framework pennylane --metric-regularization 0.001
python -m qml_tricks run natural_gradient --framework qiskit --variant baseline
```

## Logic and proof route

Read **Fubini-Study geometry and natural-gradient update; metric approximation discussion** in the [primary publication](https://doi.org/10.22331/q-2020-05-25-269). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

Computes the dense Fubini-Study metric from central derivatives of native SDK statevectors. Strategy uses a regularized QNG update; baseline uses Euclidean gradient descent with identical initialization, circuit, objective and learning rate. Even in sampled-loss mode, the metric is an exact-simulator resource, not a hardware-estimated metric.

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run natural_gradient --framework qiskit --steps 3
python -m qml_tricks run natural_gradient --framework pennylane --steps 3
python -m qml_tricks run natural_gradient --framework cirq --steps 3
python -m qml_tricks sweep natural_gradient --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/natural_gradient/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/natural_gradient/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/natural_gradient/cirq_impl.py). These downloads use the installed shared package.

**Control:** Euclidean gradient descent; same circuit, objective, initialization, learning rate and gradient estimator. QNG adds separately counted exact metric state evaluations.

Output includes a finite training trace, exact coordinate-gradient variance, measurement-noise variance, resource counts, and the implementation boundary. No monotonic improvement or asymptotic separation is expected from every small run. `--shots 4096` samples the exact native output distribution on the classical host; it does not submit a hardware job.

## Learning-task transfer

For a linear empirical objective $C(\theta)=\sum_s w_s\mathrm{Tr}[O_s U_\theta\rho_s U_\theta^\dagger]$, derive the bound for this entire sum. Label-weighted state differences and gradient cancellations can invalidate a per-example statement. A nonlinear cross-entropy or squared loss requires its own chain-rule and signal analysis. The default executable is a small state/moment objective rather than an accuracy benchmark on a real dataset. Classical features can supply preparation angles; quantum-native inputs require copies of the actual state and appropriate purity/projection factors. Do not assume access to their full amplitudes.

## Space and computational-advantage audit

The native register uses $n$ logical qubits and no added ancillas in this example. Exact pure-state simulation stores $16\cdot2^n$ bytes for one complex128 statevector; a dense mixed-state matrix stores $16\cdot4^n$ bytes before temporary arrays. The executable reports both estimates and identifies the simulator regime. Its parameter array is polynomial size; some architectures deliberately contain unused coordinates, so inspect which gates depend on each parameter.

These storage comparisons do not prove a quantum memory separation. Account for preparation, repeat copies, precision, loading classical features, gates, and measured outputs. Compare tensor-network, symmetry-sector, free-fermion, causal-cone, and quantum-data-assisted classical surrogates when applicable. The small bundled instances are classically simulated by construction. **No end-to-end quantum advantage is established by this recipe.**

## Failure modes, composition, and checks

Check zero commutators, irrelevant coordinates, maximally mixed inputs, cancellation between data points, oversized symmetry sectors, and changes of loss support. The designated diagnostic coordinate may have zero gradient even if the published full gradient norm is nonzero. Exact simulation tests wiring and algebra; repeated seeds cannot prove an asymptotic theorem.

Combine methods only after intersecting their state, ensemble, observable, and noise assumptions. Adding parameter tying can destroy independence; adding noise changes a unitary theorem; changing a loss can change its optimum. Use independent derivative checks and cross-SDK probability comparisons before interpreting a training curve.

## Evidence and maintenance

- Primary publication: [10.22331/q-2020-05-25-269](https://doi.org/10.22331/q-2020-05-25-269).
- Result locator: Fubini-Study geometry and natural-gradient update; metric approximation discussion.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
