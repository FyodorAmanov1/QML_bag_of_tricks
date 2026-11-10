# Near-identity data encoding

**Family:** Encoding · **Result scope:** analytical special case; no universal encoding guarantee · **Code:** analytically checked product-encoder special case; bounded features required

## When to use it

Bounded classical features and tunable weak quantum embeddings. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

For $C=n^{-1}\sum_i\cos(\theta_i+\epsilon x_i)$, $\partial_i C=-\sin(\theta_i+\epsilon x_i)/n$. At $\theta_i=\pi/2$ and $0\leq\epsilon<\pi/2$, $|\partial_i C|\geq\cos(\epsilon)/n$. This direct calculation establishes a signal for this example, not for arbitrary encoders, datasets, or losses. The primary paper motivates the need to audit data encoding; this particular bound is derived here.

**Required assumptions:** Bounded features $|x_i|\leq1$; $E_\epsilon(x)=\bigotimes_i R_y(\epsilon x_i)$; reference trainable angles $\theta_i=\pi/2$; normalized mean-Z objective.

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## Encoding versus initialization

This guide controls the **data** gates $E_\epsilon(x)$. Gaussian initialization
controls the starting **trainable** angles, and identity blocks pair trainable
branches. These are different decisions and can be combined only after checking
the actual data-dependent gradient.

For a globally near-identity encoder, the operator-norm telescoping bound is
$\|E_\epsilon(x)-I\|\leq\frac{|\epsilon|}{2}\sum_i|x_i|$.
Small individual angles do not imply a small global distance at growing $n$;
for a fixed global budget, scale epsilon with the feature L1 norm. The bundled
nonzero-gradient example initializes the trainable readout at pi/2, not at the
identity. Initializing that readout at zero can instead create a stationary point.
This reference initialization is deterministic, so its across-seed derivative
variance is zero even though the derivative is nonzero. Read the reported
mean-squared coordinate signal and the analytical bound; a zero variance without
a zero-mean condition is not a barren-plateau diagnosis.

Too small an encoding scale erases distinguishability: the overlap between two
encoded product states is $\prod_i\cos^2(\epsilon(x_i-y_i)/2)$. At epsilon zero,
the model contains no information about the features even though the trainable
readout can have a large gradient. Select scale using both signal and validation
performance; this implementation is classically simulable.

```bash
python -m qml_tricks run near_identity_encoding --framework cirq --encoding-scale 0.1 --features -1 -0.2 0.3 1
```

## Logic and proof route

Read **Data-encoding trainability analysis; the product-encoder derivative below is derived explicitly here** in the [primary publication](https://doi.org/10.1007/s42484-023-00103-6). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

Native bounded-angle product encoder followed by trainable Y rotations. Strategy uses encoding_scale (default 0.1); baseline uses scale 1 with the same features and initialization. Encoding controls data gates, whereas identity initialization controls trainable circuit parameters.

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run near_identity_encoding --framework qiskit --steps 3
python -m qml_tricks run near_identity_encoding --framework pennylane --steps 3
python -m qml_tricks run near_identity_encoding --framework cirq --steps 3
python -m qml_tricks sweep near_identity_encoding --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/near_identity_encoding/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/near_identity_encoding/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/near_identity_encoding/cirq_impl.py). These downloads use the installed shared package.

**Control:** Use encoding scale 1 instead of encoding_scale, keeping the bounded features and trainable reference identical.

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

- Primary publication: [10.1007/s42484-023-00103-6](https://doi.org/10.1007/s42484-023-00103-6).
- Result locator: Data-encoding trainability analysis; the product-encoder derivative below is derived explicitly here.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
