# Short-time constrained Hamiltonian ansatz

**Family:** Initialization · **Result scope:** specified short-time region · **Code:** restricted demonstration; inspect guide before transferring the published guarantee

## When to use it

Physics-informed QML with a known local generator and reference state. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

Under the threshold and nonzero-commutator condition, $|\partial_{n,m}C|\geq g/4$. Bounding evolution time controls how far the gradient can move from its reference value. Zero reference commutator invalidates this argument.

**Required assumptions:** Local generators and observable; reference commutator $g=|\mathrm{Tr}(\rho_0[H^{(m)},O])|=\Theta(1)$; nonnegative total evolution time below the theorem threshold $\tau_0=\Theta(1/n)$.

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## Logic and proof route

Read **Theorem 1, equation (30); short-time commutator analysis** in the [primary publication](https://doi.org/10.22331/q-2024-02-01-1239). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

Starts qubit zero in $|+\rangle$; Y rotations and ZZ couplings measure $Z_0$. Bounds every stored angle by $0.01/(nP)$ and projects updates. The constant 0.01 is a conservative demonstration choice, not a certified evaluation of the paper's threshold.

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run constrained_hva --framework qiskit --steps 3
python -m qml_tricks run constrained_hva --framework pennylane --steps 3
python -m qml_tricks run constrained_hva --framework cirq --steps 3
python -m qml_tricks sweep constrained_hva --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/constrained_hva/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/constrained_hva/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/constrained_hva/cirq_impl.py). These downloads use the installed shared package.

**Control:** Same circuit and task, uniformly initialized parameters.

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

- Primary publication: [10.22331/q-2024-02-01-1239](https://doi.org/10.22331/q-2024-02-01-1239).
- Result locator: Theorem 1, equation (30); short-time commutator analysis.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
