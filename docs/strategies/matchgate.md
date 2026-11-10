# Observable-aligned matchgate circuits

**Family:** Symmetry · **Result scope:** specified mixed group ensemble · **Code:** restricted demonstration; inspect guide before transferring the published guarantee

## When to use it

Free-fermion state models and quadratic observables. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

For quadratic fermionic generators, $\dim\mathfrak{so}(2n)=n(2n-1)$. The algebraic variance formula depends on inverse algebra dimension and state/observable generalized purities. Arbitrary spin observables can inhabit exponentially large modules.

**Required assumptions:** Nearest-neighbor fermionic Gaussian generators; observable in a polynomial-size algebra/module; non-exponentially-small input projection; the group-ensemble assumptions of the variance theorem.

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## Logic and proof route

Read **Lie-algebra variance theorem; applications to free-fermionic algebras** in the [primary publication](https://doi.org/10.1038/s41467-024-49909-3). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

RZ and nearest-neighbor XX gates with local Z readout. Small-depth uniform angles demonstrate the architecture rather than sampling Haar over the complete Gaussian group.

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run matchgate --framework qiskit --steps 3
python -m qml_tricks run matchgate --framework pennylane --steps 3
python -m qml_tricks run matchgate --framework cirq --steps 3
python -m qml_tricks sweep matchgate --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/matchgate/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/matchgate/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/matchgate/cirq_impl.py). These downloads use the installed shared package.

**Control:** Add Y rotations that leave the quadratic-fermion gate family. No plateau is guaranteed at the small default depth.

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

- Primary publication: [10.1038/s41467-024-49909-3](https://doi.org/10.1038/s41467-024-49909-3).
- Result locator: Lie-algebra variance theorem; applications to free-fermionic algebras.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
