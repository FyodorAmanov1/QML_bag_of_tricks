# Local-moment MMD objectives

**Family:** Objectives · **Result scope:** specified loss and shallow architecture · **Code:** restricted demonstration; inspect guide before transferring the published guarantee

## When to use it

Sample-based generative learning of selected local statistics. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

For $k(x,y)=n^{-1}\sum_i z_i(x)z_i(y)$, $\mathrm{MMD}^2=n^{-1}\sum_i(\mathbb E_p z_i-\mathbb E_q z_i)^2$. Local statistics avoid multiplying many independent overlaps. This degree-one kernel cannot distinguish distributions with identical one-bit marginals.

**Required assumptions:** An implicit sample loss; significant low-body kernel contributions; shallow compatible circuit and nonvanishing data-dependent factors. Full-distribution identification needs a richer kernel.

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## Logic and proof route

Read **Implicit-loss analysis and kernel bodyness discussion** in the [primary publication](https://doi.org/10.1038/s41534-024-00902-0). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

Fits target moments 0.5, corresponding to independent bits with P(1)=0.25. Uses a plug-in finite-shot squared loss, whose finite-sample bias is disclosed. No asymptotic lower bound is asserted for this arbitrary dataset.

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run mmd_loss --framework qiskit --steps 3
python -m qml_tricks run mmd_loss --framework pennylane --steps 3
python -m qml_tricks run mmd_loss --framework cirq --steps 3
python -m qml_tricks sweep mmd_loss --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/mmd_loss/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/mmd_loss/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/mmd_loss/cirq_impl.py). These downloads use the installed shared package.

**Control:** Use the global delta kernel, whose MMD squared equals the full probability-vector squared distance to the same independent-bit target. It distinguishes more distributions but can concentrate.

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

- Primary publication: [10.1038/s41534-024-00902-0](https://doi.org/10.1038/s41534-024-00902-0).
- Result locator: Implicit-loss analysis and kernel bodyness discussion.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
