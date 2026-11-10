# Identity-block initialization

**Family:** Initialization · **Result scope:** initialization only; shallow-block conditions · **Code:** analytically checked shallow-block special case; independent copies; initialization only

## When to use it

Deeper variational models with a useful reference and shallow identity blocks. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

For $W(a,b)=B(b)^\dagger B(a)$, initialize $b=a$ so $W=I$. The derivatives with respect to the independent copies generally remain nonzero. Tying $b\equiv a$ throughout optimization instead gives $W\equiv I$ and zero gradients. In this executable, the forward Y angle on boundary wire zero has $\partial C=-\cos(a_0)\cos(a_1)$ at matched initialization, hence variance $1/4$ for independent uniform forward X angles, independent of qubit count and the number of matched shallow blocks. This is a directly derived special case, not a new statement attributed to the paper. An initial-gradient claim does not guarantee the training trajectory remains outside a plateau.

**Required assumptions:** Each block has independent forward/backward parameters initialized to equal values; invert gate order and angles exactly. Preserve shallow effective causal cones and a nonzero state-observable commutator.

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## Correct independent-copy construction

The native time order of a block is RX(a), CZ, RY(c), RY(-d), reversed CZ,
RX(-b). Initialize b=a and d=c, but store and update all four arrays independently.
Every block then cancels at initialization on any input state. A perturbation
of c without the same perturbation of d is a real trainable direction; an
implementation that recomputes d=c after every update is not this algorithm.

For the boundary readout in the code, the conjugated Y generator involves only
wire zero and its neighbor. Evaluating its commutator with Z on the prepared
|+0...0> state gives the derivative stated above. Uniform independent a0,a1
give E[cos(a0)^2] E[cos(a1)^2]=1/4. Other blocks are identities at this point,
so they do not enlarge this causal cone. Coordinate zero, an X rotation on the
|+> reference, would instead have zero derivative; the diagnostic deliberately
selects coordinate n, the first forward Y rotation.

Use shallow identity blocks with a meaningful input/reference and a compatible
observable. Inserting a universal deep random unitary into each block can restore
the original effective-depth problem. Monitor gradients after updates: cancellation
is an initial condition, not a constraint that should hold for the entire run.

## Logic and proof route

Read **Identity-block construction and initial-gradient analysis** in the [primary publication](https://doi.org/10.22331/q-2019-12-09-214). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

Uses B=RY(c) CZ RX(a) per shallow block, with independently trainable inverse copies, matched only at initialization. Prepares |+> on wire zero and reads Z zero. Exact unitary cancellation is tested. A nonzero gradient is task dependent; no universal lower bound is asserted.

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run identity_block_init --framework qiskit --steps 3
python -m qml_tricks run identity_block_init --framework pennylane --steps 3
python -m qml_tricks run identity_block_init --framework cirq --steps 3
python -m qml_tricks sweep identity_block_init --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/identity_block_init/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/identity_block_init/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/identity_block_init/cirq_impl.py). These downloads use the installed shared package.

**Control:** Independently randomize forward and inverse branches instead of matching their initial values.

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

- Primary publication: [10.22331/q-2019-12-09-214](https://doi.org/10.22331/q-2019-12-09-214).
- Result locator: Identity-block construction and initial-gradient analysis.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
