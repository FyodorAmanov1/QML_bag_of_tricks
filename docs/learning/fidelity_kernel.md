# Fidelity quantum kernels

## When to use it

Use moderate sample counts: exact training uses quadratic pair evaluations. The finite-shot Gram matrix can be indefinite; training-only eigenspace features give a consistent PSD out-of-sample extension.

## Mathematical and evidence contract

The ideal kernel |<psi(x)|psi(y)>|^2 is PSD; the regularized downstream optimization is convex. No general trainability or predictive superiority guarantee.

Primary source: [Fidelity quantum kernels](https://doi.org/10.1038/s41586-019-0980-2). The source supports the stated mechanism;
the executable's restricted scope is specified above. A formal identity, convex
readout, or PSD approximation is not a universal barren-plateau or advantage proof.

## Implementation contract for an agent

1. Read the data-access assumptions and source result before transferring the recipe.
2. Install `.[learning]` plus the selected SDK extra, or `.[all,learning]`.
3. Run the native example and inspect estimator error, resource counts and scope.
4. Fit preprocessing on training data only. Reuse the same splits for controls.
5. Record the quantum acquisition cost separately from classical fitting and inference.
6. Validate the mathematical property on a small independently calculable case.

```bash
python -m qml_tricks learning-demo fidelity_kernel --framework qiskit
python -m qml_tricks learning-demo fidelity_kernel --framework pennylane
python -m qml_tricks learning-demo fidelity_kernel --framework cirq
```

The native implementations use [Qiskit](../../src/qml_tricks/learning/recipes/fidelity_kernel/qiskit_impl.py),
[PennyLane](../../src/qml_tricks/learning/recipes/fidelity_kernel/pennylane_impl.py), and
[Cirq](../../src/qml_tricks/learning/recipes/fidelity_kernel/cirq_impl.py). Each imports its
own native preparation backend and shares only the estimator/learning mathematics.
Finite-shot experiments are host sampling of native SDK simulator distributions.
Simulator full-state access and density embeddings are not hardware shortcuts.

## Benchmarks and failure modes

See [the project workflow](index.md) for complete models, CSV input, repeated seeds,
classical controls, serialization and resource reports. Compare predictive error
and cost together. Kernel concentration, insufficient samples, feature collapse,
invalid parameter-shift generators, singular landmarks, regularization sensitivity,
data leakage and distribution shift can invalidate an apparent success.

Each method needs its own assumptions after composition. The small native
examples are tests of mechanism and integration, not empirical evidence for an
asymptotic computational advantage.
