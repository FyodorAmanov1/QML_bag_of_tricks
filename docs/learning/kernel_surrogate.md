# Density-embedding fidelity-kernel surrogate (not Kipu DQFE)

## When to use it

Collect quantum teacher embeddings on TRAINING data, fit a classical RBF-ridge student, and evaluate new kernels without quantum calls. Here teacher density matrices use small exact simulations, an exponential acquisition cost. Held-out kernel error must be audited separately. Random Fourier approximation of an ordinary RBF kernel alone is not approximation of an arbitrary quantum kernel.

## Mathematical and evidence contract

The learned embedding inner product is PSD by construction. If density-embedding errors are uniformly <= epsilon and target norms are one, kernel errors are <= 2 epsilon + epsilon^2. The error premise is not guaranteed by finite training data. This concrete small-state student is not a reproduction of the cited shadow-model theorem.

Primary source: [Density-embedding fidelity-kernel surrogate (not Kipu DQFE)](https://www.nature.com/articles/s41467-024-49877-8). The source supports the stated mechanism;
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
python -m qml_tricks learning-demo kernel_surrogate --framework qiskit
python -m qml_tricks learning-demo kernel_surrogate --framework pennylane
python -m qml_tricks learning-demo kernel_surrogate --framework cirq
```

The native implementations use [Qiskit](../../src/qml_tricks/learning/recipes/kernel_surrogate/qiskit_impl.py),
[PennyLane](../../src/qml_tricks/learning/recipes/kernel_surrogate/pennylane_impl.py), and
[Cirq](../../src/qml_tricks/learning/recipes/kernel_surrogate/cirq_impl.py). Each imports its
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
