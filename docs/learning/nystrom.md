# Nyström approximation of quantum kernels

## When to use it

Use a smaller landmark set to reduce kernel evaluations. This still needs quantum features or kernel queries for new inputs; it is not fully classical inference.

## Mathematical and evidence contract

For PSD kernels, landmark features produce a PSD low-rank approximation; using all landmarks with positive eigenspaces recovers the ideal Gram matrix up to tolerance. Accuracy with few landmarks depends on spectrum and selection.

Primary source: [Nyström approximation of quantum kernels](https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf). The source supports the stated mechanism;
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
python -m qml_tricks learning-demo nystrom --framework qiskit
python -m qml_tricks learning-demo nystrom --framework pennylane
python -m qml_tricks learning-demo nystrom --framework cirq
```

The native implementations use [Qiskit](../../src/qml_tricks/learning/recipes/nystrom/qiskit_impl.py),
[PennyLane](../../src/qml_tricks/learning/recipes/nystrom/pennylane_impl.py), and
[Cirq](../../src/qml_tricks/learning/recipes/nystrom/cirq_impl.py). Each imports its
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
