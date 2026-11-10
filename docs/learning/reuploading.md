# Angle encoding and data re-uploading

## When to use it

Use bounded continuous features and a small register. Repeated uploads trade depth for expressivity. The native example uses one trainable Y rotation per occurrence, with angle and phase data gates.

## Mathematical and evidence contract

Expressivity results for the published constructions; no universal convergence or advantage claim for the small executable.

Primary source: [Angle encoding and data re-uploading](https://doi.org/10.22331/q-2020-02-06-226). The source supports the stated mechanism;
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
python -m qml_tricks learning-demo reuploading --framework qiskit
python -m qml_tricks learning-demo reuploading --framework pennylane
python -m qml_tricks learning-demo reuploading --framework cirq
```

The native implementations use [Qiskit](../../src/qml_tricks/learning/recipes/reuploading/qiskit_impl.py),
[PennyLane](../../src/qml_tricks/learning/recipes/reuploading/pennylane_impl.py), and
[Cirq](../../src/qml_tricks/learning/recipes/reuploading/cirq_impl.py). Each imports its
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
