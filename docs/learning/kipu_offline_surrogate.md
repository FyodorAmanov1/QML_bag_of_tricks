# Kipu-style offline quantum-feature surrogates

## When to use it

See [the full teacher-first workflow](kipu_workflow.md). Use an already-fitted quantum feature teacher or import real Kipu input/output feature pairs. Freeze the teacher before distillation, preserve the downstream predictor, and export a teacher-free classical artifact only after validation. This entry explicitly differs from density-embedding fidelity-kernel approximation and Nyström.

## Mathematical and evidence contract

Public vendor workflow: acquire quantum feature pairs, fit a classical surrogate, and deploy with a classical predictor. Ridge is the student family reported in Kipu's example. The native demos here use small reference teachers, not Kipu's DQFE circuits, proprietary SDK, datasets or claimed accuracy figures. Held-out fidelity is an empirical acceptance gate, not a universal theorem.

Primary source: [Kipu-style offline quantum-feature surrogates](https://kipu-quantum.com/blog/classical-surrogates-for-quantum-feature-extraction). The source supports the stated mechanism;
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
python -m qml_tricks learning-demo kipu_offline_surrogate --framework qiskit
python -m qml_tricks learning-demo kipu_offline_surrogate --framework pennylane
python -m qml_tricks learning-demo kipu_offline_surrogate --framework cirq
```

The native implementations use [Qiskit](../../src/qml_tricks/learning/recipes/kipu_offline_surrogate/qiskit_impl.py),
[PennyLane](../../src/qml_tricks/learning/recipes/kipu_offline_surrogate/pennylane_impl.py), and
[Cirq](../../src/qml_tricks/learning/recipes/kipu_offline_surrogate/cirq_impl.py). Each imports its
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
