# Runnable method portfolio

These are tested engineering adaptations of established method families, separate
from the [published-success catalog](../catalog.json). Choose several hypotheses
and compare them on shared data splits. Their architecture is deliberately small
and explicit so agents can inspect and extend it.

| Family | What is trained | Use first when | Guide |
|---|---|---|---|
| Fixed quantum features | Regularized classical linear/logistic head | Establishing a low-optimization-cost quantum representation baseline | [Recipe](quantum_features.md) |
| Projected kernel | Classical kernel head over local features | Checking whether local readouts need a nonlinear head | [Recipe](projected_kernel.md) |
| Fidelity kernel | SVC or kernel ridge over state overlaps | Testing state-overlap similarity at manageable quadratic cost | [Recipe](fidelity_kernel.md) |
| Trainable local re-uploading | Circuit angles plus classical head jointly | Testing task-adapted features with explicit gradient diagnostics and cost | [Recipe](trainable_local.md) |

All use the [same documented circuit contract](CIRCUIT_CONTRACT.md), numerical
feature/target interfaces and independent Qiskit/PennyLane/Cirq constructions.
They accept full numeric feature matrices through train-only imputation/scaling;
they do not default to two-component PCA. A new project should also consider the
[task-specific evidence](../CASE_SELECTION.md) and [companion methods](../COMPOSING_METHODS.md).

For larger row counts use [batched acquisition and a cached SGD readout](../SCALING.md)
for the fixed-feature family. It changes the readout and execution costs, not
the native circuit or evidence label. [Offline distillation](../OFFLINE_DISTILLATION.md)
can replace a useful frozen local-feature teacher at deployment after independent
fidelity/utility gates. Neither is an additional published-success record.
