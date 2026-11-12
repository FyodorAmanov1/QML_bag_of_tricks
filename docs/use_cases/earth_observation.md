# Satellite-image classification with a hybrid quantum image model

**Evidence: peer-reviewed reported result; not locally reproduced.**
This repository supplies an implementation and benchmark plan. The complete
MLTQNN training pipeline has not been ported into our three SDK backends.

## What motivates trying it?

[Fan et al., Quantum Machine Intelligence (2026)](https://link.springer.com/article/10.1007/s42484-026-00415-3)
report **95.8% SAT-6 test accuracy**, compared with **93.5% for MLTCNN**, its
classical multitask counterpart, and **94.9% for DenseNet**, the best classical
score in that setting. Table 4 reports respective variations of ±0.2, ±0.7 and
±1.3 percentage points across three training runs. These are paper results,
not confidence intervals computed by this repository.

The experiment used a balanced 4,200/1,200/1,200 image split, 12 qubits and a
noiseless TensorFlow Quantum simulator. These are reduced benchmark settings;
the large CNN comparisons used lightweight training conditions. The model
combines an auxiliary reconstruction task, quantum convolutions and a location
weight module. See Sections 3–4 and [Table 4 in the primary PDF](https://link.springer.com/content/pdf/10.1007/s42484-026-00415-3.pdf).

[Author code](https://github.com/zhu-xlab/MLTQNN) is public. Availability has been
checked; training, exact sampled indices and dependency compatibility have not
been verified locally.

## When to use this lead

Consider a bounded experiment when labeled imagery is costly, you already have
a compact classical representation, and you can afford simulator training for
a small head. A concrete project would classify a few land-cover categories
from a frozen satellite-image collection, then test whether the head still
helps on another region. Choose the target region and label budget first.

A standard pretrained vision model remains a necessary project comparator.
Treat a change in geographic distribution as a separate evaluation; random
image splits can put neighboring or overlapping patches in both train and test.

## Agent implementation contract

The following is our proposed adaptation protocol. A faithful paper replication
must first recover its precise encoding, patch layout, losses and sampled IDs.

1. **Freeze data access.** Save image IDs, classes, channels, source scenes and
   geographic groups. Create train/validation/test partitions before fitting
   normalization, compression or reconstruction. Keep near-duplicate patches
   in the same partition. If source metadata are unavailable, record that gap.
2. **Build the classical path first.** Fit a small encoder and reconstruction
   branch using training data only. Store a checkpoint and encoder outputs.
   Specify a supervised-plus-reconstruction objective, for example
   \(L=L_{\mathrm{class}}+\alpha L_{\mathrm{recon}}\), and select \(\alpha\)
   with validation data. This equation defines an adaptation, not an assertion
   of identical paper normalization.
3. **Change one component.** Feed the same encoded representation to a classical
   image head and a quantum head. Give both equal access to labels, auxiliary
   data and tuning. Also include a tuned pretrained model with its actual
   parameter count and compute. Parameter matching alone is insufficient.
4. **Port deliberately.** Read the paper's circuit and the upstream encoder
   before implementing a port. Construct Qiskit, PennyLane and Cirq circuits
   independently, then compare small-fixture probabilities, bit ordering,
   rotation signs, observables and gradients. A generic quantum convolution
   demo is an adaptation until those checks and the full protocol match.
5. **Evaluate a frozen choice.** Select hyperparameters and checkpoints using
   validation data only. Save held-out probabilities for every planned seed.
   Report accuracy, macro-F1, per-class recall and calibration. Add geographic
   holdout and low-label learning curves as separate, clearly named experiments.

## Trainability and cost checks

Use the [classical pretraining companion](../learning/classical_pretraining.md)
to establish the compression path, and [local objectives](../strategies/local_loss.md)
or [near-identity initialization](../strategies/near_identity_encoding.md) when
their assumptions fit a new head. The [QCNN guide](../strategies/qcnn.md) explains
a related architectural idea; its theorem does not establish trainability for
this image model. Inspect gradient norms and shot uncertainty before scaling.

Record encoder training, data-loading gates, head depth, active qubits,
measurement settings, circuits per image, gradient evaluations and shots.
Distinguish exact expectations, finite-shot simulation and noisy execution.
Statevector memory grows as \(O(2^q)\); batching and autodifferentiation add
storage. A small quantum register does not make image loading or training free.

## Completion criteria for an agent

Deliver a split manifest, environment lock, both heads, selected checkpoints,
per-example predictions and a resource report. Include the shared-encoder
ablation and the pretrained classical control. Compute paired uncertainty
without treating correlated patches as independent observations. Keep all
planned seeds. A gain is accepted only for the frozen endpoint and comparator;
a simulator accuracy gain does not establish a hardware resource advantage.

Suggested task to give an agent:

> Implement the satellite-image adaptation above. Start from a frozen dataset
> manifest and a classical encoder. Benchmark a small independently constructed
> quantum head against the same encoder with a classical head and a tuned
> pretrained vision model. Save predictions and resource counts for every seed.
> Label all deviations from Fan et al. and report the result even if QML loses.
