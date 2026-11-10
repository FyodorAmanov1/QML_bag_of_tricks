# Classical pretraining and quantum-kernel surrogates

Neither had an executable implementation in the original collection. Coverage
notes mentioned transfer learning and surrogates without project pipelines.

## Preprocessing, pretraining and warm starts

- **Preprocessing:** training-only scaling and PCA; not supervised pretraining.
- **Classical representation pretraining:** fit an MLP on training labels, freeze
  its hidden representation, and attach a quantum-feature readout. Compare a
  classical ridge head on exactly the same hidden features.
- **Classical simulator warm start:** optimize actual circuit angles with ideal
  simulation, then transfer them to the sampled phase. Separate simulator calls
  and exponential memory from subsequent measurement costs.

These two pretraining workflows are engineering companions explicitly requested
by the user. The ridge readout has a conditional convexity guarantee; there is
no general global-optimum or plateau guarantee for MLP pretraining or warm starts.
The [hybrid transfer-learning paper](https://doi.org/10.22331/q-2020-10-09-340)
motivates the approach; this small MLP example does not reproduce all its models.

## Two different kernel approximations

**Correction:** the specifically requested Kipu method is
[teacher-first offline quantum-feature distillation](kipu_workflow.md).
The density-embedding reference below is a different approximation and must not
be described as Kipu's workflow.

**Nyström:** training landmarks define a PSD low-rank approximation. New inputs
still need quantum-encoder/kernel evaluations. Accuracy depends on spectrum,
landmarks, truncation and shot noise. It is not fully classical deployment.

**Classical student:** obtain quantum density embeddings phi(x) on training inputs
using a small exact simulator. Their real inner product is the fidelity kernel.
Train a classical multioutput RBF-ridge student phi_hat(x), and define the surrogate
kernel by phi_hat(x) dot phi_hat(y). Prediction requires zero quantum-backend calls.
Tests check this, and validation inputs receive a separate teacher-error audit.

The surrogate kernel is PSD by construction. If true embeddings have unit norm
and embedding errors at both inputs are at most epsilon, expanding the inner
product gives kernel error at most 2 epsilon + epsilon squared. This bound is
conditional; a finite dataset does not certify uniform accuracy on unseen inputs.

Density teacher acquisition is exponential and is labeled simulator-only. The
[shadow-model source](https://doi.org/10.1038/s41467-024-49877-8) supports other
classically deployed quantum models under specific assumptions; this student
is not asserted to reproduce that theorem or its advantage.

Random Fourier features for an ordinary RBF kernel are not automatically
approximations to an arbitrary quantum kernel. Surrogate validity needs an
explicit quantum teacher and held-out approximation error.
