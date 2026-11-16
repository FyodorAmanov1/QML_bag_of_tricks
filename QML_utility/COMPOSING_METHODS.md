# Choose a companion because a diagnostic calls for it

The original repository is the mechanism library for the practical workflow.
[companion_map.json](companion_map.json) makes the handoff machine-readable.
Composition must preserve each method's assumptions; a list of useful ingredients
is not a proof that the combined classifier avoids plateaus or generalizes.

| Observation in the new project | Candidate response | Evidence to collect |
|---|---|---|
| Features collapse as encoding scale grows | [Weak encoding](../docs/strategies/near_identity_encoding.md) and scale validation | Feature variance/rank, quantum and classical same-angle controls, preserved signal on held-out data |
| Loss has weak gradients at initialization | [Initialization](../docs/strategies/identity_block_init.md), local objective and shallower causal cones | Coordinate gradients, seed sensitivity, relevant state/observable factors; changed objective still solves the task |
| Gradients exist but updates are ill conditioned | [Quantum natural gradient](../docs/strategies/natural_gradient.md) | Metric spectrum/regularization, equal-data comparison to Euclidean updates, metric-query and solve costs |
| A global loss seems unsuitable | [Local loss](../docs/strategies/local_loss.md) | Confirm target-equivalence in this task; never transfer a local initialization bound to arbitrary cross-entropy |
| The raw representation is unsuitable for a small circuit | [Classical pretraining](../docs/learning/classical_pretraining.md) | Train-only encoder, unchanged test partition, tuned classical head on the exact same learned features |
| Optimization is expensive but a small simulator model can help | [Simulator warm start](../docs/learning/classical_warm_start.md) | Pretraining time, circuit transfer mismatch, random-initialization control |
| Many gate coordinates need noisy derivatives | [SPSA](../docs/learning/spsa.md) or [parameter shift](../docs/learning/parameter_shift.md) | Estimator bias/variance, actual shots and total loss/expectation calls |
| Measurement dominates | [Shot allocation](../docs/learning/shot_allocation.md) or [shadows](../docs/learning/classical_shadows.md) | Pilot costs, locality, variance/confidence and reuse assumptions |
| Kernel acquisition is too expensive | [Nyström](../docs/learning/nystrom.md) | Landmark coverage, conditioning, approximation and new-input quantum queries |
| A trained teacher helps, but inference must be classical | [Kipu-style offline feature distillation](../docs/learning/kipu_workflow.md) | Held-out feature fidelity and task discrepancy; frozen teacher; zero quantum queries at student inference |

The new estimator includes weak-angle candidates, small trainable rotations and
local XYZ measurements as engineering choices. It does not implement the source
papers' plateau-avoidance constructions or a QNG optimizer. The main-repo native
companions remain available for controlled extensions. Kipu's public workflow is
a vendor-described engineering companion, not another peer-reviewed success case;
it differs from Nyström and from density-embedding kernel approximation.

Only combine methods after declaring the hypothesis and the changed assumption.
Change one axis in an ablation when possible, preserve the original run, tune on
validation and use fresh data for confirmation if test results motivated redesign.
