# Why and when to explore QML

Start with a task where the quantum model supplies a useful representation,
temporal memory or measurement protocol. Published applications and reproduced
comparisons give concrete hypotheses to test in a project.

## Two reproduced motivations on classical inputs

| Application | What we reproduced | Starting point |
|---|---|---|
| Alloy-phase screening | Ten-qubit quantum SVC: 89.12% mean accuracy versus 87.73% for the best original classical comparator across ten replay splits | [Alloy implementation and data](reproductions/alloys.md) |
| Wireless channel equalization | Six-qubit reservoir: 0.313% mean symbol error versus 4.437% for direct memoryless logistic regression across three fresh datasets | [Temporal model and signal generator](reproductions/communications.md) |

![Application motivation](assets/qml_motivation.png)

These are bounded simulations of comparisons from [Pei et al., Matter (2024)](https://doi.org/10.1016/j.matt.2024.05.035)
and [Hu et al., Nature Communications (2024)](https://doi.org/10.1038/s41467-024-51162-7).
The alloy inputs are real materials observations; the communications task uses
simulated classical messages. Stronger tuned classical alternatives lead in
our follow-up comparisons. The [full report](reproductions/results.md) preserves
those comparisons and distinguishes archive recovery from new computation.

### Small experimental datasets: test a different kernel geometry

For materials screening or other costly experiments, a quantum fidelity kernel
offers a task-dependent similarity measure to test alongside conventional
kernels. The alloy replay demonstrates that this geometry can improve on the
specific comparators used in a published application.

Use [fidelity kernels](learning/fidelity_kernel.md) when a quadratic Gram matrix
is affordable. Tune encoding scale, representation and regularization on
training/validation data. Include tuned RBF, linear and domain-informed models.
Check group/family splits before proposing new experiments. The reproduced
gain motivates testing the model; it does not establish a general small-data
advantage or a practical quantum speedup.

### Temporal signals: combine quantum memory with a simple readout

The equalization reproduction shows how a fixed quantum reservoir can transform
streaming inputs into nonlinear features with fading memory, while a convex
classical readout does the supervised learning. This supplies a route to QML
that avoids training a large variational quantum circuit.

Use the [reset-based reservoir example](reproductions/communications.md) for
signal recovery or temporal-response experiments where memory matters. Include
classical delay features, nonlinear equalizers or classical reservoirs with
comparable access to past inputs. Check reset, readout and repeated-execution
costs before considering a hardware implementation.

## Further application hypotheses

The [peer-reviewed application catalog](use_cases/index.md) provides six more
ways to explore QML on classical inputs. Each guide separates the paper's
reported outcome from our proposed implementation protocol:

- [Satellite imagery](use_cases/earth_observation.md): test a compact quantum
  head while controlling for the classical encoder and auxiliary objective.
- [Molecular carcinogenicity](use_cases/molecular_toxicity.md): compare annealed
  weak-learner selection with classical solvers on exactly the same ensemble.
- [Hyperparameter search](use_cases/hyperparameter_search.md): test whether a
  quantum-trained curve predictor saves enough training work to cover its cost.
- [Entity matching](use_cases/entity_matching.md): route ambiguous pairs to a
  jointly trained hybrid head, comparing the same route with a classical head.
- [Multiclass kernels](use_cases/multiclass_kernel.md): select a task-dependent
  fidelity geometry and include its exact classical equivalent when available.
- [Transferred image attacks](use_cases/adversarial_qvc.md): study differences
  in learned representations using a full source-to-target attack matrix.

These are implementation leads, with source and artifact-readiness records.
The last three include [runnable reduced adaptations](use_cases/models.md).
They add domains to explore; full local paper reproductions remain future work.

## Turn the motivation into a trainable implementation

| Your next decision | Repository support |
|---|---|
| Start near a useful reference | [Near-identity encoding](strategies/near_identity_encoding.md) and [identity-block initialization](strategies/identity_block_init.md) |
| Preserve a measurable training signal | [Local costs](strategies/local_loss.md), [QCNNs](strategies/qcnn.md) and task-appropriate symmetries |
| Improve conditioning when gradients are resolvable | [Quantum natural gradient](strategies/natural_gradient.md), with metric estimation and regularization costs |
| Reuse classical learning before quantum training | [Pretraining and warm-start workflows](learning/pretraining_surrogates.md) |
| Deploy a trained quantum teacher through a classical model | [Kipu-style offline feature distillation](learning/kipu_workflow.md), with held-out fidelity gates |

Each guide states the result's assumptions and the executable's scope. A useful
initialization bound, local objective or optimizer does not guarantee a full
training or generalization outcome.

## A separate motivation when the inputs are quantum

If coherent state copies are available and the observable query arrives after
acquisition, [two-copy learning](applications/experimental_learning.md) can
retain information that the tested single-copy local records lose. In the
eight-qubit, 512-copy simulated case, Bell acquisition plus L2 logistic
regression reached 100% mean balanced accuracy versus 49.14% for optimal
inference on those local records. Both protocols use the same total copies;
Bell acquisition requires twice as many simultaneously coherent qubits. An
early-known query removes this observed gap. This is an acquisition benefit
under a specified access model, distinct from the two classical-input papers.

Start with the [executed notebooks](tutorials.md), then reproduce a relevant
application and freeze a benchmark for your own task. Compare predictive
quality, learning-data requirements, acquisition and inference cost together.
