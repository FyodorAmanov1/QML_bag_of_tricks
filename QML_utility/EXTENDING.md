# Add a useful adapter, not just a model name

An adapter needs an executable contract: input modality and inference-time access;
output target; train/validation/test unit; preprocessing fit scope; supported SDKs;
fit/predict or acquisition interface; resource accounting; diagnostics; classical
controls; and a labeled verification record. Start from [the workflow](PRACTICAL_WORKFLOW.md).

## Two independent evidence questions

**Does the method have a sound implementation?** Verify probabilities, ordering,
rotations, gradients, loss chain rule, fit/predict behavior, leakage boundaries,
serialization and resource limits for the actual adapter. Cross-SDK agreement
is valuable but must be supported by an independently calculable fixture.

**Does it help the user's task?** Compare against strong same-information controls,
freeze validation selection, save all held-out predictions, quantify uncertainty
at the independent unit and include the acquisition/deployment costs. Passing
implementation checks cannot answer this performance question.

`method_library.json` records implementable transfer candidates grounded in
peer-reviewed method foundations. `catalog.json` records the ten specifically
published success tuples. Do not add a success record just because an adapter
implements a published algorithm. Update [validation](VALIDATION.md) with what
was actually executed, including the framework/task coverage matrix.

## Extension recipes

| New use case | Preserve | Add before a recommendation |
|---|---|---|
| Molecular endpoint | Molecular identity, all records and source descriptor provenance | Train-only selection, scaffold/family split, full- and same-feature GPR/SVR/boosting controls |
| Time series | Temporal information and feature release times | Causal lag construction, rolling inner validation, target-horizon embargo, chronological outer evaluation |
| Raw images or text | Original example/subject identities and original test boundaries | Frozen/train-only encoder, classical head on identical features, domain-specific pretrained controls |
| Graphs | Graph identity and permutation requirements | Graph-native loss/representation, matched GNN/graph-kernel/Fourier controls and transport solver where relevant |
| Continual tasks | Task order, persistent weights and long target trajectory | Adaptation/retention metrics, replay/reset/plasticity controls; a short IID test cannot replace this |
| Quantum states/processes | Copy/channel access and reference-memory obligations | Native measurement/calibration, matched classical access, readout/output-size and shot costs |

## Changing the numeric reference

The three `models/portable_*.py` files independently construct their native
circuits. Shared validation and local-feature mathematics are allowed; do not
replace independent construction with SDK conversion. Document any new gate
sequence, coordinate mapping or generator multiplicity before training. A
parameter reused in several gates does not have the single-occurrence two-shift
derivative without summing occurrence derivatives.

Scale beyond reference limits deliberately: minibatches and stochastic optimizers,
feature-block/locality design, low-rank kernels, measured features or offline
distillation all change costs and sometimes hypotheses. Include tensor-network
or other efficient classical simulation when the circuit admits it. Record
memory actually retained, readout precision, physical shots and any surrogate
error rather than equating a log-sized register with an end-to-end space gain.
