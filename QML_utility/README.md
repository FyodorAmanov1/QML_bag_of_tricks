# QML utility: from a use case to a tested model portfolio

**A practical project layer backed by a preserved QML research library.**

Created and maintained by **Fyodor Amanov**, with project contributors.
Part of the same Apache-2.0 project as `qml_tricks`.
See [licensing and citation](LICENSING_AND_CITATION.md) for attribution,
a copyable citation, original-paper credit and the preserved MIT history.

Give an agent the task, data access, deployment constraints and budget. It can
match published cases, propose several justified model candidates, train a
comparison, inspect diagnostics and return a reusable predictor with its limits.
The runnable reference currently covers numeric classification and regression;
specialized modalities have explicit replication/adapter routes. This remains a
research workbench with separate implementation and predictive-evidence gates.

## Try a complete comparison

From a clone, install the utility with one native SDK, then run a complete example:

```bash
python -m pip install -e ".[qiskit]"
qml-utility doctor
qml-utility setup --output QML_utility/results/local/first_setup --check-essentials
```

The [unified setup](UNIFIED_SETUP.md) brings published case protocols and the
main library's QNG, local costs, identity-block initialization and weak encoding
into one workflow. The generated project config supports supervised QNG directly.
For an immediate fit/predict example, use `qml-utility quickstart`.
That creates a fixed real-observation integration example with all input
coordinates, then compares fixed features and a fidelity kernel against five
classical controls. Add `--preset full` for all four QML families, or
`--preset scalable` for batched fixed features and linear-memory controls.
It saves every trial, split, warning, failed fit, selected held-out prediction,
diagnostic, resource record and fitted model. The 72-record wine example tests
integration; it is not an advantage benchmark. Use `--task regression` with
quickstart for the regression path. `python -m QML_utility ...` is an equivalent
entry point. Choose fresh output directories per run.

| What you need | Entry point |
|---|---|
| First install and complete example | [Start here](START_HERE.md) |
| Give a coding agent your specific project | [Copyable project brief](PROJECT_AGENT_BRIEF.md) |
| Implement and compare on your own numeric CSV | [Practical workflow](PRACTICAL_WORKFLOW.md) |
| Repeated holdouts or chronological backtests | [Evaluation](EVALUATION.md) |
| More than 512 records without a dense quantum kernel | [Batched feature adapter](SCALING.md) |
| Deploy a classical mimic of a frozen quantum feature map | [Offline distillation](OFFLINE_DISTILLATION.md) |
| Choose among runnable QML families | [Recipe index](recipes/README.md) and [method library](method_library.json) |
| Match data/access to a published successful comparison | [Task-to-case map](CASE_SELECTION.md) |
| Adapt the implementation to a new modality | [Extension contract](EXTENDING.md) |
| Diagnose and improve training/encoding/measurement/deployment | [Composition guide](COMPOSING_METHODS.md) and [companion map](companion_map.json) |
| Understand what is verified and what remains | [Validation](VALIDATION.md), [source audit](SOURCE_AUDIT.md) and [repository design](REPOSITORY_DESIGN.md) |

Executed coverage: [three-SDK implementation checks](implementation_coverage.json)
and [real-observation integration runs](results/practical_smoke.md). These record
software verification separately from any application-performance claim.

### Runnable model families

| Candidate | Trained component | Useful comparison question |
|---|---|---|
| Fixed quantum features | Regularized linear/logistic readout | Do fixed local quantum observables add useful representation? |
| Projected kernel | Classical RBF head on local features | Does a nonlinear readout improve on the same measured features? |
| Fidelity kernel | SVC/kernel ridge on state overlaps | Is overlap similarity better aligned with this target? |
| Trainable local re-uploading | Circuit angles and head jointly | Does adapting the circuit justify the gradient cost? |

Each supports independent Qiskit, PennyLane and Cirq construction. The runner
supports binary/multiclass classification, scalar regression and IID/group/time
holdouts. It uses train-only preprocessing, no automatic PCA, validation-only
selection, source/data hashes and explicit simulator costs. These are tested
engineering adaptations of established methods, separate from the ten published
success cases below. Generality comes from a controlled adaptation/evaluation
interface, not a promise that every case has a known quantum winner.

## Keep the research depth

The original trainability strategies, QNG, encoding design, pretraining,
surrogates, measurement tools and reproduction records remain available.
The [companion map](companion_map.json) tells an agent which diagnostic calls for
each method and what assumptions/costs to check. All new utility code, docs,
tests and artifacts stay here. Strategy execution and geometry reuse the
canonical `qml_tricks` implementation in the same distribution.

## Interpret the evidence correctly

The original `docs/study.md` tests particular two-qubit/PCA pipelines. Its poor
results describe the tested datasets, preprocessing, configurations, comparisons
and training budgets. They do not rule out better training, different encodings
or differently structured data. Those possibilities are hypotheses to evaluate,
rather than reasons to assume a gain. This prototype starts with a published
success, identifies its task/access conditions, and checks whether a new project
has a reason to replicate it. See [the scoped outcome ledger](CONTEXTUAL_RESULTS.md).

## What has credible success evidence?

| Task that actually matches | Method | What is supported | Where to start |
|---|---|---|---|
| Long sequences of changing image tasks | Compact-unitary QNN; amplitude encoding; local readout | Reported preservation of new-task learning versus tested MLPs, including L2 controls. Not retention of every past task. | [Continual learning](evidence/continual_plasticity.md) |
| Recovering a common graph-generating function from unaligned graphs | Re-uploading QGNR and Gromov-Wasserstein loss | Reported reconstruction/parameter benefits against named graphon baselines. | [Graphons](evidence/quantum_graphons.md) |
| Small-subsample molecular activity regression from selected descriptors | Fixed quantum reservoir and classical readout | Published robustness as sample size decreases; final SI uses 100/200/800-record subsamples and 18 descriptors. | [Molecular reservoirs](evidence/molecular_reservoir.md) |
| Short balanced sentiment sentences | Projected quantum query/key/value maps and Gaussian attention | Modest reported gains over simple attention controls; no comparison to modern language models. | [Text attention](evidence/quantum_self_attention.md) |
| Small optical classification tasks whose labels are designed from kernel geometry | Interfering-photon kernel and SVM | Better fit for these engineered targets; two photons remain classically simulable. | [Interference kernels](evidence/photonic_interference_kernel.md) |
| Historical monthly volatility, one-step target and rolling refits | Fixed Ising reservoir and classical readout | Lowest reported one-step MSE; the five-step comparison favors a classical reservoir. | [Forecasting](evidence/volatility_reservoir.md) |
| Quantum state copies; a magnitude query arrives after measurement | Two-copy Bell acquisition | Published learning-access separation; a restricted local audit is runnable here. | [Late-query learning](evidence/late_query_bell.md) |
| Learning an unknown multimode optical displacement process | EPR probes and joint CV measurement | Experimental sample-complexity benefit against entanglement-free access. | [Photonic processes](evidence/photonic_process.md) |
| Pauli-channel queries with retained quantum reference memory | Bell acquisition plus auxiliary-noise mitigation | Experimental sampling benefit under the paper's system/reference access model. | [Channel learning](evidence/pauli_channel_learning.md) |
| Special discrete-log-defined labels and fault-tolerant arithmetic | Quantum kernel estimation plus SVM | Conditional computational separation, not a practical tabular benchmark. | [Theory boundary](evidence/discrete_log_kernel.md) |

There is **no verified rule here saying that small, noisy, correlated, or
high-dimensional classical data alone makes QML outperform a tuned classical
model**. The classical-data entries motivate specific replications, the
quantum-data entries change acquisition, and the discrete-log result requires
special structure and fault tolerance. A published win establishes a comparison
in its tested setting, not a sufficient condition for success on new data.

Use [the task-to-case map](CASE_SELECTION.md) to choose a replication priority.
The [source audit](SOURCE_AUDIT.md) records unresolved paper/code differences
and selection-scope questions before an agent begins implementation.

## Exact paper-matching and replication plans

Run from the parent repository directory, using its Python 3.12 environment:

```bash
python -m QML_utility.select_case --profile QML_utility/examples/continual_project.json
python -m QML_utility.select_case --profile QML_utility/examples/generic_csv.json
python -m QML_utility.plan_project --profile QML_utility/examples/molecular_project.json --output QML_utility/results/local/molecular_plan.json
python -m QML_utility.plan_project --profile QML_utility/examples/wine_followup.json --output QML_utility/results/local/wine_plan.json
python -m QML_utility.benchmarks.late_query --output QML_utility/results/local/late_query.json
python -m pytest QML_utility/tests -q
```

The selector returns missing conditions and reproduction blockers, **not a promise
of superiority**. An ordinary CSV produces no evidence-matched recommendation;
that is not evidence of method-wide uselessness. The planner emits a draft
experiment contract without running training. The examples leave unknown budgets
blank so agents must supply them. Follow [the agent workflow](AGENT_WORKFLOW.md)
for source recovery, controls, selection and held-out evaluation.
The benchmark uses exact sufficient-statistic sampling for a specified state
family; it is not a QPU experiment or a replay of the Science experiment.

## What is implemented and what is pending?

- Installable utility package and one CLI for setup diagnostics, quickstart,
  portfolios, training, repeated evaluation, audits and prediction.
- Batched acquisition/SGD fixed-feature adapter with configurable caps up to
  20,000 records and explicit state-batch/feature-cache budgets.
- Frozen multi-run plans, paired comparisons, test-overlap records and
  expanding chronological windows with train/validation/test embargoes.
- Offline feature-map distillation with record-disjoint export gates and
  zero-quantum-query student prediction, distinct from Nyström and DQFE replication.
- Four fit/predict model families in three native SDKs, a portfolio router and
  a split-aware CSV comparison runner with saved predictor artifacts.
- Regularized full-feature and Fourier controls, prespecified candidate search,
  representation/gradient diagnostics and explicit failed-trial retention.
- Ten machine-readable evidence records with required data/access conditions.
- Seven paper-specific replication plans, a contextual follow-up protocol,
  and five pinned author repositories with inspected-file hashes.
- Independent Qiskit, PennyLane and Cirq references for SU(4) continual-model
  circuits, two-copy Bell acquisition and ideal signed Pauli-channel acquisition.
  These verify construction/readout, not training success or full device EMEEL.
- Tested Gaussian attention aggregation for measured query/key/value features;
  this component is not a complete trained text classifier.
- A machine-readable ledger of five bounded non-winning outcomes, with diagnostics
  and prospective follow-ups that preserve the original results.
- A standalone late-query audit with optimal specified-record inference,
  matched copy counts, multiple seeds and an early-query control that closes the gap.
- [Implementation contracts](IMPLEMENTATION.md), [source audit](SOURCE_AUDIT.md),
  [data conditions](DATA_CONDITIONS.md), and [promotion criteria](EVIDENCE_POLICY.md).
- [Executed local protocol audit](results/README.md), with compressed per-state
  records and an early-query ablation, kept separate from paper performance.

**Not yet completed:** full original-data QGNR training, long CIFAR-100 trajectories,
the published L2 ablations, molecular/text/volatility training, or photonic/channel
raw-data replay. Source-version and missing-artifact gaps are recorded. None is
relabeled as a local win.
The practical adapters are executable, while further scaling, modality adapters,
hardware execution and independent application-performance replications remain
work to do. See [the concrete readiness and priority map](REPOSITORY_DESIGN.md).

## First full paper-performance replication target

Prioritize the **Split CIFAR-100 plasticity** protocol: it directly addresses a
reported benefit over regularized classical networks and preserves all image
coordinates. Recover the original long trajectory first, then add tuned
plasticity-aware classical controls. Record both adaptation and forgetting.
See [the exact plan](protocols/continual_cifar100.json).

Public availability does not certify that replication gate. Do not market
these pending replications as local successes. Keep the original studies as reproducibility records.
