# Give an agent a problem, obtain a tested portfolio

The practical unit is a project contract, not a quantum circuit name. State the
target, available information at inference, data modality and independent sampling
unit, then choose candidate representations and evaluation together. The goal is
a useful measured outcome under a budget. A fit/predict integration check, a
held-out gain and a published advantage theorem are different achievements.

## The runnable path

Run these after installing the root project with
`python -m pip install -e ".[qiskit]"`. The utility and research library are
in the same package distribution. See [start here](START_HERE.md) for
the installed `qml-utility` CLI and a one-command audited example.

```bash
python -m QML_utility.make_example --output QML_utility/results/local/wine_project
python -m QML_utility.recommend --profile QML_utility/results/local/wine_project/project.json --output QML_utility/results/local/wine_portfolio.json
python -m QML_utility.run_project --config QML_utility/results/local/wine_project/project.json --output QML_utility/results/local/wine_run
python -m QML_utility.audit_project --run QML_utility/results/local/wine_run --config QML_utility/results/local/wine_project/project.json
```

The example uses a fixed 72-record, three-class wine subset and all 13 input
features. It tests integration; it is not a published application replication
or a proposed favorable benchmark. `--task regression` generates a corresponding
diabetes integration example. Dataset provenance and fixed original row IDs are
saved. To compare SDKs, generate separate projects using `--framework pennylane`
or `--framework cirq`. Re-running requires a fresh output directory.

For your data, edit the generated JSON. CSV paths are relative to the config
file's directory. List the numeric features explicitly; target, ID, group and
time columns are excluded from that list. Blank feature cells are imputed from
the training median; missing targets and infinite values are rejected. No data
rows or feature coordinates are silently dropped. Classification supports string
labels and more than two classes. Regression supports one continuous target.

The dense/trainable reference limits are 20–512 total records, 1–64 numeric
features, 1–6 qubits and 1–4 layers. The explicit [batched adapter](SCALING.md)
supports fixed features up to a configurable 20,000 records. Exceeding a limit requires an explicit design; it does
not trigger automatic PCA. A small register still requires depth proportional to
the uploaded feature count. These exact simulators submit no QPU jobs.

## Choose the split to match use

| Situation | JSON `data.split` | What the adapter guarantees | What still needs project work |
|---|---|---|---|
| Independent rows | `{"kind":"iid"}` | Seeded 60/20/20 split, classification stratification | Justify independence, duplicate/near-duplicate audit |
| Repeated entities, patients, molecules or devices | `{"kind":"group","group_column":"entity"}` | Three disjoint group sets | Choose the correct independent unit; group sizes can change row fractions |
| Strictly ordered observations | `{"kind":"chronological","time_column":"date","gap_rows":3}` | Past training, later validation, final test, row embargo | Set lag/target-horizon embargo, causal feature availability, revisions and rolling refits |

Use ISO timestamps with unique times in the chronological adapter. Panel data,
repeated timestamps, spatial dependence and overlapping label windows require a
specialized splitter. This temporal split is an initial transfer evaluation,
not the volatility paper's rolling monthly re-estimation. Classification requires
training examples of every evaluation class; do not break groups or chronology
to satisfy that requirement.

## What gets compared

Five classical controls always run: dummy, regularized linear/logistic, RBF
SVM/SVR, boosting and a regularized sinusoidal-feature head. Four QML families
are available: fixed local quantum features, projected-feature RBF, fidelity
kernel and trainable local re-uploading. See [the recipe index](recipes/README.md).

Each non-dummy family receives three prespecified candidates. The default QML
sweep varies encoding scale, while classical sweeps vary their stated settings.
This is an initial comparison, not exhaustive tuning or equal wall-clock cost.
Add domain-specific controls and stronger validation search for a confirmatory
study. The Fourier control tests whether inexpensive classical periodic features
already account for a gain. Classical full-feature controls receive all raw
features through their own train-only imputation/scaling.

Every candidate fits only training data. Validation chooses one candidate per
family and the overall validation champion. `selection.json` freezes those
choices before test predictions. Test scores never select the deployment
champion. No train+validation refit is performed. Do not change the config after
reading the test report and call the next run a fresh confirmatory evaluation.

## Inspect and reuse the result

`REPORT.md` summarizes the comparison. `report.json` records all candidates,
warnings, failures, validation predictions, selected test predictions/targets,
split IDs, training traces, representation diagnostics, versions, source/data
hashes and resource counters. A budget-limited candidate remains a failure in a
partial report. All-failed families remain visible in the trial history.

Each run retains the exact Python source under `source/QML_utility`, so later
implementation changes do not erase the code used for that run. The recorded
environment versions and raw-data hash remain necessary for reconstruction.

The independent audit recomputes validation/test metrics using NumPy definitions
and original CSV targets, checks validation-selected candidates/champion, and
verifies dataset, source-snapshot and model-artifact hashes without deserializing models. It
checks report consistency, not scientific generalization or absence of all leakage.

```python
import joblib
import pandas as pd  # optional; not required by the utility

artifact = joblib.load("QML_utility/results/local/wine_run/quantum_features.joblib")
frame = pd.read_csv("new_observations.csv")
predictions = artifact["model"].predict(frame[artifact["features"]].to_numpy())
```

Joblib artifacts are Python objects: load only artifacts you trust. Keep the
recorded environment and feature order. A QML predictor still executes its
native simulator at prediction; the fitted classifier is not a zero-query
surrogate. Per-trial state-preparation caps include fitting, validation and test;
the runtime cap is cooperative per fit/predict operation. They are reference
budgets, not OS-level timeouts. Deployment may require a separately declared
budget and a hardware or surrogate adapter.

Use [the offline student](OFFLINE_DISTILLATION.md) when classical-only inference
is required. It has its own held-out feature/prediction gate and serialization
checks. Use [multi-run evaluation](EVALUATION.md) to assess sensitivity to IID/
group splits or temporal windows; its reports preserve pairing and dependence.

Classification currently returns labels. Probability calibration, probability
metrics, uncertainty-aware decisions and threshold selection require separate
training/validation adapters; the runner does not claim calibrated probabilities.

## When no paper matches

The router accepts optional constraints to narrow the exploratory portfolio:

```json
{
  "constraints": {"n_samples": 100},
  "preferences": {"train_circuit": false, "classical_only_inference": true},
  "budget": {"max_gram_bytes": 1000, "state_preparations_per_trial": 1000}
}
```

Here it withholds circuit training and conservative dense-kernel plans, retains
the fixed-feature candidate, and flags the offline-distillation requirement.
These keys supplement the full project contract. Estimates assume 60% training;
group sizes and adapters can change costs. Withheld plans are not declarations
that an entire method cannot help. Set the runnable config's `methods` to the
chosen candidate IDs; `run_project` executes that explicit list rather than
silently rewriting your configuration. Classical-only deployment still needs
the separately verified surrogate adapter; teacher fitting alone does not satisfy it.

`recommend` can propose exploratory methods without inventing a published win.
The old strict `select_case` still answers only the publication-matching question.
That preserves the evidence boundary while making the toolkit useful for a new
case. A numeric profile receives multiple candidate hypotheses, not a prediction
that any must win. Graphs, text, continual learning and quantum acquisition route
to their evidence/protocols and require an explicit adapter; they are not forced
into a tabular estimator. Use [the extension contract](EXTENDING.md).

## Deciding whether the result is useful

Before a larger experiment, define practical thresholds for predictive gain,
uncertainty, latency, memory, measurement cost and deployment constraints. This
runner supplies one exploratory holdout, not a significance test or a confidence
interval. Repeat with independent units or a prespecified external evaluation;
account for multiple model comparisons. Include every run and uncertainty on
paired differences. If a gain vanishes under a better classical control, revise
the recommendation's scope. If no gain is observed, retain the configuration
and test a specific new hypothesis using [contextual results](CONTEXTUAL_RESULTS.md).
