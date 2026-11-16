# Try QML, then measure whether it helps

Use Python 3.12+ and a local clone. Install only the SDK you want:

```bash
python -m pip install -e ".[qiskit]"
qml-utility doctor
qml-utility setup --output QML_utility/results/local/first_setup --check-essentials
qml-utility quickstart --output QML_utility/results/local/first_run
```

One installation provides the utility and the main strategy library. Start with
[the unified setup](UNIFIED_SETUP.md) to choose published cases and essential
training methods, then fit a project. You can also run every command as
`python -m QML_utility ...` from the clone.
Replace `qiskit` with `pennylane`, `cirq` or `all` in the extra; pass
`--framework pennylane` or `--framework cirq` to quickstart. Nothing is submitted
to a QPU. The CLI defaults OpenMP to one thread unless you already set
`OMP_NUM_THREADS`. No dataset download is required for bundled examples.

Quickstart writes a project config, candidate portfolio, fitted models,
`run/REPORT.md`, full trial records, frozen source and an independent audit.
The default uses fixed features and a fidelity kernel alongside five classical
controls. `--preset full` includes projected kernels and trainable re-uploading.
`--task regression` selects the regression example. Use a new output directory
for each run; previous results are retained.

## Try the larger-data path

```bash
qml-utility quickstart --preset scalable --output QML_utility/results/local/batched_run
```

This uses all 569 records and 30 features of the bundled breast-cancer dataset
as a numeric software example, not a clinical validation. Fixed quantum features
are acquired in batches; a classical SGD readout reuses the cached features.
Linear, random-Fourier and histogram-boosting controls avoid dense N-by-N kernel
storage. Read [the scaling contract](SCALING.md) before increasing the budget.

## Bring your own data

Edit the generated `project/project.json`: set the numeric features, target,
ID/group/time columns, split, SDK and resource limits. Relative CSV paths are
relative to that config file. Do not include target/ID/group/time as features.

```bash
qml-utility recommend --profile my_project.json --output portfolio.json
qml-utility run --config my_project.json --output QML_utility/results/local/my_run
qml-utility audit --run QML_utility/results/local/my_run --config my_project.json
```

For a new modality, use [the extension contract](EXTENDING.md) and
[the project agent brief](PROJECT_AGENT_BRIEF.md). These numeric adapters do not
replace graph, text, continual-learning or physical-acquisition protocols.

## Evaluate more than one convenient split

Add `evaluation` to the config, then execute:

```json
{"evaluation":{"kind":"repeated_holdout","seeds":[7,17,27],"baseline":"linear","practical_margin":0.01}}
```

```bash
qml-utility evaluate --config my_project.json --output QML_utility/results/local/my_evaluation
```

Read [the evaluation guide](EVALUATION.md) for group repeats, expanding windows,
paired effects, failures and dependence. The margin uses primary metric units:
0.01 balanced accuracy is one percentage point; an RMSE margin uses target units.

## Reuse a model

```bash
qml-utility predict --model QML_utility/results/local/my_run/quantum_features.joblib --csv new_records.csv --id-column record_id --output predictions.csv
```

The artifact fixes feature order. Input needs those features but no labels.
The prediction sidecar records data/model/output hashes and preparation counts.
Load only trusted joblib files; they contain executable Python objects. Native
models still run a simulator for new observations. An explicit `--state-budget`
can set an additional preparation allowance for that prediction operation.

## Classical-only deployment

```bash
qml-utility surrogate_demo --output QML_utility/results/local/student_demo
```

This trains a small reference teacher, freezes it, distills its feature map,
checks a separate held-out gate and exports a classical student only if it passes.
It is a synthetic integration example, not Kipu's proprietary DQFE model or an
industrial accuracy claim. Use [offline distillation](OFFLINE_DISTILLATION.md)
for your own frozen teacher or recorded measured features.

The research library and ten published cases remain available through
[the companion map](COMPOSING_METHODS.md) and [case selection](CASE_SELECTION.md).
The package includes utility evidence/protocol resources; companion links into
the main repository require the full clone. None of the example commands
publishes artifacts or changes repository visibility.
