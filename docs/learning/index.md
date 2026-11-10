# From a dataset to a benchmark

This layer complements the 19 circuit guides with 14 learning/estimation guides,
three independent native preparation backends, and 42 native runnable entrypoints.
Shared estimators do not convert circuits between SDKs.

For the intended Kipu surrogate, use [teacher-first offline feature distillation](kipu_workflow.md).

## Install and run

```bash
python -m pip install -e ".[all,learning]"
python -m qml_tricks learning-demo parameter_shift --framework cirq
python -m qml_tricks benchmark --task classification --framework qiskit --output results/local/classification.json
python -m qml_tricks benchmark --task regression --framework pennylane --output results/local/regression.json
python -m qml_tricks benchmark --task quantum --framework cirq --output results/local/quantum.json
python -m qml_tricks benchmark --task generative --framework qiskit --output results/local/generative.json
```

Supervised examples use fixed synthetic datasets and repeated split seeds.
The quantum task uses **known simulator preparation descriptions**, not opaque
hardware state copies. Classical controls receive those same descriptions; this
does not establish a quantum-data-access separation. The generative task uses
correlated all-zero/all-one bits and classical independent/resampling controls.

## Methods and selection

| Need | Guide |
|---|---|
| Correct expectation derivatives | [Parameter shift](parameter_shift.md) |
| Small-register trainable model | [Encoding and re-uploading](reuploading.md) |
| Kernel classification/regression | [Fidelity](fidelity_kernel.md), [projected](projected_kernel.md) |
| Convex readout training | [Frozen features](frozen_features.md) |
| Noisy many-parameter objectives | [SPSA](spsa.md) |
| Measurement allocation | [Shot allocation](shot_allocation.md) |
| Many local quantum observables | [Classical shadows](classical_shadows.md) |
| Sample-based generative learning | [Born-machine MMD](born_mmd.md) |
| Train an encoder classically first | [Classical pretraining](classical_pretraining.md) |
| Transfer simulator-trained circuit angles | [Warm starts](classical_warm_start.md) |
| Fewer quantum kernel queries | [Nyström](nystrom.md) |
| Density-embedding fidelity-kernel approximation | [Earlier reference](kernel_surrogate.md) |
| Kipu-style offline quantum-feature distillation | [Teacher-first workflow](kipu_workflow.md) |

## Agent integration

```python
from qml_tricks.learning.models import QMLModel
model = QMLModel(method="kernel_surrogate", task="classification", framework="cirq")
model.fit(X_train, y_train)
labels = model.predict(X_test)
scores = model.predict_proba(X_test)
model.save("model.joblib")
restored = QMLModel.load("model.joblib")
```

Inputs are finite numeric matrices; classification is binary and regression has
one scalar target. Probability outputs are transformed scores, not calibrated
probabilities. Models are capped at 512 training rows and six qubits. Load only
trusted joblib artifacts: Python serialization can execute code.

`BornMachine.fit(binary_samples)` and `sample(count)` provide the generative API.
Mathematical primitives live in `qml_tricks.learning.methods`.

## CSV workflow

```bash
python -m qml_tricks benchmark --task classification --csv data.csv --target label --models frozen_features reuploading kernel_surrogate --seeds 7 17 27 --artifact-dir results/local/models --output results/local/report.json
```

CSV features must be numeric; labels may be strings. Regression targets must be
numeric. The runner assumes IID rows. Grouped, temporal, repeated-subject and
spatial data need a project-specific splitter; this runner does not automatically
handle those regimes. Custom binary samples use the BornMachine Python API.

## Benchmark contract

- Shared 60/20/20 train/validation/test indices per seed; classification is stratified.
- Fit scaling, PCA and supervised pretraining only on the training partition.
- Fixed configurations and zero hyperparameter-search trials for every model.
  Validation is diagnostic; the runner performs no automatic selection.
- Classical controls: linear/logistic, RBF SVM/SVR, small MLP and gradient boosting.
  They retain full raw features; quantum models use train-only compression.
- Pretrained models also report a classical ridge head on the same hidden features.
- Reports include held-out predictions, scores, seed-bootstrap intervals, source
  hashes, versions, warnings, training/inference time and resource counters.
- Small seed counts give weak interval coverage. No statistical superiority or
  computational advantage is certified.
- `state_preparations` means native simulator executions, not physical copies.
  `measurement_shots` counts host-simulated measurements. Exact kernel pairs are
  classical calculations on native statevectors; no QPU job is submitted.
- No hardware noise model is included in this learning layer. Exact and finite-shot
  modes are labeled; full density teachers require exponential memory.

See [the pretraining/surrogate audit](pretraining_surrogates.md) for the explicit
engineering exceptions and the difference between Nyström and classical deployment.

## Recorded reference results

`results/learning_reference.json` contains all 42 native entrypoint runs and four
workflow benchmarks, including repeated seeds, classical controls and held-out
surrogate errors. Regenerate with `python scripts/learning_reference.py`.
These are bounded synthetic integration references, not real-world performance
claims or full reproductions of the source papers.
