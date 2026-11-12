# Run and adapt the three application models

The repository now includes `TfidfQuantumMatcher`, `MulticlassQuantumKernel` and
`AmplitudeQVCClassifier`. Their model families have peer-reviewed reported
successes; these particular implementations are **reduced reference adaptations**.
Their small local benchmarks are separate from the published comparisons.

## Install and run

From a checkout with Python 3.12, install one SDK and the learning dependency:

```bash
python -m pip install -e ".[qiskit,learning]"
python -m qml_tricks.learning.case_models --case entity_matching --framework qiskit --output results/local/entity.json
python -m qml_tricks.learning.case_models --case multiclass_kernel --framework qiskit --output results/local/multiclass.json
python -m qml_tricks.learning.case_models --case adversarial_qvc --framework qiskit --output results/local/robustness.json
```

Use `.[pennylane,learning]` or `.[cirq,learning]` and change `--framework` to
select another SDK. The multiclass example selects configurations on a
validation partition. The trainable heads have fixed settings and four SPSA
steps by default; change `--steps` for a new, labeled experiment. Select that
budget without inspecting test outcomes.

## Use your own data

The classes accept ordinary Python/NumPy data and provide `fit` and `predict`.
The two parametric classifiers also expose `predict_proba`.

```python
from qml_tricks.learning.case_models import (
    TfidfQuantumMatcher, MulticlassQuantumKernel, AmplitudeQVCClassifier,
)

# Split data first. Choose settings on validation data, then freeze the model.
matcher = TfidfQuantumMatcher(framework="qiskit", steps=20, seed=7)
matcher.fit(train_text_pairs, train_match_labels)  # labels: 0 or 1
match_predictions = matcher.predict(test_text_pairs)

kernel = MulticlassQuantumKernel(framework="pennylane", topology="linear", C=1.)
kernel.fit(X_train, y_train)
kernel_predictions = kernel.predict(X_test)

qvc = AmplitudeQVCClassifier(framework="cirq", depth=2, steps=20, seed=7)
qvc.fit(image_vectors_train, y_train)
qvc_predictions = qvc.predict(image_vectors_test)
```

`train_text_pairs` is a sequence of `(left_text, right_text)` pairs. Numerical
models need finite two-dimensional arrays. The kernel uses one qubit per
feature; apply any feature reduction using training data only. The QVC needs
at least one readout qubit per class and enough amplitude qubits to load the
vector. The reference implementation limits statevectors to 12 qubits; this
protects a teaching example from accidental large allocations and is not a
scalability result.

## Native construction and shared mathematics

The independently authored circuit sources are downloadable:
[Qiskit](../source/case_models/qiskit.py),
[PennyLane](../source/case_models/pennylane.py) and
[Cirq](../source/case_models/cirq.py).
Shared [model/training code](../source/case_models/models.py) contains numerical
preprocessing, SPSA and classifiers; [benchmark code](../source/case_models/benchmarks.py)
contains fixed fixtures, validation selection, classical controls and attack
diagnostics. There is no SDK-to-SDK circuit conversion.

The backends are generated from `scripts/assemble_learning.py`. Edit that
source, regenerate, inspect changes and run numerical/structural tests. The
existing 99 recipe entrypoints remain; these are three additional application
model classes with nine native state constructors.

## What the saved reference report establishes

Install `.[all,learning]`, then run all three models in all three SDKs with:

```bash
python scripts/run_case_model_reference.py
```

The [saved nine-run report](../source/results/case_models_reference.json)
records environments, source hashes, split IDs, validation choices, training
losses, targets, predictions, warnings, timings and state-evaluation counts.
The robustness runs also retain perturbed inputs. The entity fixture splits
underlying template entities across partitions. None uses the author dataset,
trained checkpoint and complete protocol together.

All runs use one seed and exact classical statevector simulation. No hardware
jobs or finite-shot measurements are performed. State-evaluation counts are
simulator calls, not claims of equivalent cheap hardware acquisition. The kernel
report separately gives overlap-query counts for a hypothetical measurement
implementation. Stronger controls, repeated splits and realistic noise remain
project obligations. These reports preserve classical wins and exact classical
equivalences rather than requiring a favorable quantum result.

In the saved seed-7 runs, all SDKs give the same task accuracies. Entity
matching ties TF-IDF and the logistic control at 80%; the selected Iris quantum
and classical SVCs tie at 93.33%. The short QVC fit has 50% clean accuracy,
62.5% on the classical-source perturbations and 0% on its own white-box
perturbations. The independent classical target has 100%, 0% and 95.83%,
respectively. The QVC is underfit: these diagnostics do not establish a useful
defense. The captured MLP convergence warning is retained in every SDK run.
