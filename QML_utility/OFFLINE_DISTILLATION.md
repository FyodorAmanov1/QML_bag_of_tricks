# Freeze the quantum feature map, deploy a classical student

This is the intended [Kipu-style offline workflow](https://kipu-quantum.com/blog/classical-surrogates-for-quantum-feature-extraction):
learn a classical mapping from raw inputs to the frozen teacher's measured
features. Kipu's description is vendor evidence; this engineering companion
is not another peer-reviewed success case, a replica of proprietary DQFE or a
reproduction of its industrial accuracy figures.

If the teacher has trainable parameters, finish their training first. The
student does not train, copy or retain the quantum circuit. It preserves a
supported frozen classical head, or the trained local model's linear-logit head,
while a Ridge or random-Fourier-plus-Ridge mapper replaces acquisition.

```python
from QML_utility.offline_surrogate import OfflineFeatureStudent

gates = dict(
    max_feature_rmse=0.02,
    max_feature_relative_l2=0.15,
    max_prediction_disagreement=0.05,
    max_task_degradation=0.02,
    min_teacher_balanced_accuracy=0.80,
)
student = OfflineFeatureStudent(kind="ridge", alpha=1.0)
student.fit_from_teacher(
    frozen_fitted_teacher, X_distill, X_gate, y_gate,
    distill_ids=distill_ids, gate_ids=gate_ids,
    teacher_used_ids=all_teacher_train_selection_and_evaluation_ids,
    gates=gates,
)
print(student.gate_report_)
student.export("student.joblib")  # refuses export if any gate fails
```

The numbers above illustrate explicit tolerances; choose practical thresholds
before evaluating your gate data. Regression instead requires
`max_prediction_rmse` and `max_teacher_rmse`, expressed in target units, along
with the same feature and task-degradation bounds. A perfectly mimicked but poor
teacher should fail the declared teacher-quality requirement.

## Separate the data roles

- Teacher fit/selection/evaluation records belong to the teacher's prior history.
- Distillation records fit the classical feature mapper; they may include teacher
  training records when appropriate.
- Gate records must be disjoint from both, identified explicitly. No mapper
  hyperparameter selection is performed on them.
- Keep a further untouched final test for evaluation after the export decision.

Record IDs establish only declared record disjointness. They cannot detect an
undeclared reuse, repeated patient, shared molecular scaffold or future-information
leakage. Use a scientifically appropriate group/time/external split. Repeatedly
changing student settings after seeing the same gate turns that gate into tuning
data; reserve a fresh gate for the new frozen student.

## What is measured

Feature RMSE compares measured-feature vectors. Relative L2 divides their total
error by the gate teacher features' deviation from the distillation feature mean.
A nearly constant teacher makes this relative measure unstable; an unresolved
zero denominator fails unless the feature error is also negligible. This is
feature fidelity, not quantum-state fidelity.

The gate also records teacher/student prediction agreement, actual downstream
balanced accuracy or RMSE, permitted degradation and the declared minimum teacher
quality. Every check must pass. These are empirical acceptance checks, not
confidence-certified fidelity bounds. Teacher feature acquisition and its
preparation/shot/job counts are separate from student inference.

## Supply real measured pairs

`fit_from_pairs` accepts numeric raw/quantum-feature pairs from a frozen provider
without requiring an SDK, statevector or density matrix:

```python
from QML_utility.offline_surrogate import classical_head_spec

spec = classical_head_spec(fitted_classical_head, "classification", classes=labels)
student.fit_from_pairs(
    X_distill, Q_distill, X_gate, Q_gate, y_gate,
    head_spec=spec, distill_ids=distill_ids, gate_ids=gate_ids,
    teacher_used_ids=prior_ids, gates=gates,
    teacher_acquisition={"physical_shots": total_shots, "qpu_jobs": total_jobs},
)
```

Classification heads must predict integer indices into `labels`; the helper
preserves that mapping. Supported heads are stock classical logistic/ridge,
SGD, non-precomputed SVC/kernel-ridge models, with an optional fitted standard
scaler. Arbitrary callable kernels or provider callbacks are rejected. An
arbitrary fidelity-kernel Gram classifier is not a local-feature teacher; it
needs a separately justified kernel-distillation representation.

## Verify deployment

The serialized student contains only its classical mapper, frozen classical
head/arrays and gate metadata. Tests load and predict in a fresh process with
Qiskit, PennyLane, Cirq and the utility's native-model package blocked. Prediction
requires zero quantum queries. The CLI can use a trusted accepted student:

```bash
qml-utility predict --model student.joblib --csv new_data.csv --features x0 x1 --output student_predictions.csv
```

Provide the original ordered feature names. Load only trusted joblib artifacts.
This deployment path is distinct from Nyström, which retains new-input queries,
and from density-embedding approximations, which mimic a different object.
