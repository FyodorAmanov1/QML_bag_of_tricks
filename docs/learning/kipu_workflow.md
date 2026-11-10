# Kipu-style offline feature distillation: the intended surrogate

The earlier density-embedding fidelity-kernel approximation was a different
method. Kipu's [public offline DQFE workflow](https://kipu-quantum.com/blog/classical-surrogates-for-quantum-feature-extraction)
collects input/quantum-feature pairs and fits a classical regressor to replace
the quantum feature extractor. Its published example uses a Ridge surrogate.
This vendor description is not a universal accuracy theorem.

## Train the quantum teacher first

Freeze the teacher before fitting its classical mimic. If its circuit/kernel
has task-trained parameters, complete that training first. Kipu's public
description does not require variational kernel optimization for every feature
map. The local reference below uses a trained re-uploading circuit; it does
not implement Kipu's proprietary DQFE circuit, service, datasets or reported gains.

```python
from qml_tricks.learning.models import QMLModel
from qml_tricks.learning.offline_surrogate import OfflineFeatureSurrogate

teacher = QMLModel("reuploading", framework="qiskit", steps=20)
teacher.fit(X_train, y_train)
distiller = OfflineFeatureSurrogate(student_kind="ridge")
distiller.fit_teacher(teacher, X_train, y_train, X_validation, y_validation)
print(distiller.validation_)
distiller.export_student("offline_student.joblib")  # rejects a failed gate
```

The adapter copies the fitted circuit, acquires its local measured features,
fits a Ridge predictor on the actual features, then freezes that predictor
while the classical regressor learns the feature mapping. It does not silently
retrain the supplied circuit or claim to preserve an arbitrary pre-existing
nonlinear downstream prediction head.

## Supply real Kipu measurements

Use recorded feature pairs to replace the reference simulator teacher:

```python
distiller.fit_from_pairs(
    X_train, Q_train, y_train,
    X_validation, Q_validation, y_validation,
)
student = distiller.student()
predictions = student.predict(X_test)
```

Q_train and Q_validation are measured feature vectors from the same frozen map,
with the same feature ordering and normalization. No density matrix is needed.
The adapter performs no Kipu service calls and requires no provider credentials.
If only Gram entries are available, they need a separate consistent kernel
distillation representation; arbitrary train/test Gram rows are not interchangeable
with the measured feature vectors expected by this API.

## What “high accuracy” means

The model reports feature RMSE, relative RMSE, R-squared, teacher/student prediction
agreement, and downstream task performance. Classification export defaults require
at least 95% held-out agreement, at most 2 percentage points accuracy loss, and
relative feature RMSE at most 0.2. Regression gates feature RMSE and reports
prediction RMSE and both task MAEs. These are empirical acceptance checks,
not confidence-certified universal deployment guarantees.

Task usefulness needs its own absolute quality threshold: a poor teacher can
be mimicked perfectly. Keep a final untouched test set and do not repeatedly
optimize the same validation set. Preserve failed runs.

Ridge is the default student, matching the family in Kipu's reported example.
Optional polynomial features and RBF-ridge students are local extensions.
The bounded native demo uses an RBF-ridge student, reports validation and test
results, and is an integration example rather than an industrial performance claim.

```bash
python -m qml_tricks learning-demo kipu_offline_surrogate --framework qiskit
python -m qml_tricks learning-demo kipu_offline_surrogate --framework pennylane
python -m qml_tricks learning-demo kipu_offline_surrogate --framework cirq
```

The exported ClassicalFeatureStudent contains classical transforms, surrogate
weights, the frozen readout and validation metadata, with no teacher reference.
Inference uses no quantum backend. Tests cover teacher freezing, failed-gate
rejection, held-out separation and teacher-free reload.
