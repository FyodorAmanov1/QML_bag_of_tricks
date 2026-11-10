"""Generate three readable notebooks; execute separately with run_notebooks.py."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def notebook(name,title,intro,cells):
    setup='''import importlib.util, os, subprocess, sys
# Locally: clone this repository, create a Python 3.12 environment, and install
# pip install -e ".[qiskit,learning,notebooks]"
# In Colab: set the public GitHub URL here once the repository is published.
REPO_URL = os.environ.get("QML_TRICKS_REPO_URL", "")
if importlib.util.find_spec("qml_tricks") is None:
    if not REPO_URL:
        raise RuntimeError("Set REPO_URL to the public GitHub repository, or install this checkout first.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"qml-bag-of-tricks[qiskit,learning,notebooks] @ git+{REPO_URL}"])
FRAMEWORK = "qiskit"  # switch to pennylane or cirq after installing its extra
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from qml_tricks.learning.models import QMLModel
ARTIFACTS = Path("results/local/notebooks")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
print("Python:", sys.version.split()[0], "Framework:", FRAMEWORK)
'''
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('# '+title+'\n\n'+intro),nbf.v4.new_markdown_cell('## 1. Set up\n\nExpected runtime is measured in the execution report, not guaranteed for every machine. The examples run locally on simulators; no QPU account is needed.'),nbf.v4.new_code_cell(setup)]+[nbf.v4.new_markdown_cell(text) if kind=='md' else nbf.v4.new_code_cell(text) for kind,text in cells])
    nb.metadata.update(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python',version='3.12'))
    dest=ROOT/'notebooks'/name;dest.parent.mkdir(exist_ok=True);nbf.write(nb,dest)


notebook('01_classification.ipynb','Classify real tabular data and compare a classical baseline',
'Learn the complete fit/predict workflow with a bounded real wine dataset. This is a small binary subset, not evidence of production performance or quantum advantage.',[
('md','## 2. Make an untouched test partition\n\nThe packaged dataset supplies provenance. Classes 0 and 1 are retained. Split **before** fitting scaling or PCA; this notebook uses fixed settings rather than tuning on the test set.'),
('code','''from qml_tricks.learning.study import dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
X, y, _, provenance = dataset("wine_binary", 96)
train, test = train_test_split(np.arange(len(X)), test_size=.25, random_state=7, stratify=y)
assert not set(train) & set(test)
print(provenance["used_shape"], provenance["content_sha256"])
'''),
('md','## 3. Fit both models\n\nThe quantum feature encoder is fixed and its regularized readout is trained classically. The classical baseline uses all raw features after train-only scaling. We compare complete pipelines; differences do not isolate a quantum mechanism.'),
('code','''quantum = QMLModel("frozen_features", framework=FRAMEWORK, n_qubits=2, seed=7)
classical = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
quantum.fit(X[train], y[train]); classical.fit(X[train], y[train])
assert np.allclose(quantum.scaler_.mean_, X[train].mean(axis=0))
reports = {}
for label, model in [("Quantum frozen features", quantum), ("Classical logistic", classical)]:
    predictions = model.predict(X[test])
    reports[label] = {"accuracy": float(accuracy_score(y[test], predictions)), "balanced_accuracy": float(balanced_accuracy_score(y[test], predictions))}
print(json.dumps(reports, indent=2))
'''),
('code','''plt.bar(list(reports), [r["balanced_accuracy"] for r in reports.values()], color=["#4658d6", "#12a48d"])
plt.ylim(0,1); plt.ylabel("Test balanced accuracy"); plt.title("One fixed split: compare with repeated-seed study")
plt.xticks(rotation=12); plt.tight_layout(); plt.show()
print(json.dumps(quantum.resources_, indent=2))
'''),
('md','## 4. Save, reload, and decide what to try next\n\nA classical model may win; report that result. For stronger conclusions run the real-data study with validation tuning and repeated seeds. Change the CSV input only after checking group/time leakage. Load only model artifacts you trust.'),
('code','''path = ARTIFACTS / "wine_frozen_features.joblib"
quantum.save(path)
restored = QMLModel.load(path)
assert np.array_equal(restored.predict(X[test]), quantum.predict(X[test]))
(ARTIFACTS / "classification.json").write_text(json.dumps(reports, indent=2))
print("Saved:", path)
''')])

notebook('02_regression.ipynb','Regress a real target with quantum and classical features',
'Use a bounded subset of real diabetes measurements to learn preprocessing, regression metrics, and saved-model parity. This is a software benchmark, not a medical decision model.',[
('md','## 2. Split before preprocessing\n\nThe QML estimator scales targets internally using training statistics. The classical control also scales targets inside its fit. RMSE and MAE below are returned in the original target units.'),
('code','''from qml_tricks.learning.study import dataset
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
X, y, _, provenance = dataset("diabetes", 128)
train, test = train_test_split(np.arange(len(X)), test_size=.25, random_state=17)
assert not set(train) & set(test)
quantum = QMLModel("frozen_features", task="regression", framework=FRAMEWORK, seed=17)
classical = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), Ridge(alpha=.1)), transformer=StandardScaler())
quantum.fit(X[train], y[train]); classical.fit(X[train], y[train])
'''),
('md','## 3. Evaluate once on held-out rows\n\nLower error is better. Two qubits and PCA are a deliberate small representation: the stronger full-feature classical control is essential.'),
('code','''reports = {}; predictions = {}
for label, model in [("Quantum frozen features", quantum), ("Classical ridge", classical)]:
    pred = model.predict(X[test]); predictions[label] = pred
    reports[label] = {"rmse": float(np.sqrt(mean_squared_error(y[test], pred))), "mae": float(mean_absolute_error(y[test], pred))}
print(json.dumps(reports, indent=2))
for label, pred in predictions.items(): plt.scatter(y[test], pred, label=label, alpha=.7)
limits=[min(y[test]), max(y[test])]; plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.xlabel("Observed target"); plt.ylabel("Predicted target"); plt.legend(); plt.tight_layout(); plt.show()
'''),
('md','## 4. Save and reproduce\n\nAvoid selecting settings from this test plot. Use the separate repeated-seed, validation-selected benchmark for comparative evidence.'),
('code','''assert np.allclose(quantum.scaler_.mean_, X[train].mean(axis=0))
path = ARTIFACTS / "diabetes_frozen_features.joblib"; quantum.save(path)
assert np.allclose(QMLModel.load(path).predict(X[test]), predictions["Quantum frozen features"])
(ARTIFACTS / "regression.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(quantum.resources_, indent=2))
''')])

notebook('03_offline_surrogate.ipynb','Train a quantum teacher, then deploy a classical feature surrogate',
'Follow Kipu’s public offline feature-distillation protocol using a small native reference teacher. This does not reproduce Kipu’s proprietary DQFE teacher or industrial results. Faithful imitation and useful task accuracy are separate measurements.',[
('md','## 2. Train and freeze the teacher first\n\nUse independent training, validation and test inputs. Labels define a bounded synthetic classification task. Only the first 100 training rows fit the native teacher; the larger training acquisition set teaches the student.'),
('code','''from qml_tricks.learning.offline_surrogate import OfflineFeatureSurrogate, ClassicalFeatureStudent
rng = np.random.default_rng(7)
X = rng.uniform(-1, 1, (512,2)); y = (np.sum(X**2, axis=1) > .7).astype(int)
teacher = QMLModel("reuploading", framework=FRAMEWORK, seed=7, steps=3).fit(X[:100],y[:100])
frozen_parameters = teacher.theta_.copy()
distiller = OfflineFeatureSurrogate(alpha=1e-6, student_kind="rbf_ridge")
distiller.fit_teacher(teacher, X[:384], y[:384], X[384:448], y[384:448])
assert np.array_equal(teacher.theta_, frozen_parameters)
print(json.dumps(distiller.validation_, indent=2))
'''),
('md','## 3. Audit the untouched test inputs\n\nBoth feature fidelity and teacher/student task accuracy are printed. High agreement can faithfully reproduce a weak teacher. The validation gate is an empirical criterion, not a guaranteed confidence bound. The RBF-ridge student is a local extension; Ridge is the family reported in Kipu’s public example.'),
('code','''Q_test = distiller.teacher_.quantum_features(X[448:])
audit = distiller.audit_pairs(X[448:], Q_test, y[448:])
print(json.dumps(audit, indent=2))
predicted_features = distiller.surrogate_.predict(X[448:])
plt.scatter(Q_test.ravel(), predicted_features.ravel(), alpha=.6)
plt.plot([-1,1],[-1,1],"k--"); plt.xlabel("Teacher feature"); plt.ylabel("Classical surrogate feature")
plt.title("Held-out feature fidelity, not task accuracy"); plt.tight_layout(); plt.show()
'''),
('md','## 4. Export only if the gate passes\n\nThe exported student has no teacher reference. For real Kipu measurements, use `fit_from_pairs(X_train, Q_train, y_train, X_validation, Q_validation, y_validation)`. No density matrix or quantum device is needed for student inference.'),
('code','''path = ARTIFACTS / "classical_student.joblib"
distiller.export_student(path)
student = ClassicalFeatureStudent.load(path)
assert not hasattr(student, "teacher_")
before = distiller.teacher_.resources_["state_preparations"]
predictions = student.predict(X[448:])
assert distiller.teacher_.resources_["state_preparations"] == before
assert np.array_equal(predictions, distiller.student().predict(X[448:]))
(ARTIFACTS / "surrogate.json").write_text(json.dumps(audit, indent=2))
print("Classical-only artifact:", path, "Test predictions:", len(predictions))
''')])
print('Generated three walkthrough notebooks.')
