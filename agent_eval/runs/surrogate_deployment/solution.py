"""Synthetic recorded-pair deployment through the repository's fidelity gate."""
from pathlib import Path
import builtins
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from qml_tricks.learning.offline_surrogate import (
    ClassicalFeatureStudent, OfflineFeatureSurrogate,
)


def main():
    output = Path(__file__).resolve().parent
    seed = 41
    rng = np.random.default_rng(seed)
    # This known map generates synthetic recorded features; it is not a quantum
    # circuit or a reconstruction of a proprietary feature extractor.
    mapping = np.array([[0.5, -0.1], [-0.2, 0.4], [0.1, 0.3]])
    def data(count):
        x = rng.uniform(-1, 1, size=(count, 3))
        q = x @ mapping
        y = (q[:, 0] - 0.35*q[:, 1] > 0).astype(int)
        return x, q, y

    x_train, q_train, y_train = data(96)
    x_validation, q_validation, y_validation = data(48)
    x_test, q_test, y_test = data(48)
    assert len({tuple(row) for row in np.vstack([x_train, x_validation, x_test])}) == 192
    model = OfflineFeatureSurrogate(task="classification", student_kind="ridge",
                                    alpha=1e-6, degree=1)
    model.fit_from_pairs(x_train, q_train, y_train,
                         x_validation, q_validation, y_validation)
    assert model.validation_["passed"]
    artifact = output / "offline_student.joblib"
    model.export_student(artifact)
    restored = ClassicalFeatureStudent.load(artifact)
    assert set(vars(restored)) == {"surrogate", "head", "classes_", "validation_"}

    # Guard actual restored inference against loading any supported quantum SDK
    # or repository backend. This script has no teacher object or quantum calls.
    native_imports = []
    original_import = builtins.__import__
    def classical_only_import(name, *args, **kwargs):
        if (name.split(".")[0] in {"qiskit", "cirq", "pennylane"}
                or name.startswith("qml_tricks.learning.backends")):
            native_imports.append(name)
            raise RuntimeError("Unexpected quantum dependency during student inference")
        return original_import(name, *args, **kwargs)
    builtins.__import__ = classical_only_import
    try:
        prediction = restored.predict(x_test)
        features = restored.transform(x_test)
        probabilities = restored.predict_proba(x_test)
    finally:
        builtins.__import__ = original_import
    reference = model.student().predict(x_test)
    assert np.array_equal(prediction, reference)
    assert not native_imports
    test_audit = model.audit_pairs(x_test, q_test, y_test)
    report = {
        "task": "surrogate_deployment", "seed": seed,
        "validation": model.validation_,
        "inference_quantum_calls": 0,
        "quantum_backend_import_attempts": native_imports,
        "teacher_free_artifact": True,
        "artifact": str(artifact.relative_to(ROOT)),
        "test_predictions": prediction.tolist(),
        "test_labels": y_test.tolist(),
        "test_probabilities": probabilities.tolist(),
        "test_accuracy": float(np.mean(prediction == y_test)),
        "test_feature_rmse": float(np.sqrt(np.mean((features-q_test)**2))),
        "test_audit": test_audit,
        "reload_predictions_identical": bool(np.array_equal(prediction, reference)),
        "known_linear_map": mapping.tolist(),
        "split_sizes": {"train": 96, "validation": 48, "test": 48},
        "resources": model.resources_,
        "preprocessing_train_mean": model.surrogate_[0].mean_.tolist(),
        "actual_train_mean": x_train.mean(axis=0).tolist(),
        "limitations": [
            "Synthetic recorded pairs from a known linear map; no quantum data were acquired and no proprietary circuits were implemented.",
            "Independent IID train, validation, and untouched test inputs; fixed thresholds and hyperparameters with no validation tuning.",
            "The gate measures held-out imitation and downstream accuracy retention. Accurate imitation of a poor teacher would not establish useful task accuracy.",
            "All input loading, readout fitting, storage, and inference are classical. Zero shots and no statevector simulation. Pair storage scales with rows times feature dimension.",
            "No industrial performance or quantum advantage claim. Export is contingent on the repository's held-out validation gate."
        ]
    }
    np.testing.assert_allclose(model.surrogate_[0].mean_, x_train.mean(axis=0), rtol=0, atol=0)
    np.savez(output / "recorded_pairs.npz", known_linear_map=mapping,
             x_train=x_train, q_train=q_train, y_train=y_train,
             x_validation=x_validation, q_validation=q_validation, y_validation=y_validation,
             x_test=x_test, q_test=q_test, y_test=y_test)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation_passed": model.validation_["passed"],
                      "validation_relative_feature_rmse": model.validation_["relative_feature_rmse"],
                      "validation_agreement": model.validation_["teacher_student_agreement"],
                      "test_accuracy": report["test_accuracy"],
                      "inference_quantum_calls": 0,
                      "reload_predictions_identical": report["reload_predictions_identical"]}))


if __name__ == "__main__":
    main()
