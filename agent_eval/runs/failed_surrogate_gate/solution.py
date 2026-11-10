"""Demonstrate public export-gate rejection on unrelated synthetic pairs."""
import json
from pathlib import Path

import numpy as np
from qml_tricks.learning.offline_surrogate import OfflineFeatureSurrogate


def main():
    destination = Path(__file__).resolve().parent
    # Separate random streams make both features independent of their inputs,
    # and the held-out inputs and targets independent of fitting observations.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(43).spawn(4)]
    x_train = streams[0].normal(size=(80, 4))
    q_train = streams[1].uniform(-1, 1, size=(80, 3))
    x_validation = streams[2].normal(size=(60, 4))
    q_validation = streams[3].uniform(-1, 1, size=(60, 3))
    y_train = (q_train[:, 0] > 0).astype(int)
    y_validation = (q_validation[:, 0] > 0).astype(int)
    assert set(map(tuple, x_train)).isdisjoint(map(tuple, x_validation))
    distiller = OfflineFeatureSurrogate(
        task="classification", alpha=1e-3, student_kind="ridge",
        min_agreement=0.99, max_accuracy_drop=0.0,
        max_relative_feature_rmse=0.01,
    ).fit_from_pairs(x_train, q_train, y_train,
                     x_validation, q_validation, y_validation)
    validation = distiller.validation_
    assert validation["passed"] is False
    artifact = destination / "rejected_student.joblib"
    assert not artifact.exists(), "Unexpected preexisting artifact; preserve it and investigate"
    exception_type = None
    message = None
    try:
        distiller.export_student(artifact)
    except ValueError as error:
        exception_type = type(error).__name__
        message = str(error)
    export_rejected = exception_type == "ValueError"
    assert export_rejected, "A failed fidelity gate unexpectedly allowed export"
    assert not artifact.exists()
    assert not Path(str(artifact) + ".validation.json").exists()
    thresholds = validation["acceptance_thresholds"]
    report = {
        "task": "failed_surrogate_gate", "seed": 43,
        "validation": validation,
        "export_rejected": export_rejected,
        "exception_type": exception_type,
        "exception_message": message,
        "artifact_path_attempted": str(artifact),
        "artifact_created": artifact.exists(),
        "failed_checks": {
            "agreement": validation["teacher_student_agreement"] < thresholds["min_agreement"],
            "accuracy_drop": validation["accuracy_drop"] > thresholds["max_accuracy_drop"],
            "relative_feature_rmse": validation["relative_feature_rmse"] > thresholds["max_relative_feature_rmse"],
        },
        "data": {
            "train_rows": len(x_train), "validation_rows": len(x_validation),
            "input_dimension": 4, "feature_dimension": 3,
            "train_validation_inputs_disjoint": True,
            "construction": "Independent random streams for training inputs, unrelated feature targets, validation inputs and independent validation feature targets; labels derive from each target's first coordinate.",
            "feature_source": "Synthetic numerical targets only; no quantum acquisition or proprietary circuits",
            "preprocessing": "Repository StandardScaler and Ridge pipeline fitted on training inputs only",
        },
        "resources": {**distiller.resources_, "actual_quantum_calls": 0,
                      "hardware_executed": False, "quantum_shots": 0},
        "limitations": [
            "This intentionally invalid input-to-feature relationship is a negative control for the validation and export gate. A deterministic frozen teacher would not produce independent random feature targets on independent inputs. It does not estimate performance of any physical teacher or real task.",
            "Accurate imitation alone would not establish useful task accuracy: a perfectly copied teacher can be inaccurate, biased or uninformative for the target labels. Absolute task accuracy on untouched test data and appropriate classical controls remain necessary, even after high feature fidelity and teacher-student agreement.",
            "The held-out empirical thresholds are strict acceptance checks, not universal generalization guarantees. Thresholds and model were fixed before validation; the gate was neither overwritten nor bypassed, and no student was exported.",
            "All observations are classical synthetic arrays; no loading, quantum readout, shots or statevector simulation was performed. The synthetic task provides no quantum advantage, industrial-performance or proprietary Kipu DQFE evidence, and concerns neither density-embedding approximation nor Nystrom.",
        ],
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"export_rejected": export_rejected, "exception_type": exception_type,
                      "validation": validation, "artifact_created": artifact.exists()}))


if __name__ == "__main__":
    main()
