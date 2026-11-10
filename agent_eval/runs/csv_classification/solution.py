"""Recreate a bounded CSV classification experiment with repository learning APIs.

Run from the repository root with .venv-clean/Scripts/python.exe.
The data, predictions and split indices are reproducible; elapsed timings vary.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
from sklearn.datasets import make_moons

from qml_tricks.learning.benchmark import benchmark


SEED = 29
OUTPUT = Path(__file__).resolve().parent


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, y = make_moons(n_samples=80, noise=0.16, random_state=SEED)
    csv_path = OUTPUT / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature_0", "feature_1", "label"])
        writer.writerows((float(a), float(b), int(label)) for (a, b), label in zip(x, y))

    # Fit the values read from disk so the CSV is the experiment's actual input.
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    x = np.asarray([[float(row["feature_0"]), float(row["feature_1"])] for row in rows])
    y = np.asarray([int(row["label"]) for row in rows])
    assert x.shape == (80, 2) and set(y) == {0, 1} and np.isfinite(x).all()

    # benchmark fits each scaler and each quantum PCA only on its training rows.
    # Four classical controls receive the same row splits and original features.
    report = benchmark(
        task="classification", framework="qiskit", seeds=(SEED,),
        methods=("frozen_features", "projected_kernel"),
        n_qubits=2, depth=1, steps=0, shots=None, x=x, y=y,
    )
    first = report["runs"][0]
    splits = {name: first[name] for name in ("train_indices", "validation_indices", "test_indices")}
    train, validation, test = (set(splits[name]) for name in splits)
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(len(x)))
    for run in report["runs"]:
        assert all(run[name] == indices for name, indices in splits.items())
        assert run["test_targets"] == y[splits["test_indices"]].tolist()
        assert len(run["test_predictions"]) == len(test)

    report.update(
        task_id="csv_classification",
        seed=SEED,
        dataset={"path": "data.csv", "rows": len(x), "feature_columns": ["feature_0", "feature_1"],
                 "target_column": "label", "labels": [0, 1],
                 "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                 "generation": "sklearn make_moons, noise=0.16, random_state=29; synthetic IID reference"},
        split_indices=splits,
        actual_train_mean=x[splits["train_indices"]].mean(axis=0).tolist(),
        test_targets=y[splits["test_indices"]].tolist(),
        test_predictions={run["model"]: run["test_predictions"] for run in report["runs"]},
        metrics={run["model"]: {"validation": run["validation"], "test": run["test"]}
                 for run in report["runs"]},
        resource_counts={run["model"]: {
            "training": run["training_resources"], "total": run["total_resources"],
            "test_state_preparations": run["test_state_preparations"],
            "hardware_jobs": 0,
            "measurement_shots": run["total_resources"].get("measurement_shots", 0),
        } for run in report["runs"]},
        assumptions={
            "loading": "Numeric CSV rows; train-only StandardScaler and quantum PCA followed by tanh angle encoding.",
            "readout": "Exact local Z expectation features from two-qubit native Qiskit statevectors. Frozen features use a linear ridge head; projected kernel uses an RBF SVC head.",
            "shots": "Exact simulation, shots=None, zero measurement shots; no hardware execution or noise model.",
            "classical_simulation": "All quantum state preparations are native classical simulator executions; training and inference acquisitions are included separately.",
            "memory": "Two-qubit complex128 statevector: 64 bytes per state, excluding simulator and Python overhead. Cached quantum features and classical learner/kernel matrices consume additional memory.",
            "validation": "Diagnostic only; fixed configurations and zero hyperparameter search; test rows are used only for final evaluation.",
        },
        limitations="One small synthetic dataset and one split seed; no evidence of quantum advantage, hardware performance, statistical superiority, or generalization to other datasets. Repeated timings vary.",
    )
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT / "report.json"), "seed": SEED,
                      "rows": len(x), "split_sizes": [len(train), len(validation), len(test)],
                      "test_accuracy": {run["model"]: run["test"]["accuracy"] for run in report["runs"]}}, indent=2))


if __name__ == "__main__":
    main()
