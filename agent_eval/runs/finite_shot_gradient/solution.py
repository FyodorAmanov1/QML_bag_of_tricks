"""Regenerate exact/shot local-loss diagnostics using the public native API."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from qml_tricks import RunConfig
from qml_tricks.fixtures import initial_parameters
from qml_tricks.strategies.local_loss.qiskit_impl import run


def main():
    config = RunConfig(n_qubits=4, seed=37, steps=0, initializations=8,
                       repetitions=8, variant="strategy")
    exact = run(config).to_dict()
    sampled = run(replace(config, shots=4096)).to_dict()
    # The local product-state loss is mean(sin(theta_j/2)**2).
    analytic = np.array([
        np.sin(initial_parameters("local_loss", 4, config.depth, 37 + k,
                                  "strategy")[0]) / 8
        for k in range(config.initializations)
    ])
    analytic_variance = float(np.var(analytic, ddof=1))
    # One initial trace evaluation, two exact references per initialization,
    # and two sampled evaluations per repetition per initialization.
    sampled_evaluations = 1 + 2 * config.initializations * config.repetitions
    all_evaluations = sampled_evaluations + 2 * config.initializations
    assert exact["resources"]["objective_evaluations"] == all_evaluations
    assert sampled["resources"]["objective_evaluations"] == all_evaluations
    assert sampled["resources"]["sampled_objective_evaluations"] == sampled_evaluations
    assert sampled["resources"]["simulated_measurement_shots"] == sampled_evaluations * 4096
    assert exact["resources"]["simulated_measurement_shots"] == 0
    assert len(exact["training_trace"]) == len(sampled["training_trace"]) == 1
    np.testing.assert_allclose(exact["gradient_statistics"]["exact_coordinate_variance"],
                               analytic_variance, rtol=1e-8)
    step = sampled["gradient_statistics"]["difference_step"]
    np.testing.assert_allclose(sampled["gradient_statistics"]["exact_coordinate_variance"],
                               analytic_variance * (np.sin(step) / step)**2,
                               rtol=1e-10)
    report = {
        "task": "finite_shot_gradient", "seed": 37,
        "exact": exact, "sampled": sampled,
        "comparison": {
            "coordinate": 0,
            "analytic_derivative_variance_on_same_initializations": analytic_variance,
            "sampled_minus_exact_loss": sampled["loss"] - exact["loss"],
            "sampled_minus_exact_reported_reference_variance":
                sampled["gradient_statistics"]["exact_coordinate_variance"]
                - exact["gradient_statistics"]["exact_coordinate_variance"],
            "sampled_noise_corrected_variance_minus_same_step_reference":
                sampled["gradient_statistics"]["noise_corrected_variance"]
                - sampled["gradient_statistics"]["exact_coordinate_variance"],
            "sampled_reference_finite_difference_scale": float(np.sin(step) / step),
        },
        "shot_accounting": {
            "shots_per_sampled_objective": 4096,
            "sampled_objective_evaluations": sampled_evaluations,
            "exact_reference_evaluations_in_sampled_run": 2 * config.initializations,
            "total_objective_evaluations_per_run": all_evaluations,
            "total_sampled_shots": sampled_evaluations * 4096,
            "readout": "One computational-basis bitstring gives all four local diagonal terms; shots are not multiplied by qubit count.",
        },
        "limitations": [
            "Central finite differences have O(h^2) truncation bias. The shot run uses h=0.1, whereas the exact run uses h=1e-5; even its exact reference gradients use h=0.1. For this local loss the finite difference equals the true derivative times sin(h)/h. The sampled finite-difference estimate is not an unbiased parameter-shift estimator.",
            "The loss is linear in measured probabilities, so multinomial shot sampling adds variance without plug-in loss bias. Expected sampled differences equal the biased central differences. Noise-corrected variance can fluctuate or become negative with small ensembles; eight initializations and eight repetitions give limited uncertainty resolution.",
            "Each plus/minus evaluation uses 4096 fresh simulated shots. Zero training steps still require 1+2*8*8=129 sampled objectives, totaling 528384 shots. Sixteen extra exact reference objectives use no shots; 145 probability evaluations are made per run. The exact run also repeats objectives but samples no shots.",
            "All native probabilities are computed with a classical Qiskit statevector simulator, then sampled on the host; no hardware was executed. Input loading is four RY angles into the all-zero state, with four native single-qubit gates and no ancillas or entangling gates. Readout concerns a single diagnostic coordinate, not a full gradient norm.",
            "One four-qubit complex128 statevector needs 256 bytes; the hypothetical dense density matrix needs 4096 bytes, excluding temporary arrays, probability vectors, Python, SDK, and parameter overhead. Only four of the stored parameter coordinates drive gates in this warm-up circuit.",
            "This analytically checked product-state initialization diagnostic establishes neither general plateau avoidance nor training success, generalization, or quantum advantage. It is not an implementation of the publication's general alternating-layer theorem.",
        ],
    }
    destination = Path(__file__).resolve().parent / "report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(destination), "sampled_shots": sampled_evaluations * 4096,
                      "analytic_variance": analytic_variance,
                      "sampled_noise_corrected_variance": sampled["gradient_statistics"]["noise_corrected_variance"]}))


if __name__ == "__main__":
    main()
