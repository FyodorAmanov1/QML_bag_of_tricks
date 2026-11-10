"""Independent physical, analytic, sampling, and cross-SDK checks."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
from qml_tricks.core import RunConfig, catalog, experiment, module, gradient
from qml_tricks.fixtures import initial_parameters, loss_from_probabilities

SDKS = [os.environ["QML_SDK"]] if "QML_SDK" in os.environ else ["qiskit", "pennylane", "cirq"]
IDS = [r["id"] for r in catalog()]


@pytest.mark.parametrize("sid", IDS)
@pytest.mark.parametrize("variant", ["strategy", "baseline"])
def test_native_probability_agreement(sid, variant):
    cfg = RunConfig(n_qubits=4, steps=0, variant=variant)
    theta = initial_parameters(sid, 4, cfg.depth, cfg.seed, variant)
    vectors = [module(sid, sdk).probabilities(theta, cfg) for sdk in SDKS]
    for p in vectors:
        assert np.all(p >= 0)
        assert p.sum() == pytest.approx(1, abs=1e-12)
        np.testing.assert_allclose(p, vectors[0], atol=1e-9)


@pytest.mark.parametrize("sdk", SDKS)
@pytest.mark.parametrize("variant", ["strategy", "baseline"])
def test_local_loss_and_derivative_against_closed_form(sdk, variant):
    cfg = RunConfig(variant=variant, steps=0)
    theta = initial_parameters("local_loss", 4, 2, 12, variant)
    def evaluate(t):
        return loss_from_probabilities(module("local_loss", sdk).probabilities(t, cfg), "local_loss", 4, variant)
    if variant == "strategy":
        expected = np.mean(np.sin(theta[:4]/2)**2)
        derivative = np.sin(theta[0])/8
    else:
        expected = 1 - np.prod(np.cos(theta[:4]/2)**2)
        derivative = np.sin(theta[0])/2 * np.prod(np.cos(theta[1:4]/2)**2)
    assert evaluate(theta) == pytest.approx(expected, abs=1e-12)
    assert gradient(evaluate, theta)[0] == pytest.approx(derivative, abs=1e-8)


def test_local_variance_exact_quadrature():
    # Analytic gradients integrated over the full independent angle grid.
    x = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    a, b, c = np.meshgrid(x, x, x, indexing="ij")
    local = np.sin(a)/6
    glob = np.sin(a)/2 * np.cos(b/2)**2 * np.cos(c/2)**2
    assert np.var(local) == pytest.approx(1/(8*3**2))
    assert np.var(glob) == pytest.approx((3/8)**2/8)


def test_haar_initialization_diagnostic_resamples_fixed_blocks():
    sdk = SDKS[0]
    result = experiment("mps", sdk, RunConfig(steps=0, initializations=4))
    assert result.gradient_statistics["exact_coordinate_variance"] > 1e-8


@pytest.mark.parametrize("sdk", SDKS)
def test_conservation_and_shared_parameter(sdk):
    cfg = RunConfig(steps=0)
    t = initial_parameters("low_excitation", 4, 2, 7, "strategy")
    p = module("low_excitation", sdk).probabilities(t, cfg)
    weights = np.array([i.bit_count() for i in range(16)])
    assert p[weights != 1].sum() < 1e-12
    control = module("low_excitation", sdk).probabilities(t, RunConfig(steps=0, variant="baseline"))
    assert control[weights != 1].sum() > .01
    t[0] = 0.73
    def evaluate(theta):
        p = module("correlated_parameters", sdk).probabilities(theta, cfg)
        return loss_from_probabilities(p, "correlated_parameters", 4, "strategy")
    assert evaluate(t) == pytest.approx(np.cos(.73))
    assert gradient(evaluate, t)[0] == pytest.approx(-np.sin(.73), abs=1e-8)
    baseline_cfg = RunConfig(variant="baseline", steps=0)
    def independent(theta):
        p = module("correlated_parameters", sdk).probabilities(theta, baseline_cfg)
        return loss_from_probabilities(p, "correlated_parameters", 4, "baseline")
    assert independent(t) == pytest.approx(np.mean(np.cos(t[:4])))
    assert gradient(independent, t)[0] == pytest.approx(-np.sin(.73)/4, abs=1e-8)


@pytest.mark.parametrize("sdk", SDKS)
def test_permutation_equivariance_on_output(sdk):
    cfg = RunConfig(steps=0)
    theta = initial_parameters("permutation_equivariant", 4, 2, 9, "strategy")
    p = module("permutation_equivariant", sdk).probabilities(theta, cfg)
    # Symmetric reference + equivariant unitary implies exchangeable probabilities.
    for weight in range(5):
        group = p[[i for i in range(16) if i.bit_count() == weight]]
        np.testing.assert_allclose(group, group[0], atol=1e-10)


@pytest.mark.parametrize("sid", IDS)
@pytest.mark.parametrize("sdk", SDKS)
def test_resource_reporting(sid, sdk):
    cfg = RunConfig(steps=0)
    theta = initial_parameters(sid, 4, 2, 7, "strategy")
    result = module(sid, sdk).resources(theta, cfg)
    assert result["qubits"] == 4
    assert result["native_operations"] >= result["two_qubit_operations"]
    assert result["native_depth"] >= 1


@pytest.mark.parametrize("sdk", SDKS)
def test_sampling_and_training(sdk):
    exact = experiment("local_loss", sdk, RunConfig(steps=3))
    assert exact.loss < exact.training_trace[0]
    sampled = experiment("local_loss", sdk, RunConfig(steps=0, shots=32768))
    assert abs(sampled.loss-exact.training_trace[0]) < .025
    assert sampled.resources["simulated_measurement_shots"] == sampled.resources["sampled_objective_evaluations"]*32768
    assert sampled.resources["sampled_objective_evaluations"] < sampled.resources["objective_evaluations"]
    json.dumps(sampled.to_dict(), allow_nan=False)


@pytest.mark.parametrize("cfg,sid", [(RunConfig(n_qubits=3), "qcnn"), (RunConfig(shots=1), "local_loss"), (RunConfig(n_qubits=12), "nonunital_noise")])
def test_invalid_configuration(cfg, sid):
    with pytest.raises(ValueError): cfg.validate(sid)


def test_catalog_has_guides_and_all_native_sources():
    root = Path(__file__).resolve().parents[1]
    assert len(set(IDS)) == len(IDS)
    for row in catalog():
        guide = root/"docs/strategies"/(row["id"]+".md")
        assert row["source"] in guide.read_text(encoding="utf-8")
        assert row["locator"]
        for sdk in ["qiskit", "pennylane", "cirq"]:
            assert (root/"src/qml_tricks/strategies"/row["id"]/(sdk+"_impl.py")).is_file()
