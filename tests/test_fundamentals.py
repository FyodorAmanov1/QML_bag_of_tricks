"""Physics and optimizer checks independent of the generated implementation."""
import numpy as np
import pytest
from qml_tricks.core import RunConfig, module, gradient, experiment
from qml_tricks.fixtures import initial_parameters, loss_from_probabilities
from qml_tricks.geometry import fubini_study_metric, natural_gradient_direction
from test_recipes import SDKS


@pytest.mark.parametrize("sdk", SDKS)
def test_near_identity_encoder_derivative(sdk):
    cfg = RunConfig(features=(-1., -.2, .3, 1.), encoding_scale=.12, steps=0)
    theta = initial_parameters("near_identity_encoding", 4, 2, 7, "strategy")
    def objective(t):
        return loss_from_probabilities(module("near_identity_encoding", sdk).probabilities(t, cfg), "near_identity_encoding", 4, "strategy")
    expected = -np.cos(.12*np.array(cfg.features))/4
    np.testing.assert_allclose(gradient(objective, theta), expected, atol=1e-9)
    assert np.min(np.abs(expected)) >= np.cos(.12)/4 - 1e-12
    zero = RunConfig(encoding_scale=0, steps=0)
    expected_state = module("local_loss", sdk).statevector(theta, RunConfig(steps=0))
    np.testing.assert_allclose(module("near_identity_encoding", sdk).statevector(theta, zero), expected_state, atol=1e-10)


@pytest.mark.parametrize("sdk", SDKS)
def test_identity_blocks_cancel_but_independent_copy_has_signal(sdk):
    cfg = RunConfig(steps=0)
    theta = initial_parameters("identity_block_init", 4, 2, 7, "strategy")
    impl = module("identity_block_init", sdk)
    state = impl.statevector(theta, cfg)
    expected = np.zeros(16, complex); expected[[0,8]] = 1/np.sqrt(2)
    np.testing.assert_allclose(state, expected, atol=1e-10)
    def objective(t):
        return loss_from_probabilities(impl.probabilities(t, cfg), "identity_block_init", 4, "strategy")
    actual = gradient(objective, theta)[4]
    assert actual == pytest.approx(-np.cos(theta[0])*np.cos(theta[1]), abs=1e-9)
    # Explicitly tying copies at every evaluation destroys this signal.
    delta = np.zeros_like(theta); delta[4] = .2; delta[8] = .2
    assert objective(theta + delta) == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("sdk", SDKS)
def test_fs_metric_against_product_state_geometry(sdk):
    cfg = RunConfig(n_qubits=2, steps=0)
    theta = np.array([.8, 1.1, .3, -.6])
    impl = module("natural_gradient", sdk)
    metric = fubini_study_metric(lambda t: impl.statevector(t, cfg), theta)
    expected = np.diag([.25, .25, np.sin(.8)**2/4, np.sin(1.1)**2/4])
    np.testing.assert_allclose(metric, expected, atol=1e-9)
    # The projection must remove a parameter-dependent global phase.
    phased = fubini_study_metric(lambda t: np.exp(1j*3*t[0])*impl.statevector(t, cfg), theta)
    np.testing.assert_allclose(phased, expected, atol=1e-8)
    direction = natural_gradient_direction(metric, np.ones(4), 1e-3)
    np.testing.assert_allclose(direction, 1/(np.diag(expected)+1e-3), rtol=1e-8)


@pytest.mark.parametrize("sdk", SDKS)
def test_qng_update_and_metric_accounting(sdk):
    cfg = RunConfig(n_qubits=2, steps=1, initializations=2)
    impl = module("natural_gradient", sdk)
    theta = initial_parameters("natural_gradient", 2, 2, 7, "strategy")
    def cost(t):
        return loss_from_probabilities(impl.probabilities(t, cfg), "natural_gradient", 2, "strategy")
    grad = gradient(cost, theta)
    metric = fubini_study_metric(lambda t: impl.statevector(t, cfg), theta)
    expected_theta = theta - cfg.learning_rate * natural_gradient_direction(metric, grad)
    result = experiment("natural_gradient", sdk, cfg)
    assert result.loss == pytest.approx(cost(expected_theta), abs=1e-10)
    assert result.resources["metric_state_evaluations"] == 9
    assert not result.gradient_statistics["optimizer"]["barren_plateau_avoidance_guarantee"]
    baseline = experiment("natural_gradient", sdk, RunConfig(n_qubits=2, steps=1, variant="baseline", initializations=2))
    assert baseline.training_trace[0] == pytest.approx(result.training_trace[0])
    assert baseline.resources["metric_state_evaluations"] == 0


def test_regularization_handles_singular_metric():
    direction = natural_gradient_direction(np.diag([.25, 0.]), np.array([.2, 0.]), 1e-3)
    assert np.all(np.isfinite(direction))
    assert direction[1] == 0


@pytest.mark.parametrize("cfg", [RunConfig(features=(2.,0.,0.,0.)), RunConfig(encoding_scale=float("nan")), RunConfig(metric_regularization=0)])
def test_fundamental_configuration_bounds(cfg):
    with pytest.raises(ValueError): cfg.validate("near_identity_encoding")
