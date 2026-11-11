"""Independent physical identities, access contracts and selection checks."""
from copy import deepcopy
from importlib import import_module
import json

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

from qml_tricks.learning.experimental_learning import (
    acquire_bell, acquire_local, bell_distribution, bell_signs, local_counts,
    local_log_likelihood_ratio, pauli_state,
)
from qml_tricks.learning.opportunities import (
    PROTOCOL, classical_champion, fourier_features, pauli_case, ramsey_case, run_opportunities,
)


@pytest.mark.parametrize('sdk', ['qiskit', 'pennylane', 'cirq'])
def test_native_ramsey_matches_independent_fourier_identity(sdk):
    native = import_module(f'qml_tricks.learning.application_backends.{sdk}')
    phases = np.array([-4.1, -.7, 0., .2, 2.3, 8.9])
    np.testing.assert_allclose(native.ramsey_features(phases), fourier_features(phases), atol=1e-12)


@pytest.mark.parametrize('sdk', ['qiskit', 'pennylane', 'cirq'])
@pytest.mark.parametrize('word', ['X', 'Y', 'Z', 'XY', 'YZ', 'IX'])
def test_native_bell_distribution_including_y_sign_and_bit_order(sdk, word):
    rho = pauli_state(word, -.63)
    native = import_module(f'qml_tricks.learning.application_backends.{sdk}')
    probabilities = native.bell_probabilities(rho)
    np.testing.assert_allclose(probabilities, bell_distribution(word, -.63), atol=1e-10)
    np.testing.assert_allclose(np.trace(rho), 1.)
    np.testing.assert_allclose(np.trace(rho@rho), (1+.63**2)/2**len(word))
    assert np.min(np.linalg.eigvalsh(rho)) >= -1e-12


def test_bell_sampler_matches_native_born_law_and_readout_attenuation():
    word = 'XY'; count = 80000; amplitude = .7
    clean = acquire_bell(word, amplitude, count, np.random.default_rng(11))
    outcome = clean@np.array([8, 4, 2, 1])
    empirical = np.bincount(outcome, minlength=16)/(count//2)
    np.testing.assert_allclose(empirical, bell_distribution(word, amplitude), atol=.006)
    noisy = acquire_bell(word, amplitude, count, np.random.default_rng(12), .1)
    # X uses one output bit, Y two: three independent readout flips.
    assert abs(bell_signs(noisy, word).mean()-amplitude**2*.8**3) < .02


def test_acquisition_settings_do_not_depend_on_hidden_state_or_late_query():
    a = acquire_local('XYZ', .8, 400, np.random.default_rng(8))
    b = acquire_local('ZZX', 0., 400, np.random.default_rng(8))
    np.testing.assert_array_equal(a[0], b[0])
    for query in ('XYZ', 'ZZX', 'IYX'):
        plus, minus = local_counts(*a, query)
        assert 0 <= plus+minus <= 400


def test_local_likelihood_ratio_matches_full_record_probability():
    # Enumerate all X/Z local settings and signs for a two-qubit state.
    # Nonmatching settings carry no a dependence and cancel in the ratio.
    rho0 = pauli_state('XY', 0.)
    rhop = pauli_state('XY', .8); rhom = pauli_state('XY', -.8)
    from qml_tricks.learning.experimental_learning import PAULIS
    bases = np.array([[0, 1], [2, 0], [0, 1]])
    outcomes = np.array([[1, 1], [-1, 1], [1, -1]])
    likelihood = []
    for rho in (rho0, rhop, rhom):
        value = 1.
        for axes, signs in zip(bases, outcomes):
            projector = np.array([[1.]], complex)
            for axis, sign in zip(axes, signs):
                projector = np.kron(projector, (np.eye(2)+sign*PAULIS['XYZ'[axis]])/2)
            value *= np.trace(rho@projector).real
        likelihood.append(value)
    plus, minus = local_counts(bases, outcomes, 'XY')
    expected = np.log((likelihood[1]+likelihood[2])/2/likelihood[0])
    np.testing.assert_allclose(local_log_likelihood_ratio(plus, minus, .8), expected)
    assert local_log_likelihood_ratio(0, 0, .8) == 0.


def test_exact_quantum_feature_gain_is_reproduced_by_classical_fourier_control():
    config = json.loads(PROTOCOL.read_text())['ramsey']
    config.update(validation_size=12, test_size=20)
    case = ramsey_case(config, 91, 12, None, 'qiskit', ['fourier_ridge', 'quantum_ramsey_ridge'])
    a, b = case['models']
    np.testing.assert_allclose(a['test_predictions'], b['test_predictions'], atol=1e-10)
    assert case['native_fourier_max_error'] < 1e-12
    for row in case['models']:
        assert row['test']['rmse'] == np.sqrt(mean_squared_error(row['test_targets'], row['test_predictions']))
        selected = min(row['trials'], key=lambda x:x['validation']['rmse'])
        assert selected['trial'] == row['selected_trial']


def test_pauli_copy_accounting_metrics_and_train_only_refit():
    config = json.loads(PROTOCOL.read_text())['pauli']
    config.update(train_size=12, validation_size=8, test_size=16)
    case = pauli_case(config, 71, 4, 128, .01, ['local_logistic_l2', 'local_bayes', 'bell_logistic_l2', 'query_known_bayes'])
    train, val, test = [case['split_indices'][k] for k in ('train', 'validation', 'test')]
    assert not set(train)&set(val) and not set(val)&set(test) and not set(train)&set(test)
    assert len(set(case['measurement_record_sha256'])) == 36
    assert case['acquisition']['total_copies_per_protocol'] == 36*128
    assert case['acquisition']['bell_pairs_per_state'] == 64
    for row in case['models']:
        assert row['test']['balanced_accuracy'] == balanced_accuracy_score(row['test_targets'], row['test_predictions'])
        if row['trials']:
            selected = max(row['trials'], key=lambda x:x['validation']['balanced_accuracy'])
            assert selected['trial'] == row['selected_trial']
            features = np.array(case['bell_features'] if row['model'].startswith('bell') else case['local_features'])
            np.testing.assert_allclose(row['refit_feature_mean'], features[train+val].mean(axis=0))


def test_small_runner_retains_complete_protocol_and_failures(tmp_path):
    protocol = deepcopy(json.loads(PROTOCOL.read_text()))
    protocol['seeds'] = [71]
    protocol['pauli'].update(qubits=[2], state_copies=[32], readout_bit_flip=[0.],
                            train_size=12, validation_size=8, test_size=16,
                            models=['local_bayes', 'bell_logistic_l2'])
    result = run_opportunities(tmp_path/'run.json', protocol=protocol, applications=('pauli',))
    assert len(result['cases']) == 1
    assert len(result['summary']) == 2
    assert not result['paired_comparisons'][0]['observed_gain']  # one seed cannot meet reporting rule
    assert result['native_verification']['bell_probability_max_error'] < 1e-10
    assert json.loads((tmp_path/'run.json').read_text())['protocol'] == protocol


def test_classical_family_selection_ignores_opposing_test_ranking():
    rows = [dict(model='ridge', selected_trial=0,
                 trials=[dict(trial=0, validation=dict(rmse=.1))],
                 test=dict(rmse=10.), test_predictions=[10.], test_targets=[0.]),
            dict(model='rbf_svr', selected_trial=0,
                 trials=[dict(trial=0, validation=dict(rmse=.2))],
                 test=dict(rmse=.01), test_predictions=[.01], test_targets=[0.])]
    chosen = classical_champion(rows, 'regression')
    assert chosen['selected_family'] == 'ridge'
    assert chosen['test']['rmse'] == 10.


@pytest.mark.parametrize('word, amplitude, copies', [('II', .5, 10), ('XY', 1.2, 10), ('XY', .5, 11)])
def test_invalid_physical_acquisitions_rejected(word, amplitude, copies):
    with pytest.raises(ValueError):
        acquire_bell(word, amplitude, copies, np.random.default_rng(1))
