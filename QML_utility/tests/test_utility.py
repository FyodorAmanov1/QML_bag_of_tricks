import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from QML_utility.select_case import select
from QML_utility.benchmarks.late_query import audit, balanced_accuracy
from QML_utility.models.numerics import amplitude_input, su4_gate, pauli_state, bell_signs, last_z

ROOT = Path(__file__).resolve().parents[1]


def test_no_generic_csv_fallback_or_invented_guarantee():
    result = select(json.loads((ROOT/'examples/generic_csv.json').read_text()))
    assert not result['matches']
    result = select(json.loads((ROOT/'examples/continual_project.json').read_text()))
    assert [x['id'] for x in result['matches']] == ['continual_plasticity']
    assert not result['matches'][0]['guaranteed_better']
    profile = json.loads((ROOT/'examples/late_query_project.json').read_text())
    profile['capabilities']['late_observable_query'] = False
    assert not select(profile)['matches']
    profile['capabilities']['late_observable_query'] = 'yes'
    with pytest.raises(ValueError):
        select(profile)


def test_evidence_and_protocols_do_not_promote_reference_construction():
    data = json.loads((ROOT/'catalog.json').read_text())
    assert len({x['id'] for x in data['cases']}) == 10
    for case in data['cases']:
        assert (ROOT/case['guide']).is_file()
        assert case['publication']['doi'] and case['publication']['locator']
        assert case['data_conditions'] and case['necessary_controls'] and case['blockers']
        assert case['locally_reproduced_paper_win'] is False
    for record in json.loads((ROOT/'upstream_manifest.json').read_text())['sources'].values():
        assert len(record['commit']) == 40 and not record['executed_here']
        assert all(len(f['sha256']) == 64 for f in record['files'])


def test_amplitude_loading_does_not_drop_coordinates():
    x = amplitude_input(np.arange(1, 785))
    assert x.shape == (1024,) and np.all(x[784:] == 0)
    assert np.isclose(np.linalg.norm(x), 1)
    with pytest.raises(ValueError):
        amplitude_input(np.zeros(784))


def test_su4_half_angle_convention():
    theta = np.zeros(15)
    theta[0] = .4  # I tensor X
    expected = np.array([np.cos(.2), -1j*np.sin(.2), 0, 0])
    assert np.allclose(su4_gate(theta)[:, 0], expected)


def test_noncommuting_exponential_coordinates_cannot_be_wrapped_individually():
    theta = np.zeros(15);theta[0]=.4;theta[2]=.7
    wrapped = theta.copy();wrapped[0]+=2*np.pi
    a, b = su4_gate(theta)[:,0], su4_gate(wrapped)[:,0]
    assert abs(np.vdot(a,b)) < .999


@pytest.mark.parametrize('sdk', ['qiskit', 'pennylane', 'cirq'])
def test_native_state_order_and_readout_derivative(sdk):
    pytest.importorskip(sdk)
    native = importlib.import_module('QML_utility.models.'+sdk+'_impl')
    x = np.array([1, 0, 0, 0], complex)
    theta = np.zeros((1, 1, 15));theta[0, 0, 0] = .4
    out = native.continual_state(x, theta)
    assert np.allclose(out, [np.cos(.2), -1j*np.sin(.2), 0, 0])
    h = 1e-5;plus=theta.copy();minus=theta.copy();plus[0,0,0]+=h;minus[0,0,0]-=h
    derivative = (last_z(native.continual_state(x, plus))-last_z(native.continual_state(x, minus)))/(2*h)
    assert np.isclose(derivative, -np.sin(.4), atol=1e-7)


def test_independent_native_brickwall_construction():
    for sdk in ('qiskit', 'pennylane', 'cirq'):
        pytest.importorskip(sdk)
    rng = np.random.default_rng(19)
    x = amplitude_input(rng.normal(size=13)+1j*rng.normal(size=13))
    theta = rng.normal(scale=.2, size=(2, 3, 15))
    states = [importlib.import_module('QML_utility.models.'+s+'_impl').continual_state(x,theta)
              for s in ('qiskit', 'pennylane', 'cirq')]
    assert all(np.allclose(states[0], state, atol=1e-8) for state in states[1:])
    assert np.isclose(np.linalg.norm(states[0]), 1)


@pytest.mark.parametrize('sdk', ['qiskit', 'pennylane', 'cirq'])
def test_native_bell_order_yy_sign_and_squared_signal(sdk):
    pytest.importorskip(sdk)
    native = importlib.import_module('QML_utility.models.'+sdk+'_impl')
    for word in ('X', 'Y', 'Z', 'XY', 'YZ', 'ZYX'):
        for amplitude in (-.8, 0, .6):
            probabilities = native.bell_probabilities(pauli_state(word, amplitude))
            assert np.min(probabilities) > -1e-9 and np.isclose(probabilities.sum(), 1)
            assert np.isclose(probabilities @ bell_signs(word), amplitude**2)


def test_restricted_gap_and_query_timing_ablation():
    row = audit(n=8, copies=512, states=4000, seed=17)
    assert row['scores']['two_copy_bell'] > .99
    assert row['scores']['optimal_random_local_records'] < .56
    assert row['scores']['query_known_early'] > .99
    assert row['configuration']['bell_rounds']*2 == row['configuration']['state_copies_per_example']
    for key, predictions in row['records']['predictions'].items():
        assert row['scores'][key] == balanced_accuracy(np.array(row['records']['labels']), np.array(predictions))
    assert audit(n=2, states=4000, seed=17)['scores']['optimal_random_local_records'] > .98


def test_readout_noise_and_invalid_copy_count():
    assert audit(n=8, seed=17, readout_error=.49)['scores']['two_copy_bell'] < .56
    with pytest.raises(ValueError):
        audit(copies=511)
