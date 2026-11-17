"""Contracts that prevent publication evidence becoming an invented local win."""
import hashlib
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from QML_utility.plan_project import build_plan
from QML_utility.select_case import select
from QML_utility.models.projected_attention import gaussian_projected_attention
from QML_utility.models.pauli_channel import (
    channel_spec, eigenvalues_from_error_probabilities,
    eigenvalues_from_bell_probabilities, remove_known_memory_attenuation)

ROOT = Path(__file__).resolve().parents[1]
NEW_CASES = {'molecular_reservoir', 'quantum_self_attention',
             'photonic_interference_kernel', 'volatility_reservoir', 'pauli_channel_learning'}


@pytest.mark.parametrize('case_id', sorted(NEW_CASES))
def test_new_case_matches_only_its_declared_protocol_and_is_not_local_win(case_id):
    case = next(c for c in json.loads((ROOT/'catalog.json').read_text())['cases'] if c['id'] == case_id)
    assert case['publication']['peer_reviewed'] is True
    assert not case['locally_reproduced_paper_win']
    protocol = json.loads((ROOT/case['protocol']).read_text())
    assert protocol['case_id'] == case_id
    assert protocol['status'] == 'paper_scoped_replication_plan_not_executed_here'
    assert protocol['must_recover'] and protocol['stop_conditions']
    profile = dict(data_kind=case['data_kind'], objective=case['objective'],
                   capabilities=dict.fromkeys(case['required_capabilities'], True))
    result = select(profile)
    assert [match['id'] for match in result['matches']] == [case_id]
    profile['capabilities'][case['required_capabilities'][0]] = False
    result = select(profile)
    assert not result['matches']
    assert 'not evidence' in result['interpretation']
    assert all(item['assessment_scope'] == 'protocol_requirements_not_method_wide_usefulness' for item in result['rejected'])


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_bounded_outcome_preserves_original_data_and_allows_prospective_followup(newline):
    ledger = json.loads((ROOT/'experiment_outcomes.json').read_text())
    for episode in ledger['episodes']:
        assert episode['inference_scope'] == 'configuration_and_dataset_specific_no_gain'
        assert episode['follow_up_hypotheses'] and episode['does_not_establish']
        assert episode['allowed_action'] == 'validation_only_prespecified_followup'
        assert (ROOT/episode['retest_protocol']).is_file()
        if 'source_sha256' in episode:
            lf = (ROOT/episode['source']).read_bytes().replace(b'\r\n', b'\n')
            checkout = lf.replace(b'\n', newline)
            assert hashlib.sha256(checkout.replace(b'\r\n', b'\n')).hexdigest() == episode['source_sha256_lf']
            # Preserve and verify the original snapshot, accepting only EOL spelling.
            assert episode['source_sha256'] in {
                hashlib.sha256(contents).hexdigest() for contents in (lf, lf.replace(b'\n', b'\r\n'))}
    profile = json.loads((ROOT/'examples/wine_followup.json').read_text())
    plan = build_plan(profile)
    assert [e['id'] for e in plan['selection']['contextual_outcomes']] == ['pca2_wine_binary']
    assert plan['prospective_followup'] and not plan['case_plans']
    assert not plan['executed'] and not plan['guaranteed_better']
    assert plan['missing_project_details'] == ['compute_budget']
    assert plan['status'] == 'draft_requires_project_details_or_matched_evidence'
    profile['project']['compute_budget'] = 'CPU, 16 GB, 1 hour per model; 12 validation candidates; 0 QPU shots'
    assert build_plan(profile)['status'] == 'draft_ready_for_agent_review'
    profile['intent'] = 'promote_without_evaluation'
    with pytest.raises(ValueError):
        build_plan(profile)


def test_plan_retains_paper_blockers_even_when_project_details_are_supplied():
    profile = json.loads((ROOT/'examples/molecular_project.json').read_text())
    plan = build_plan(profile)
    assert set(plan['missing_project_details']) == {'data_provenance', 'compute_budget'}
    assert plan['case_plans'][0]['case_id'] == 'molecular_reservoir'
    assert plan['case_plans'][0]['blockers']
    assert plan['case_plans'][0]['replication_protocol']['must_recover']
    assert plan['prospective_followup'] is None


def test_attention_matches_gaussian_rule_and_ignores_masked_values():
    q, k = np.array([-.7,.2]), np.array([-.8,0,.5])
    v = np.array([[1.,2.],[3.,4.],[1e8,-1e8]])
    out, weights = gaussian_projected_attention(q,k,v,[True,True,False])
    expected = np.exp(-(q[:,None]-k[None,:2])**2)
    expected /= expected.sum(axis=1,keepdims=True)
    assert np.allclose(weights[:,:2],expected) and np.all(weights[:,2] == 0)
    assert np.allclose(out,expected@v[:2])
    permutation = [2,0,1]
    permuted, _ = gaussian_projected_attention(q,k[permutation],v[permutation],np.array([True,True,False])[permutation])
    assert np.allclose(out,permuted)
    with pytest.raises(ValueError):
        gaussian_projected_attention(q,k,v,[False,False,False])
    with pytest.raises(ValueError):
        gaussian_projected_attention([1.1],k,v)


@pytest.mark.parametrize('sdk', ['qiskit', 'pennylane', 'cirq'])
def test_channel_bell_mapping_and_signed_eigenvalues(sdk):
    pytest.importorskip(sdk)
    native = importlib.import_module('QML_utility.models.'+sdk+'_impl')
    # I,X,Y,Z errors decode to 00,01,11,10, rather than Pauli index order.
    for error, index in enumerate([0,1,3,2]):
        p = np.eye(4)[error]
        observed = native.pauli_channel_probabilities(p)
        assert np.allclose(observed,np.eye(4)[index],atol=1e-9)
        words, expected = eigenvalues_from_error_probabilities(p)
        decoded_words, decoded = eigenvalues_from_bell_probabilities(np.maximum(observed,0)/observed.sum())
        assert words == decoded_words and np.allclose(decoded,expected,atol=1e-9)
        if error:
            assert np.min(decoded) < -.999  # not the nonnegative state-square identity
    rng = np.random.default_rng(23)
    p = rng.dirichlet(np.ones(16))
    observed = native.pauli_channel_probabilities(p)
    assert np.min(observed) >= -1e-9 and np.isclose(observed.sum(),1)
    assert np.allclose(eigenvalues_from_bell_probabilities(np.maximum(observed,0)/observed.sum())[1],
                       eigenvalues_from_error_probabilities(p)[1],atol=1e-9)


def test_channel_calibration_is_guarded_and_does_not_silently_clip():
    _, eigenvalues = eigenvalues_from_error_probabilities([.7,.1,.15,.05])
    assert np.allclose(eigenvalues,[1,.6,.7,.5])
    auxiliary = np.array([1,.8,.6,.7])
    assert np.allclose(remove_known_memory_attenuation(eigenvalues*auxiliary,auxiliary),eigenvalues)
    assert remove_known_memory_attenuation([.9],[.8])[0] > 1
    with pytest.raises(ValueError):
        remove_known_memory_attenuation([.2],[.01])
    for invalid in ([1,0,0], [.9,0,0,0], [1.1,-.1,0,0], [np.nan,0,0,0]):
        with pytest.raises(ValueError):
            channel_spec(invalid)
