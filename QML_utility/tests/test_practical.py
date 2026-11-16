import copy
import csv
import importlib
import json
from pathlib import Path
import joblib
import numpy as np
import pytest
from QML_utility.models.portable_contract import local_features
from QML_utility.models.estimators import QuantumEstimator,ResourceLimit,bounded_angles
from QML_utility.project_data import load_data,split_indices
from QML_utility.recommend import recommend
from QML_utility.run_project import run
from QML_utility.audit_project import audit

ROOT = Path(__file__).resolve().parents[1]


def project(tmp_path,task='classification'):
    source = tmp_path/'data.csv'
    with source.open('w',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id','x0','x1','target','group','time'])
        for i in range(60):
            writer.writerow([i,np.sin(i),i/59.,str(i%3) if task=='classification' else 2*np.sin(i),
                             str(i//5),'2025-01-01T'+f'{i//60:02}:{i%60:02}:00'])
    return dict(data_kind='classical',modality='numeric_table',task=task,objective='iid_'+task,
                capabilities={},allow_exploratory=True,framework='qiskit',seed=7,
                methods=['quantum_features','projected_kernel','fidelity_kernel','trainable_local'],
                quantum=dict(qubits=1,layers=1,steps=1),budget=dict(max_rows=512,state_preparations_per_trial=10000,seconds_per_operation=120),
                data=dict(path=str(source),features=['x0','x1'],target='target',id_column='id',split=dict(kind='iid')))


def test_portfolio_is_separate_from_publication_matching_and_keeps_companions():
    profile = dict(data_kind='classical',modality='numeric_table',task='classification',objective='iid_classification',capabilities={})
    result = recommend(profile)
    assert len(result['method_candidates']) == 4 and not result['published_case_matches']
    assert not result['guaranteed_better'] and not result['executed']
    assert all(not m['verified_advantage_for_this_project'] for m in result['method_candidates'])
    companions = json.loads((ROOT/'companion_map.json').read_text())['companions']
    assert {'near_identity_encoding','natural_gradient','local_loss','kipu_offline_surrogate','classical_pretraining'} <= {c['id'] for c in companions}
    assert all((ROOT/c['guide']).is_file() and c['obligation'] for c in companions)
    profile['modality'] = 'graph_collection'
    assert not recommend(profile)['method_candidates']
    profile['data_kind'] = 'quantum'
    assert recommend(profile)['executable_route'] is None


def test_portfolio_responds_to_gradient_memory_and_deployment_constraints():
    profile = dict(data_kind='classical',modality='numeric_table',task='classification',objective='iid_classification',capabilities={},
                   constraints=dict(n_samples=100),preferences=dict(train_circuit=False,classical_only_inference=True),
                   budget=dict(max_gram_bytes=1000,state_preparations_per_trial=1000))
    result = recommend(profile)
    assert [m['id'] for m in result['method_candidates']] == ['quantum_features']
    assert len(result['withheld_candidates']) == 3
    assert result['required_companions'] == ['kipu_offline_surrogate']
    assert result['method_candidates'][0]['deployment_status'] == 'requires_offline_distillation_adapter'
    assert result['method_candidates'][0]['estimated_state_preparations_per_trial'] == 100
    profile['preferences']['train_circuit'] = 'no'
    with pytest.raises(ValueError):recommend(profile)


@pytest.mark.parametrize('sdk',['qiskit','pennylane','cirq'])
def test_native_half_angles_local_y_sign_and_gate_gradient(sdk):
    pytest.importorskip(sdk)
    native = importlib.import_module('QML_utility.models.portable_'+sdk)
    theta = np.array([[[.7,-.4]]])
    state = native.state([0],theta)
    assert np.allclose(local_features(state),[np.cos(.7),np.sin(.7)*np.cos(-.4),np.sin(.7)*np.sin(-.4)])
    x = np.array([.2,-.4,.7])
    for index in range(theta.size):
        plus,minus = theta.copy(),theta.copy()
        plus.flat[index] += np.pi/2;minus.flat[index] -= np.pi/2
        shift = (local_features(native.state(x,plus))-local_features(native.state(x,minus)))/2
        plus,minus = theta.copy(),theta.copy()
        plus.flat[index] += 1e-5;minus.flat[index] -= 1e-5
        finite = (local_features(native.state(x,plus))-local_features(native.state(x,minus)))/2e-5
        assert np.allclose(shift,finite,atol=1e-8)


def test_all_coordinates_and_independent_three_sdk_state_order():
    for sdk in ('qiskit','pennylane','cirq'):
        pytest.importorskip(sdk)
    rng = np.random.default_rng(11)
    x,theta = rng.normal(size=7),rng.normal(size=(2,3,2))
    states = [importlib.import_module('QML_utility.models.portable_'+s).state(x,theta) for s in ('qiskit','pennylane','cirq')]
    assert all(np.allclose(states[0],state,atol=1e-8) for state in states[1:])
    changed = x.copy();changed[-1] += .3
    native = importlib.import_module('QML_utility.models.portable_qiskit')
    assert abs(np.vdot(states[0],native.state(changed,theta))) < .9999


@pytest.mark.parametrize('task',['classification','regression'])
def test_joint_loss_gradient_uses_chain_rule(task):
    pytest.importorskip('qiskit')
    x = np.array([[-1.],[-.2],[.4],[1.]])
    y = np.array(['a','a','b','b']) if task=='classification' else np.array([-1.,-.2,.4,1.])
    model = QuantumEstimator('trainable_local',task=task,qubits=1,steps=1).fit(x,y)
    angles = bounded_angles(model.preprocessor_.transform(x))
    targets = np.eye(2)[model.encoder_.transform(y)] if task=='classification' else ((y-model.target_mean_)/model.target_scale_)[:,None]
    parameters = np.concatenate([model.theta_.ravel(),model.weights_.ravel(),model.bias_])
    loss,gradient = model._loss_gradient(angles,targets,parameters)
    numerical = []
    for i in range(len(parameters)):
        plus,minus = parameters.copy(),parameters.copy();plus[i]+=1e-5;minus[i]-=1e-5
        numerical.append((model._loss_gradient(angles,targets,plus)[0]-model._loss_gradient(angles,targets,minus)[0])/2e-5)
    assert np.isfinite(loss) and np.allclose(gradient,numerical,atol=2e-7)


def test_train_only_preprocessing_serialization_and_budget(tmp_path):
    pytest.importorskip('qiskit')
    x = np.array([[1.,np.nan],[2.,3.],[3.,4.],[4.,5.]])
    model = QuantumEstimator(qubits=1).fit(x,['a','a','b','b'])
    means = model.preprocessor_.steps[-1][1].mean_.copy()
    predictions = model.predict([[1e8,3.]])
    assert np.allclose(means,[2.5,4.]) and np.array_equal(model.preprocessor_.steps[-1][1].mean_,means)
    artifact = tmp_path/'model.joblib';joblib.dump(model,artifact)
    restored = joblib.load(artifact)
    assert np.array_equal(restored.predict([[1e8,3.]]),predictions)
    assert restored.resource_['measurement_shots'] == 0 and restored.resource_['qpu_jobs'] == 0
    with pytest.raises(ResourceLimit):
        QuantumEstimator(qubits=1,max_state_preparations=1).fit(x,['a','a','b','b'])


def test_group_and_temporal_splits_do_not_mix_entities_or_future(tmp_path):
    config = project(tmp_path)
    x,y,rows,ids,_ = load_data(config)
    splits = split_indices(y,rows,dict(kind='group',group_column='group'),7,'classification')
    groups = [{rows[i]['group'] for i in splits[k]} for k in ('train','validation','test')]
    assert all(not groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2)))
    splits = split_indices(y,rows,dict(kind='chronological',time_column='time',gap_rows=2),7,'classification')
    assert max(splits['train'])+2 < min(splits['validation'])
    assert max(splits['validation'])+2 < min(splits['test'])
    bad = copy.deepcopy(config);bad['data']['features'].append('target')
    with pytest.raises(ValueError):load_data(bad)
    bad = copy.deepcopy(config);bad['budget']['max_rows']=30
    with pytest.raises(ValueError):load_data(bad)
    duplicate_times = copy.deepcopy(rows);duplicate_times[1]['time']=duplicate_times[0]['time']
    with pytest.raises(ValueError):
        split_indices(y,duplicate_times,dict(kind='chronological',time_column='time'),7,'classification')


@pytest.mark.parametrize('task',['classification','regression'])
@pytest.mark.parametrize('framework',['qiskit','pennylane','cirq'])
def test_real_project_run_all_families_and_reload_selected_predictor(tmp_path,task,framework):
    pytest.importorskip(framework)
    config = project(tmp_path,task)
    config['framework'] = framework
    report = run(config,tmp_path/'run')
    assert report['status'] == 'completed'
    assert len(report['models']) == 9 and len(report['trials']) == 25
    assert all(t['status']=='completed' for t in report['trials'])
    assert not report['paper_replication'] and not report['advantage_established']
    selection = json.loads((tmp_path/'run/selection.json').read_text())
    assert selection['validation_champion'] == report['validation_champion']
    assert audit(tmp_path/'run',config)['evaluated_families'] == 9
    for method in ('quantum_features','projected_kernel','fidelity_kernel','trainable_local'):
        result = next(m for m in report['models'] if m['method']==method)
        artifact = joblib.load(tmp_path/'run'/result['artifact'])
        x,y,rows,ids,_ = load_data(config)
        test = [ids.index(i) for i in result['test_ids']]
        assert np.allclose(artifact['model'].predict(x[test]).astype(float),np.array(result['test_predictions']).astype(float))
    with pytest.raises(ValueError):run(config,tmp_path/'run')
    damaged = json.loads((tmp_path/'run/report.json').read_text())
    damaged['models'][0]['test_score'] += .1
    (tmp_path/'run/report.json').write_text(json.dumps(damaged))
    with pytest.raises(ValueError,match='Test metric mismatch'):
        audit(tmp_path/'run',config)


def test_failed_trials_stay_visible_in_partial_report(tmp_path):
    pytest.importorskip('qiskit')
    config = project(tmp_path)
    config['methods'] = ['quantum_features']
    config['budget']['state_preparations_per_trial'] = 1
    report = run(config,tmp_path/'partial')
    assert report['status'] == 'partial_with_failures'
    failed = [t for t in report['trials'] if t['method']=='quantum_features']
    assert len(failed)==3 and all(t['status']=='failed' and t['resources']['state_preparations']==1 for t in failed)
    assert not report['advantage_established']
