import importlib
import json
from pathlib import Path
import tomllib
import numpy as np
import pytest
from QML_utility.strategies import essentials,resolve,check
from QML_utility.setup_project import setup
from QML_utility.models.estimators import QuantumEstimator,bounded_angles


def test_one_distribution_and_default_test_discovery():
    root=Path(__file__).resolve().parents[2]
    metadata=tomllib.loads((root/'pyproject.toml').read_text())
    assert {'qml-utility','qml-tricks'} <= metadata['project']['scripts'].keys()
    assert 'QML_utility/tests' in metadata['tool']['pytest']['ini_options']['testpaths']
    assert metadata['tool']['setuptools']['package-dir']['QML_utility']=='QML_utility'
    assert not (root/'QML_utility/pyproject.toml').exists()


@pytest.mark.parametrize('framework',['qiskit','pennylane','cirq'])
@pytest.mark.parametrize('name',['ngd','local_cost','near_identity_initialization','near_identity_encoding'])
def test_utility_executes_canonical_essential_strategy_and_control(framework,name):
    pytest.importorskip(framework)
    result=check(name,framework,compare=True)
    assert result['strategy']==resolve(name)
    assert len(result['results'])==2
    for row in result['results']:
        assert row['strategy']==resolve(name) and row['framework']==framework
        assert np.isfinite(row['loss']) and len(row['training_trace'])>=2
    assert result['results'][0]['configuration']['variant']=='strategy'
    assert result['results'][1]['configuration']['variant']=='baseline'


def test_setup_includes_published_cases_and_essentials_without_fabricating_capabilities(tmp_path):
    result=setup(tmp_path/'case',case_id='continual_plasticity')
    assert len(result['published_cases'])==10
    assert result['selected_case']['id']=='continual_plasticity'
    assert not any(result['project']['capabilities'].values())
    assert not result['portfolio']['published_case_matches']
    assert (tmp_path/'case/case_protocol.json').is_file()
    assert {row['id'] for row in result['training_essentials']}=={'natural_gradient','local_loss','identity_block_init','near_identity_encoding'}
    assert resolve('near_identity_initialization') != resolve('near_identity_encoding')


def test_default_setup_creates_qng_project_and_reuses_canonical_catalog(tmp_path):
    result=setup(tmp_path/'example')
    assert result['project']['quantum']['optimizer']=='qng'
    assert Path(result['project']['data']['path']).is_file()
    from qml_tricks.core import catalog
    source={row['id']:row for row in catalog()}
    assert all(row['canonical']==source[row['id']] for row in essentials())
    config=json.loads((tmp_path/'example/project.json').read_text())
    from QML_utility.run_project import candidates
    candidate=next(candidates('trainable_local','classification',config))[1]
    assert candidate.optimizer=='qng' and candidate.metric_regularization==.001


@pytest.mark.parametrize('framework',['qiskit','pennylane','cirq'])
@pytest.mark.parametrize('task',['classification','regression'])
def test_supervised_qng_matches_analytic_single_qubit_geometry(framework,task):
    pytest.importorskip(framework)
    x=np.array([[-.7,.1],[-.2,.4],[.3,-.2],[.8,.5]])
    y=np.array(['a','a','b','b']) if task=='classification' else x[:,0]+.2*x[:,1]
    model=QuantumEstimator('trainable_local',task=task,framework=framework,qubits=1,steps=1,
                           optimizer='qng',learning_rate=.005,metric_regularization=.03).fit(x,y)
    # P=2: N*(2P+1) metric preparations, plus the separately counted gradient.
    assert model.resource_['metric_state_preparations']==20
    assert model.resource_['state_preparations']==44
    assert model.trace_[0]['optimizer']=='qng'
    theta=np.random.default_rng(7).normal(0,.01,(1,1,2))
    weights=np.random.default_rng(8).normal(0,.1,(3,2 if task=='classification' else 1))
    bias=np.zeros(weights.shape[1])
    angles=bounded_angles(model.preprocessor_.transform(x))
    native=importlib.import_module('QML_utility.models.portable_'+framework)
    a,b=theta.ravel()
    ry=np.array([[np.cos(a/2),-np.sin(a/2)],[np.sin(a/2),np.cos(a/2)]])
    rz=np.diag(np.exp(np.array([-1j,1j])*b/2))
    pauli_y=np.array([[0,-1j],[1j,0]]);pauli_z=np.diag([1,-1])
    metric=np.zeros((2,2))
    for row in angles:
        initial=native.state(row,np.zeros((1,1,2)))
        psi=rz@ry@initial
        derivatives=np.column_stack([rz@(-.5j*pauli_y)@ry@initial,(-.5j*pauli_z)@psi])
        overlaps=derivatives.conj().T@psi
        metric+=np.real(derivatives.conj().T@derivatives-np.outer(overlaps,overlaps.conj()))/len(x)
    target=np.eye(2)[model.encoder_.transform(y)] if task=='classification' else ((y-model.target_mean_)/model.target_scale_)[:,None]
    parameters=np.concatenate([theta.ravel(),weights.ravel(),bias])
    gradient=model._loss_gradient(angles,target,parameters)[1][:2]
    expected=theta.ravel()-.005*np.linalg.solve(metric+.03*np.eye(2),gradient)
    assert np.allclose(model.theta_.ravel(),expected,atol=1e-8)


def test_fixed_map_does_not_silently_claim_circuit_qng():
    with pytest.raises(ValueError,match='only to trainable_local'):
        QuantumEstimator(optimizer='qng').fit(np.arange(8).reshape(4,2),[0,0,1,1])


def test_encoding_setup_preserves_same_scale_classical_control():
    from QML_utility.run_project import candidates
    config=dict(quantum=dict(encoding_scale_candidates=[.2,.6,1.2]))
    quantum=[settings['scale'] for settings,_ in candidates('quantum_features','classification',config)]
    classical=[settings['encoding_scale'] for settings,_ in candidates('fourier_linear','classification',config)]
    assert quantum==classical==[.2,.6,1.2]
