"""Independent learning integration and statistical mechanism tests."""
import json
import numpy as np
import pytest
from qml_tricks.learning.models import QMLModel,MODELS
from qml_tricks.learning.methods import parameter_shift,allocate_shots,mmd_unbiased,pauli_shadows,shadow_expectation,born_mmd_gradient
from qml_tricks.learning.benchmark import benchmark,generative_benchmark
from qml_tricks.learning.demos import run_recipe
from importlib import import_module


@pytest.fixture
def data():
    rng=np.random.default_rng(5);x=rng.normal(size=(24,3));return x,(x[:,0]>.0).astype(int)


def test_native_learning_state_parity():
    states=[import_module(f'qml_tricks.learning.backends.{sdk}').state(np.array([.3,.4,-.2,.1]),np.array([.1,.5]),2,2)
            for sdk in ('qiskit','pennylane','cirq')]
    for s in states:np.testing.assert_allclose(s,states[0],atol=1e-10)


@pytest.mark.parametrize('sdk',['qiskit','pennylane','cirq'])
def test_native_shift_and_shadow(sdk):
    assert run_recipe('parameter_shift',sdk)['maximum_error']<1e-8
    result=run_recipe('classical_shadows',sdk)
    assert abs(result['estimate']-result['exact'])<.2


@pytest.mark.parametrize('method',MODELS)
@pytest.mark.parametrize('task',['classification','regression'])
def test_learning_fit_predict_and_save(method,task,data,tmp_path):
    x,y=data
    if task=='regression':y=x[:,0]*.5+x[:,1]
    model=QMLModel(method,task,steps=1).fit(x[:16],y[:16]);pred=model.predict(x[16:])
    assert np.isfinite(pred).all();assert len(pred)==8
    path=tmp_path/'model.joblib';model.save(path)
    np.testing.assert_allclose(model.load(path).predict(x[16:]),pred)
    assert model.scaler_.mean_==pytest.approx(x[:16].mean(axis=0))
    if method=='kernel_surrogate':
        calls=model.resources_['state_preparations'];model.predict(x[16:]+.02)
        assert model.resources_['state_preparations']==calls
        audit=model.audit_surrogate(x[16:]);assert audit['kernel_rmse']>=0
    if method=='classical_pretraining':assert len(model.predict_classical_control(x[16:]))==8
    if method=='classical_warm_start':assert model.resources_['warm_start_state_preparations']>0


def test_exact_full_landmark_kernel_reconstruction(data):
    x,y=data;model=QMLModel('fidelity_kernel',steps=0).fit(x[:12],y[:12]);tx=model._transform(x[:12])
    c=model._kernel(tx,model.landmarks_);features=c@model.projection_
    np.testing.assert_allclose(features@features.T,model._kernel(tx,tx),atol=1e-8)


def test_shot_allocation_and_mmd():
    counts=allocate_shots([1,.1],[.9,.1],200)
    assert counts.sum()==200 and np.all(counts>0) and counts[0]>counts[1]
    x=np.array([[0,0],[0,0]]);y=np.array([[1,1],[1,1]])
    assert mmd_unbiased(x,x)==pytest.approx(0)
    assert mmd_unbiased(x,y)==pytest.approx(2-2*np.exp(-2))


def test_spsa_is_two_evaluations():
    from qml_tricks.learning.methods import spsa
    calls=[]
    def f(t):calls.append(1);return np.dot(t,t)
    spsa(f,np.ones(20),steps=4)
    assert len(calls)==8


def test_born_gradient_against_exact_finite_difference():
    # Replace sampled averages by large draws of an exactly calculable Bernoulli model.
    rng=np.random.default_rng(7);theta=np.array([.8]);target=np.zeros((100,1),int)
    def sampler(t,count):return rng.binomial(1,np.sin(t[0]/2)**2,(count,1))
    g=born_mmd_gradient(sampler,theta,target,count=2000)[0]
    p=np.sin(.8/2)**2
    expected=4*p*(1-np.exp(-1))*np.sin(.8)/2
    assert g==pytest.approx(expected,abs=.04)


def test_benchmark_splits_and_controls(tmp_path):
    result=benchmark(seeds=[7,17],methods=['kernel_surrogate'],samples=30,steps=0)
    assert len(result['runs'])==10
    for row in result['runs']:
        assert not set(row['train_indices'])&set(row['test_indices'])
        assert not set(row['validation_indices'])&set(row['test_indices'])
        if row['model']=='kernel_surrogate':assert row['test_state_preparations']==0
    json.dumps(result,allow_nan=False)
    assert len(generative_benchmark(seeds=[7],steps=1)['runs'])==3
