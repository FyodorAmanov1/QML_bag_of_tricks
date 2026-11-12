"""Check native conventions, model behavior, attack access and saved outcomes."""
from importlib import import_module
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from scipy.special import expit

from qml_tricks.learning.case_models import (
    AmplitudeQVCClassifier, MulticlassQuantumKernel, TfidfQuantumMatcher,
    TOPOLOGIES, classical_product_kernel, finite_difference_input_gradient,
    projected_gradient_attack,
)

SDKS = ("qiskit", "pennylane", "cirq")
ROOT = Path(__file__).resolve().parents[1]


def backend(sdk):
    return import_module(f"qml_tricks.learning.case_backends.{sdk}")


@pytest.mark.parametrize("sdk",SDKS)
@pytest.mark.parametrize("n",[2,4])
@pytest.mark.parametrize("topology",TOPOLOGIES)
def test_kernel_rotation_and_interaction_phases(sdk,n,topology):
    x = np.linspace(-.8,1.1,n)
    state = backend(sdk).kernel_state(x,topology)
    if topology in ("x","y","z"):
        expected = np.array([1.],dtype=complex)
        for angle in x:
            if topology == "x": factor = [np.cos(angle/2),-1j*np.sin(angle/2)]
            elif topology == "y": factor = [np.cos(angle/2),np.sin(angle/2)]
            else: factor = np.exp(np.array([-1j,1j])*angle/2)/np.sqrt(2)
            expected = np.kron(expected,factor)
    else:
        if topology == "full": edges = [(i,j) for i in range(n) for j in range(i+1,n)]
        else:
            edges = [(i,i+1) for i in range(n-1)]
            if topology == "circular" and n > 2: edges.append((n-1,0))
        z = 1-2*np.array([[int(b) for b in format(i,f'0{n}b')] for i in range(2**n)])
        phase = z@x
        for i,j in edges: phase += x[i]*x[j]*z[:,i]*z[:,j]
        expected = np.exp(-.5j*phase)/np.sqrt(2**n)
    np.testing.assert_allclose(state,expected,atol=1e-12)


@pytest.mark.parametrize("sdk",SDKS)
@pytest.mark.parametrize("topology",TOPOLOGIES)
def test_kernel_psd_and_product_classical_equivalence(sdk,topology):
    x = np.random.default_rng(20).normal(size=(5,3))
    states = np.array([backend(sdk).kernel_state(row,topology) for row in x])
    gram = np.abs(states.conj()@states.T)**2
    np.testing.assert_allclose(gram,gram.T,atol=1e-12)
    np.testing.assert_allclose(np.diag(gram),1,atol=1e-12)
    assert np.linalg.eigvalsh(gram).min() > -1e-12
    if topology in ("x","y","z"):
        np.testing.assert_allclose(gram,classical_product_kernel(x,x),atol=1e-12)


@pytest.mark.parametrize("sdk",SDKS)
@pytest.mark.parametrize("index",[1,3,6])
def test_amplitude_probability_ordering(sdk,index):
    state = np.eye(8)[index]
    output = backend(sdk).qvc_state(state,np.zeros((1,3,3)))
    np.testing.assert_allclose(np.abs(output)**2,state,atol=1e-12)


@pytest.mark.parametrize("sdk",SDKS)
def test_qvc_is_a_linear_cz_chain(sdk):
    initial = np.ones(8)/np.sqrt(8)
    output = backend(sdk).qvc_state(initial,np.zeros((1,3,3)))
    expected = np.array([(-1)**(int(b[0])*int(b[1])+int(b[1])*int(b[2])) for b in [format(i,'03b') for i in range(8)]])/np.sqrt(8)
    np.testing.assert_allclose(output,expected,atol=1e-12)


@pytest.mark.parametrize("sdk",SDKS)
@pytest.mark.parametrize("n",[2,3])
def test_entity_independent_sdk_states_and_single_gate_gradient(sdk,n):
    rng = np.random.default_rng(14)
    angles = rng.normal(size=n)
    mixing = rng.normal(size=(1,n+(1 if n==2 else n)))
    strong = rng.normal(size=(1,n,3))
    reference = backend("qiskit").entity_state(angles,mixing,strong)
    state = backend(sdk).entity_state(angles,mixing,strong)
    np.testing.assert_allclose(abs(np.vdot(reference,state)),1,atol=1e-12)
    def expectation(t):
        changed = strong.copy(); changed[0,1,1] = t
        p = abs(backend(sdk).entity_state(angles,mixing,changed))**2
        return p[:2**(n-1)].sum()-p[2**(n-1):].sum()
    t = strong[0,1,1]
    shift = .5*(expectation(t+np.pi/2)-expectation(t-np.pi/2))
    numerical = (expectation(t+1e-5)-expectation(t-1e-5))/(2e-5)
    np.testing.assert_allclose(shift,numerical,atol=1e-9)


@pytest.mark.parametrize("sdk",SDKS)
def test_qvc_single_gate_gradient(sdk):
    rng = np.random.default_rng(2)
    amplitudes = rng.normal(size=8); amplitudes /= np.linalg.norm(amplitudes)
    theta = rng.normal(size=(2,3,3))
    def expectation(t):
        changed = theta.copy(); changed[0,1,1] = t
        p = abs(backend(sdk).qvc_state(amplitudes,changed))**2
        return p[:4].sum()-p[4:].sum()
    t = theta[0,1,1]
    shift = .5*(expectation(t+np.pi/2)-expectation(t-np.pi/2))
    numerical = (expectation(t+1e-5)-expectation(t-1e-5))/(2e-5)
    np.testing.assert_allclose(shift,numerical,atol=1e-9)


@pytest.mark.parametrize("sdk",SDKS)
def test_three_models_fit_and_predict_with_native_backend(sdk):
    x = np.random.default_rng(4).uniform(size=(6,4))
    y = np.array([0,1,0,1,0,1])
    qvc = AmplitudeQVCClassifier(framework=sdk,steps=0).fit(x,y)
    p = qvc.predict_proba(x)
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert np.isfinite(p).all() and qvc.state_evaluations_ == 12
    kernel = MulticlassQuantumKernel(framework=sdk).fit(x[:,:2],np.array([0,1,2,0,1,2]))
    assert kernel.predict(x[:2,:2]).shape == (2,)
    pairs = [("red sensor","red sensor"),("blue motor","blue motor"),("red sensor","blue motor"),
             ("green car","green car"),("green car","blue motor"),("red sensor","green car")]
    matcher = TfidfQuantumMatcher(framework=sdk,steps=0).fit(pairs,[1,1,0,1,0,0])
    p = matcher.predict_proba([("unseenword sensor","red sensor")])
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert "unseenword" not in matcher.vectorizer_.vocabulary_


def test_input_gradient_matches_analytic_logistic_gradient():
    x = np.array([[.2,.3],[.7,.8]])
    y = np.array([0,1])
    w = np.array([1.2,-.8])
    def predict(a):
        p = expit(a@w)
        return np.column_stack([1-p,p])
    expected = (predict(x)[:,1]-y)[:,None]*w
    np.testing.assert_allclose(finite_difference_input_gradient(predict,x,y),expected,atol=1e-9)


def test_pgd_respects_pixel_and_linf_constraints_and_attacks_a_known_model():
    x = np.array([[.2,.3],[.7,.8]])
    y = np.array([0,1])
    def predict(a):
        p = expit(a.sum(axis=1)-.5)
        return np.column_stack([1-p,p])
    attacked = projected_gradient_attack(predict,x,y,epsilon=.1,steps=3)
    assert np.max(abs(attacked-x)) <= .1+1e-12
    assert np.all((attacked>=0)&(attacked<=1))
    assert np.all(predict(attacked)[np.arange(2),y] < predict(x)[np.arange(2),y])
    np.testing.assert_array_equal(projected_gradient_attack(predict,x,y,epsilon=0),x)


def test_reference_report_has_scoped_evidence_isolated_splits_and_real_metrics():
    report = json.loads((ROOT/"results/case_models_reference.json").read_text(encoding="utf-8"))
    assert len(report["runs"]) == 9
    for name,digest in report["sources"].items():
        actual = hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        assert actual == digest, f"Stale case-model report: {name}"
    for run in report["runs"]:
        assert run["evidence_status"] == "local_reference_adaptation_not_paper_reproduction"
        assert run["qpu_jobs"] == 0 and run["shots"] is None
        partitions = [set(run["split_ids"][name]) for name in ("train","validation","test")]
        assert not (partitions[0]&partitions[1] or partitions[0]&partitions[2] or partitions[1]&partitions[2])
        if run["case"] == "entity_matching":
            groups = [{item.split(":")[0] for item in part} for part in partitions]
            assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        targets = np.asarray(run["test_targets"])
        for model in run["models"].values():
            endpoints = [model] if "accuracy" in model else model.values()
            for endpoint in endpoints:
                predictions = np.asarray(endpoint["predictions"])
                assert predictions.shape == targets.shape
                assert endpoint["accuracy"] == np.mean(predictions==targets)
        if run["case"] == "multiclass_kernel":
            assert run["models"]["pauli_quantum_control"]["predictions"] == run["models"]["identical_closed_form_pauli"]["predictions"]
        if run["case"] == "adversarial_qvc":
            original = np.asarray(run["attack"]["clean_inputs"])
            for field in ("classical_perturbed_inputs","quantum_perturbed_inputs"):
                perturbed = np.asarray(run["attack"][field])
                assert np.max(abs(original-perturbed)) <= run["attack"]["epsilon"]+1e-12
                assert np.all((perturbed>=0)&(perturbed<=1))
