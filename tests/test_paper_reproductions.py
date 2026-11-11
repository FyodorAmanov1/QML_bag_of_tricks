"""Paper circuit conventions, physical update and leakage/access checks."""
from importlib import import_module
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
from scipy.linalg import expm

from qml_tricks.learning.paper_reproductions import (
    alloy_features, channel_messages, delay_features, fixed_point,
    published_alloy_archive, reservoir_hamiltonian, reservoir_isometry,
    reservoir_step,
)


@pytest.mark.parametrize('n', [2,3,8])
@pytest.mark.parametrize('sdk', ['qiskit','pennylane','cirq'])
def test_native_zz_map_agrees_with_qiskit_library_convention(sdk,n):
    from qiskit.circuit.library import zz_feature_map
    from qiskit.quantum_info import Statevector
    x=np.random.default_rng(18).uniform(-1,1,n)
    circuit=zz_feature_map(feature_dimension=n,reps=1,entanglement='circular')
    state=np.asarray(Statevector.from_instruction(circuit.assign_parameters(x)))
    order=[int(format(i,f'0{n}b')[::-1],2) for i in range(2**n)]
    native=import_module(f'qml_tricks.learning.replication_backends.{sdk}').zz_state(x)
    np.testing.assert_allclose(abs(np.vdot(state[order],native)),1.,atol=1e-12)


@pytest.mark.parametrize('sdk',['qiskit','pennylane','cirq'])
@pytest.mark.parametrize('memory_qubits,readout_qubits',[(1,1),(2,4)])
def test_native_density_evolution_and_isometry_partial_trace(sdk,memory_qubits,readout_qubits):
    h0,h1=reservoir_hamiltonian(memory=memory_qubits,readout=readout_qubits)
    u=expm(-1j*(h0+.37*h1))
    d,r=2**memory_qubits,2**readout_qubits
    a=np.random.default_rng(17).normal(size=(d,d))+1j*np.random.default_rng(18).normal(size=(d,d))
    rho=a@a.conj().T;rho/=np.trace(rho)
    reset=np.zeros((r,r));reset[0,0]=1.
    initial=np.kron(rho,reset)
    native=import_module(f'qml_tricks.learning.replication_backends.{sdk}')
    evolved=native.evolve(initial,u)
    np.testing.assert_allclose(evolved,u@initial@u.conj().T,atol=1e-10)
    v=reservoir_isometry(h0,h1,.37,readout=readout_qubits)
    memory,p=reservoir_step(rho,v,readout=readout_qubits)
    full=evolved.reshape(d,r,d,r)
    np.testing.assert_allclose(memory,np.einsum('arbr->ab',full),atol=1e-10)
    np.testing.assert_allclose(p,np.einsum('arar->r',full).real,atol=1e-10)
    np.testing.assert_allclose(np.trace(memory),1.)
    assert np.linalg.eigvalsh(memory).min()>-1e-12


def test_null_fixed_point_and_probabilities():
    h0,h1=reservoir_hamiltonian(memory=1,readout=1)
    rho=fixed_point(h0,h1,readout=1)
    after,p=reservoir_step(rho,reservoir_isometry(h0,h1,0,readout=1),readout=1)
    np.testing.assert_allclose(after,rho,atol=1e-12)
    np.testing.assert_allclose(p.sum(),1.)


def test_preprocessing_replay_and_clean_protocol_are_distinct():
    rng=np.random.default_rng(2);x=rng.normal(size=(30,5));x[20:]+=20
    train=np.arange(20)
    _,clean=alloy_features(x,train,3,False)
    _,replay=alloy_features(x,train,3,True)
    assert clean['maximum_fit_indices']==train.tolist()
    assert clean['pca_fit_indices']==clean['bounds_fit_indices']==train.tolist()
    np.testing.assert_allclose(clean['maximum'],x[train].max(axis=0))
    assert replay['bounds_fit_indices']==list(range(30))
    assert replay['pca_fit_indices']==train.tolist()


def test_temporal_features_are_causal_and_do_not_cross_message_boundaries():
    messages=np.array([[1.,2.,3.],[7.,8.,9.]])
    x=delay_features(messages,degree=2,lags=2).reshape(2,3,4)
    np.testing.assert_array_equal(x[0,0],[0.,1.,0.,1.])
    np.testing.assert_array_equal(x[1,0],[0.,7.,0.,49.])
    changed=messages.copy();changed[:,2]+=100
    np.testing.assert_array_equal(delay_features(messages,2,2).reshape(2,3,4)[:,:2],
                                  delay_features(changed,2,2).reshape(2,3,4)[:,:2])


def test_channel_is_reproducible_and_labels_are_classical_four_pam():
    a,y=channel_messages(np.random.RandomState(4),3,20,20)
    b,z=channel_messages(np.random.RandomState(4),3,20,20)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(y,z)
    assert a.shape==y.shape==(3,20) and set(y.ravel())=={-3,-1,1,3}


def test_archived_scores_and_vendored_data_have_pinned_provenance():
    root=Path(__file__).resolve().parents[1]
    for record in json.loads((root/'reproductions/sources.json').read_text()):
        if record['local_path'].startswith('reproductions/'):
            b=(root/record['local_path']).read_bytes()
            lf=b.replace(b'\r\n',b'\n')
            assert record['sha256'] in {hashlib.sha256(a).hexdigest() for a in (b,lf,lf.replace(b'\n',b'\r\n'))}
    rows=published_alloy_archive(root/'reproductions/alloys')
    assert all(r['splits']==300 for r in rows)
    assert abs(rows[1]['quantum_mean']-.8943160690571047)<1e-12
