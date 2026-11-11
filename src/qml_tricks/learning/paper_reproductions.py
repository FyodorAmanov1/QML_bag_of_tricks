"""Bounded numerical replications of two classical-data application papers.

Run from the source checkout: python -m qml_tricks.learning.paper_reproductions
Provenance and the frozen protocol are in reproductions/. No hardware jobs.
"""
import argparse
import gzip
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.linalg import eigh, svd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def scores(y, prediction):
    return dict(accuracy=float(accuracy_score(y,prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y,prediction)),
                error=float(1-accuracy_score(y,prediction)))


class LegacyPCA:
    """Full SVD with U-based svd_flip used by the author's legacy sklearn stack."""
    def __init__(self, n): self.n=n
    def fit(self, x):
        self.mean_=x.mean(axis=0)
        u, _, v=svd(x-self.mean_, full_matrices=False)
        signs=np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
        self.components_=(v*signs[:,None])[:self.n]
        return self
    def transform(self, x): return (x-self.mean_)@self.components_.T


def alloy_features(x, train, n, author_replay=False):
    fit_indices=np.arange(len(x)) if author_replay else np.asarray(train)
    maximum=np.max(x[fit_indices], axis=0); maximum[maximum==0]=1
    scaled=x/maximum
    pca=LegacyPCA(n).fit(scaled[train]); reduced=pca.transform(scaled)
    scaler=StandardScaler().fit(reduced[train]); z=scaler.transform(reduced)
    bounds=MinMaxScaler((-1,1)).fit(z[fit_indices]); features=bounds.transform(z)
    return features, dict(maximum_fit_indices=fit_indices.tolist(), pca_fit_indices=list(map(int,train)),
        bounds_fit_indices=fit_indices.tolist(), maximum=maximum.tolist(), pca_mean=pca.mean_.tolist(),
        scaler_mean=scaler.mean_.tolist(), bounds_data_min=bounds.data_min_.tolist(),
        bounds_data_max=bounds.data_max_.tolist())


def state_bank(features, framework):
    native=import_module(f'qml_tricks.learning.replication_backends.{framework}')
    return np.array([native.zz_state(row) for row in features])


def fidelity(a,b): return np.clip(np.abs(a.conj()@b.T)**2,0,1)


def fit_record(name, estimator, x, y, train, test, parameters=None):
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');estimator.fit(x[train],y[train]);prediction=estimator.predict(x[test])
    return dict(model=name,parameters=parameters or {},test=scores(y[test],prediction),
        test_predictions=np.asarray(prediction).tolist(),test_targets=y[test].tolist(),
        seconds=time.perf_counter()-started,warnings=[str(w.message) for w in caught])


def published_alloy_archive(directory):
    result=[]
    for n in (8,10):
        with (directory/f'published_{n}.npy').open('rb') as f:
            arrays=[np.load(f,allow_pickle=False) for _ in range(7)]
        q=arrays[3];classical=arrays[5]
        result.append(dict(qubits=n,splits=len(q),quantum_mean=float(q.mean()),
            quantum_seed_sd=float(q.std(ddof=1)),
            classical_means={k:float(classical[:,j].mean()) for j,k in enumerate(('linear','poly','rbf','sigmoid'))},
            quantum_scores=q.tolist(),classical_scores=classical.tolist(),
            status='Recomputed from author-supplied score archive; not fresh circuit execution'))
    return result


def reproduce_alloys(root, protocol, framework):
    directory=root/'reproductions/alloys'
    data=np.loadtxt(directory/'data.csv',delimiter=',',skiprows=1);x,y=data[:,:-1],data[:,-1].astype(int)
    assert np.isfinite(data).all() and x.shape==(1252,15)
    archived=published_alloy_archive(directory);replays=[];controlled=[]
    for seed in protocol['fresh_replay_seeds']:
        train,test=train_test_split(np.arange(len(x)),test_size=.2,random_state=seed)
        for n in protocol['qubits']:
            started=time.perf_counter();features,prep=alloy_features(x,train,n,True)
            states=state_bank(features,framework);gram=fidelity(states,states)
            model=SVC(kernel='precomputed');model.fit(gram[np.ix_(train,train)],y[train]);pred=model.predict(gram[np.ix_(test,train)])
            rows=[dict(model='quantum',test=scores(y[test],pred),test_predictions=pred.tolist(),test_targets=y[test].tolist())]
            for kernel in ('linear','poly','rbf','sigmoid'):
                rows.append(fit_record(kernel,SVC(kernel=kernel),features,y,train,test))
            reference=next(a for a in archived if a['qubits']==n)['quantum_scores'][seed-22]
            replays.append(dict(seed=seed,qubits=n,train_indices=train.tolist(),test_indices=test.tolist(),
                models=rows,preprocessing=prep,author_archived_quantum_accuracy=reference,
                fresh_minus_archived=float(rows[0]['test']['accuracy']-reference),
                seconds=time.perf_counter()-started,
                resources=dict(native_statevector_executions=len(x),complex_statevector_bytes=16*2**n,
                    cached_gram_bytes=gram.nbytes,hardware_shots=0)))
        print(f'Alloy author-workflow replay seed {seed}',flush=True)
    for seed in protocol['controlled_seeds']:
        train,hold=train_test_split(np.arange(len(x)),test_size=.4,random_state=seed,stratify=y)
        val,test=train_test_split(hold,test_size=.5,random_state=seed,stratify=y[hold])
        histories=[];clock=time.perf_counter();executions=0
        for n in protocol['qubits']:
            features,_=alloy_features(x,train,n,False)
            states=state_bank(features[np.r_[train,val]],framework);executions+=len(states)
            ktrain=fidelity(states[:len(train)],states[:len(train)])
            kval=fidelity(states[len(train):],states[:len(train)])
            for c in protocol['controlled_quantum_C']:
                model=SVC(C=c,kernel='precomputed').fit(ktrain,y[train])
                histories.append(dict(qubits=n,C=c,validation_accuracy=float(model.score(kval,y[val]))))
        selected=max(histories,key=lambda r:r['validation_accuracy']);refit=np.r_[train,val]
        features,prep=alloy_features(x,refit,selected['qubits'],False)
        states=state_bank(features[np.r_[refit,test]],framework);executions+=len(states)
        ktrain=fidelity(states[:len(refit)],states[:len(refit)])
        ktest=fidelity(states[len(refit):],states[:len(refit)])
        q=SVC(C=selected['C'],kernel='precomputed').fit(ktrain,y[refit]);pred=q.predict(ktest)
        rows=[dict(model='quantum',selected=selected,trials=histories,test=scores(y[test],pred),
            test_predictions=pred.tolist(),test_targets=y[test].tolist(),seconds=time.perf_counter()-clock,
            native_statevector_executions=executions,preprocessing=prep)]
        for name in ('logistic_l2','rbf_svc','boosting'):
            trials=[]
            grid=([dict(C=c) for c in (.01,.1,1.,10.,100.)] if name=='logistic_l2' else
                  [dict(C=c,gamma=g) for c in (.1,1.,10.,100.) for g in (.01,.1,1.)] if name=='rbf_svc' else
                  [dict(max_leaf_nodes=l,learning_rate=r) for l in (7,15,31) for r in (.03,.1,.3)])
            def estimator(parameters):
                model=(LogisticRegression(max_iter=1000,**parameters) if name=='logistic_l2' else
                       SVC(**parameters) if name=='rbf_svc' else HistGradientBoostingClassifier(random_state=seed,**parameters))
                return make_pipeline(StandardScaler(),model)
            for params in grid:
                model=estimator(params).fit(x[train],y[train])
                trials.append(dict(parameters=params,validation_accuracy=float(model.score(x[val],y[val]))))
            best=max(trials,key=lambda r:r['validation_accuracy'])
            row=fit_record(name,estimator(best['parameters']),x,y,refit,test,best['parameters'])
            row['trials']=trials;rows.append(row)
        controlled.append(dict(seed=seed,train_indices=train.tolist(),validation_indices=val.tolist(),test_indices=test.tolist(),models=rows))
        print(f'Alloy leakage-free controlled seed {seed}',flush=True)
    return dict(dataset=dict(rows=len(x),features=x.shape[1],sha256=hashlib.sha256((directory/'data.csv').read_bytes()).hexdigest(),
        duplicate_feature_rows=int(len(x)-len(np.unique(x,axis=0))),labels=dict(zip(*[list(map(int,a)) for a in np.unique(y,return_counts=True)]))),
        author_archive=archived,fresh_replays=replays,controlled=controlled,
        limits='Author replay deliberately retains global max and train+test MinMax fitting. Controlled extension removes it and tunes baselines. Ten fresh splits per configuration, not the full 300-split parameter sweep. Neither alloy discovery nor language-model screening is rerun.')


def reservoir_hamiltonian(memory=2, readout=4, seed=17462):
    n=memory+readout;rng=np.random.RandomState(seed)
    hx=(1+rng.randn(n))*2;hz=(1+rng.randn(n))*.5
    j=np.triu(rng.rand(n,n),1)
    ident=np.eye(2);xx=np.array([[0,1],[1,0]]);zz=np.diag([1,-1])
    def local(op,k):
        value=np.array([[1.]])
        for i in range(n):value=np.kron(value,op if i==k else ident)
        return value
    x=[local(xx,k) for k in range(n)];z=[local(zz,k) for k in range(n)]
    h0=sum(hx[k]*x[k] for k in range(n))+sum(j[k,l]*z[k]@z[l] for k in range(n) for l in range(k+1,n))
    h1=sum(hz[k]*z[k] for k in range(n))
    return h0,h1


def reservoir_isometry(h0,h1,u,readout=4):
    values,vectors=eigh(h0+float(u)*h1)
    # Columns whose readout register starts in |0...0>. Only a 64x4 isometry
    # is required instead of evolving a full 64x64 density matrix every step.
    columns=np.arange(len(h0)//2**readout)*2**readout
    return (vectors*np.exp(-1j*values))@vectors[columns,:].conj().T


def reservoir_step(rho,isometry,readout=4):
    d=len(rho);v=isometry.reshape(d,2**readout,d)
    next_rho=np.einsum('ari,ij,brj->ab',v,rho,v.conj(),optimize=True)
    probabilities=np.einsum('ari,ij,arj->r',v,rho,v.conj(),optimize=True).real
    probabilities=np.maximum(probabilities,0);probabilities/=probabilities.sum()
    return next_rho,probabilities


def fixed_point(h0,h1,readout=4):
    d=len(h0)//2**readout;rho=np.eye(d,dtype=complex)/d
    v=reservoir_isometry(h0,h1,0.,readout)
    for _ in range(10000):
        after,_=reservoir_step(rho,v,readout)
        if np.linalg.norm(after-rho)<1e-13:return after
        rho=after
    raise RuntimeError('Null-input fixed point did not converge')


def channel_messages(rng,count,length,snr=20):
    y=2*rng.randint(0,4,(count,length))-3
    coefficients=[1.,.18,-.1,.091,-.05,.04,.03,.01]
    x=[]
    for message in y:
        filtered=np.convolve(coefficients,np.r_[message[-8:],message],mode='full')[8:-7]
        x.append(filtered+.06*filtered**2-.01*filtered**3+10**(-snr/20)*rng.randn(length))
    return np.array(x),y


def reservoir_features(messages,h0,h1,rho0,readout=4):
    features=[]
    for message in messages:
        rho=rho0.copy();row=[]
        for value in message:
            rho,p=reservoir_step(rho,reservoir_isometry(h0,h1,value,readout),readout);row.append(p)
        features.append(row)
    return np.array(features)


def delay_features(messages,degree=1,lags=8):
    # Each message starts a separate stream; no future input or preceding
    # message is used. Initial unavailable lags are zero padded.
    padded=np.pad(messages,((0,0),(lags-1,0)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,lags,axis=1)
    return np.concatenate([windows**power for power in range(1,degree+1)],axis=2).reshape(-1,lags*degree)


def reproduce_communications(protocol):
    h0,h1=reservoir_hamiltonian();rho0=fixed_point(h0,h1);runs=[]
    for seed in protocol['data_seeds']:
        rng=np.random.RandomState(seed);length=protocol['symbols_per_message'];clock=time.perf_counter()
        train=channel_messages(rng,protocol['train_messages'],length,protocol['snr_db'])
        val=channel_messages(rng,protocol['validation_messages'],length,protocol['snr_db'])
        test=channel_messages(rng,protocol['test_messages'],length,protocol['snr_db'])
        inputs=np.concatenate([train[0],val[0],test[0]])
        y=np.concatenate([train[1],val[1],test[1]]).reshape(-1)
        tr=np.arange(train[0].size);va=np.arange(train[0].size,train[0].size+val[0].size)
        te=np.arange(train[0].size+val[0].size,len(y))
        features=reservoir_features(inputs,h0,h1,rho0)
        quantum=features.reshape(-1,16);raw=inputs.reshape(-1,1)
        rows=[]
        for name,xx in (('quantum_reservoir',quantum),('memoryless_logistic',raw)):
            lr=LogisticRegression(C=1e6,solver='newton-cg',max_iter=500)
            rows.append(fit_record(name,lr,xx,y,tr,te,dict(C=1e6)))
        refit=np.r_[tr,va]
        for name in ('polynomial_delay_logistic','delay_mlp_l2'):
            trials=[]
            for choice in range(9):
                degree=choice//3+1 if name=='polynomial_delay_logistic' else 1
                xx=delay_features(inputs,degree)
                def estimator(k):
                    model=(LogisticRegression(C=(.01,1.,100.)[k%3],max_iter=500) if name=='polynomial_delay_logistic' else
                           MLPClassifier(hidden_layer_sizes=((16,),(32,),(64,))[k//3],alpha=(.0001,.01,1.)[k%3],random_state=seed,max_iter=200))
                    return make_pipeline(StandardScaler(),model)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model=estimator(choice).fit(xx[tr],y[tr])
                    score=float(model.score(xx[va],y[va]))
                trials.append(dict(choice=choice,degree=degree,validation_accuracy=score,warnings=[str(w.message) for w in caught]))
            best=max(trials,key=lambda r:r['validation_accuracy'])
            xx=delay_features(inputs,best['degree'])
            row=fit_record(name,estimator(best['choice']),xx,y,refit,te,dict(choice=best['choice'],degree=best['degree']))
            row['trials']=trials;rows.append(row)
        runs.append(dict(seed=seed,models=rows,total_seconds=time.perf_counter()-clock,
            train_messages=protocol['train_messages'],validation_messages=protocol['validation_messages'],test_messages=protocol['test_messages'],
            test_inputs=test[0].tolist(),test_labels=test[1].tolist(),
            features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
            resources=dict(quantum_qubits=6,memory_qubits=2,readout_qubits=4,
                density_steps=len(y),hardware_shots=0,sampling='Exact unconditional mean dynamics, S=infinity',
                full_density_bytes=16*64**2,stored_memory_density_bytes=16*4**2,
                isometry_bytes=16*64*4,feature_bytes=features.nbytes,
                implementation='Classical SciPy diagonalization and Kraus/isometry contraction; not native SDK jobs or hardware. Native one-step equivalence is tested separately.')))
        print(f'Communications numerical replication seed {seed}',flush=True)
    return dict(runs=runs,null_fixed_point_residual=float(np.linalg.norm(reservoir_step(rho0,reservoir_isometry(h0,h1,0))[0]-rho0)),
        limits='One SNR and three newly seeded message datasets; not exact upstream randomness or the full six-SNR/hardware study. Complete graph is explicit because the upstream notebook overwrites it with a split graph before later cells. Additional temporal controls use validation messages and refit on train+validation; author-style models fit train messages only.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--study',choices=['all','alloys','communications'],default='all')
    p.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    args=p.parse_args();root=args.root.resolve();protocol=json.loads((root/'reproductions/protocol.json').read_text())
    started=time.perf_counter();result=dict(protocol=protocol,framework=args.framework)
    with threadpool_limits(limits=1):
        if args.study in ('all','alloys'):result['alloys']=reproduce_alloys(root,protocol['alloys'],args.framework)
        if args.study in ('all','communications'):result['communications']=reproduce_communications(protocol['communications'])
    source=Path(__file__).resolve().parents[1]
    result.update(elapsed_seconds=time.perf_counter()-started,versions={p:version(p) for p in ('numpy','scipy','scikit-learn',{'cirq':'cirq-core'}.get(args.framework,args.framework))},
        provenance=json.loads((root/'reproductions/sources.json').read_text()),
        source_hashes={str(f.relative_to(source)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*.py')})
    out=root/'results'/f'paper_{args.study}.json.gz'
    out.write_bytes(gzip.compress(json.dumps(result,indent=2,allow_nan=False).encode(),mtime=0))
    print('Saved',out.name,flush=True)


if __name__=='__main__':main()
