"""Shared splits, train-only transforms, classical controls and resource reports."""
import argparse
import json
import time
import warnings
import hashlib
from importlib.metadata import version
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score,balanced_accuracy_score,mean_squared_error,mean_absolute_error
from .models import QMLModel,MODELS


def score(task,y,pred):
    if task=='classification':return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)))
    return dict(rmse=float(np.sqrt(mean_squared_error(y,pred))),mae=float(mean_absolute_error(y,pred)))


def benchmark(task='classification',framework='qiskit',seeds=(7,17,27),methods=('frozen_features','fidelity_kernel','kernel_surrogate'),
              samples=60,n_qubits=2,depth=1,steps=3,shots=None,x=None,y=None,artifact_dir=None):
    from sklearn.datasets import make_moons,make_regression
    if task=='quantum':
        # Known state preparations, not a black-box quantum-access separation.
        rng=np.random.default_rng(123);x=rng.normal(size=(samples,2));y=(x[:,0]>0).astype(int)
        learning_task='classification'
    else:
        learning_task=task
        if x is None:
            if task=='classification':x,y=make_moons(n_samples=samples,noise=.15,random_state=123)
            else:x,y=make_regression(n_samples=samples,n_features=4,n_informative=3,noise=5,random_state=123)
    x,y=np.asarray(x,float),np.asarray(y)
    if not 20<=len(x)<=512 or not np.isfinite(x).all():raise ValueError('Require 20..512 finite numeric rows for bounded reference benchmarks')
    runs=[]
    for seed in seeds:
        indices=np.arange(len(x));strat=y if learning_task=='classification' else None
        train,hold=train_test_split(indices,test_size=.4,random_state=seed,stratify=strat)
        val,test=train_test_split(hold,test_size=.5,random_state=seed,stratify=y[hold] if strat is not None else None)
        baselines={
            'classical_linear':make_pipeline(StandardScaler(),LogisticRegression(max_iter=300) if learning_task=='classification' else Ridge(alpha=.1)),
            'classical_rbf':make_pipeline(StandardScaler(),SVC(C=1.) if learning_task=='classification' else SVR(C=1.)),
            'classical_mlp':make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(8,),max_iter=200,random_state=seed) if learning_task=='classification' else MLPRegressor(hidden_layer_sizes=(8,),max_iter=200,random_state=seed)),
            'classical_boosting':HistGradientBoostingClassifier(max_iter=50,random_state=seed) if learning_task=='classification' else HistGradientBoostingRegressor(max_iter=50,random_state=seed)}
        for method in methods:
            baselines[method]=QMLModel(method,learning_task,framework,n_qubits,depth,seed,shots,steps)
        for name,model in baselines.items():
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always');model.fit(x[train],y[train])
                training=dict(getattr(model,'resources_',{}));fit_seconds=time.perf_counter()-start
                vp=model.predict(x[val]);before=dict(getattr(model,'resources_',{}));clock=time.perf_counter()
                tp=model.predict(x[test]);inference=time.perf_counter()-clock
            row=dict(model=name,seed=int(seed),train_indices=train.tolist(),validation_indices=val.tolist(),test_indices=test.tolist(),
                     validation=score(learning_task,y[val],vp),test=score(learning_task,y[test],tp),fit_seconds=fit_seconds,
                     test_inference_seconds=inference,training_resources=training,total_resources=dict(getattr(model,'resources_',{})),
                     test_state_preparations=getattr(model,'resources_',{}).get('state_preparations',0)-before.get('state_preparations',0),
                     warnings=[str(w.message) for w in captured])
            row['test_predictions']=np.asarray(tp).tolist();row['test_targets']=y[test].tolist()
            if name=='kernel_surrogate':row['held_out_teacher_audit']=model.audit_surrogate(x[val])
            if name=='classical_pretraining':
                row['same_pretrained_features_classical_head']=score(learning_task,y[test],model.predict_classical_control(x[test]))
            if artifact_dir:
                import joblib
                dest=Path(artifact_dir);dest.mkdir(parents=True,exist_ok=True);joblib.dump(model,dest/f'{name}-{seed}.joblib')
            runs.append(row)
    summary=[];metric='accuracy' if learning_task=='classification' else 'rmse'
    for name in sorted({r['model'] for r in runs}):
        values=np.array([r['test'][metric] for r in runs if r['model']==name])
        rng=np.random.default_rng(19);boot=rng.choice(values,(1000,len(values))).mean(axis=1)
        summary.append(dict(model=name,metric=metric,mean=float(values.mean()),seed_values=values.tolist(),
                            bootstrap_seed_mean_interval_95=np.quantile(boot,[.025,.975]).tolist()))
    source_root=Path(__file__).resolve().parents[1]
    hashes={str(p.relative_to(source_root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.rglob('*.py')}
    return dict(task=task,framework=framework,source_hashes=hashes,configuration=dict(seeds=list(seeds),samples=len(x),n_qubits=n_qubits,depth=depth,steps=steps,shots=shots),
                protocol=dict(split='60/20/20, stratified for binary classification',tuning='fixed configurations; zero search trials for ALL models',
                              limitations='small seed count gives weak interval coverage; no statistical superiority or quantum advantage certified',
                              quantum_data_access='known preparation descriptions' if task=='quantum' else 'classical vectors',
                              preprocessing='train-only scaling/PCA; classical controls retain full original features',hardware_executed=False),
                versions={name:version(name) for name in ('numpy','scipy','scikit-learn',{'cirq':'cirq-core'}.get(framework,framework))},runs=runs,summary=summary)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['classification','regression','quantum','generative'],default='classification')
    p.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    p.add_argument('--models',nargs='+',choices=MODELS,default=['frozen_features','fidelity_kernel','kernel_surrogate'])
    p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27]);p.add_argument('--samples',type=int,default=60)
    p.add_argument('--n-qubits',type=int,default=2);p.add_argument('--depth',type=int,default=1);p.add_argument('--steps',type=int,default=3)
    p.add_argument('--shots',type=int);p.add_argument('--csv',type=Path);p.add_argument('--target');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--artifact-dir',type=Path)
    a=p.parse_args(argv);x=y=None
    if a.csv and a.task=='generative':p.error('Generative CLI currently uses the built-in correlated-bit task; use BornMachine.fit for custom binary samples')
    if a.csv:
        import csv
        if not a.target:p.error('--target is required with --csv')
        with a.csv.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        fields=[k for k in rows[0] if k!=a.target];x=np.array([[float(r[k]) for k in fields] for r in rows]);y=np.array([r[a.target] for r in rows])
        if a.task=='regression':y=y.astype(float)
    result=(generative_benchmark(a.framework,a.seeds,a.n_qubits,a.steps) if a.task=='generative' else
            benchmark(a.task,a.framework,a.seeds,a.models,a.samples,a.n_qubits,a.depth,a.steps,a.shots,x,y,a.artifact_dir))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['summary'],indent=2))


def generative_benchmark(framework='qiskit',seeds=(7,17,27),n_qubits=2,steps=5):
    from .generative import BornMachine
    from .methods import mmd_unbiased
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed)
        train=np.repeat(rng.integers(0,2,(64,1)),n_qubits,axis=1)
        test=np.repeat(rng.integers(0,2,(128,1)),n_qubits,axis=1)
        model=BornMachine(framework,n_qubits,steps=steps,seed=seed).fit(train)
        samples=model.sample(128)
        controls={'born_mmd':samples,'classical_independent_bits':rng.binomial(1,train.mean(axis=0),(128,n_qubits)),
                  'classical_empirical_resampling':train[rng.integers(0,len(train),128)]}
        for name,data in controls.items():
            rows.append(dict(model=name,seed=int(seed),test_mmd_squared=mmd_unbiased(data,test),
                             resources=dict(model.resources_) if name=='born_mmd' else {},
                             interpretation='unbiased finite-sample MMD squared; negative values retained'))
    return dict(task='generative',framework=framework,runs=rows,summary=rows,
                protocol='correlated all-zero/all-one bits; independent train/test draws; fixed configurations; no advantage claim')
