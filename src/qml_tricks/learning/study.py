"""Bounded real-data study with shared splits and equal validation trial counts."""
import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
from importlib.metadata import version
import numpy as np
from sklearn.datasets import load_wine,load_breast_cancer,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.svm import SVC,SVR
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from .models import QMLModel
from .benchmark import score

STUDY_MODELS=('linear','rbf','mlp','boosting','linear_pca2','frozen_features','projected_kernel','fidelity_kernel')


def dataset(name,limit=128):
    loaders={'wine_binary':load_wine,'breast_cancer':load_breast_cancer,'diabetes':load_diabetes}
    if name not in loaders:raise ValueError(name)
    data=loaders[name]();x,y=data.data,data.target
    source=np.arange(len(x));task='regression' if name=='diabetes' else 'classification'
    if name=='wine_binary':
        mask=y<2;x,y,source=x[mask],y[mask],source[mask]
    if len(x)>limit:
        chosen,_=train_test_split(np.arange(len(x)),train_size=limit,random_state=123,stratify=y if task=='classification' else None)
        x,y,source=x[chosen],y[chosen],source[chosen]
    return x,y,task,dict(name=name,source_rows=source.tolist(),original_shape=list(data.data.shape),used_shape=list(x.shape),
        source='scikit-learn bundled real-observation dataset',description=data.DESCR,
        content_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        scope='small illustrative real-data subset; not clinical or production validation')


def estimator(name,task,trial,seed,framework='qiskit'):
    if name in ('frozen_features','projected_kernel','fidelity_kernel'):
        return QMLModel(name,task,framework,n_qubits=2,depth=1,seed=seed,steps=0,regularization=(.01,.1,1.)[trial])
    if name in ('linear','linear_pca2'):
        model=LogisticRegression(C=(.1,1.,10.)[trial],max_iter=500) if task=='classification' else Ridge(alpha=(.01,.1,1.)[trial])
    elif name=='rbf':model=SVC(C=(.1,1.,10.)[trial]) if task=='classification' else SVR(C=(.1,1.,10.)[trial])
    elif name=='mlp':
        cls=MLPClassifier if task=='classification' else MLPRegressor
        model=cls(hidden_layer_sizes=((4,),(8,),(16,))[trial],max_iter=300,random_state=seed)
    elif name=='boosting':
        cls=HistGradientBoostingClassifier if task=='classification' else HistGradientBoostingRegressor
        model=cls(max_leaf_nodes=(7,15,31)[trial],max_iter=75,random_state=seed)
    else:raise ValueError(name)
    steps=[StandardScaler()]
    if name=='linear_pca2':steps.append(PCA(2))
    steps.append(model);out=make_pipeline(*steps)
    return TransformedTargetRegressor(regressor=out,transformer=StandardScaler()) if task=='regression' else out


def run_study(output,seeds=tuple(range(10)),limit=128,trials=3,framework='qiskit',datasets=('wine_binary','breast_cancer','diabetes'),models=STUDY_MODELS):
    if not 20<=limit<=512 or not 1<=trials<=3 or not seeds:raise ValueError('Bounded limit, trials and nonempty seeds required')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    rows=[];provenance=[];started=time.perf_counter()
    for name in datasets:
        x,y,task,meta=dataset(name,limit);provenance.append(meta)
        for seed in seeds:
            allidx=np.arange(len(x));train,hold=train_test_split(allidx,test_size=.4,random_state=seed,stratify=y if task=='classification' else None)
            val,test=train_test_split(hold,test_size=.5,random_state=seed,stratify=y[hold] if task=='classification' else None)
            metric='balanced_accuracy' if task=='classification' else 'rmse'
            for method in models:
                history=[];fit_start=time.perf_counter()
                for trial in range(trials):
                    model=estimator(method,task,trial,seed,framework);clock=time.perf_counter()
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');model.fit(x[train],y[train]);pred=model.predict(x[val])
                        metrics=score(task,y[val],pred)
                        history.append(dict(trial=trial,validation=metrics,seconds=time.perf_counter()-clock,
                                            resources=dict(getattr(model,'resources_',{})),warnings=[str(w.message) for w in caught]))
                    except Exception as e:
                        history.append(dict(trial=trial,error=f'{type(e).__name__}: {e}',seconds=time.perf_counter()-clock))
                valid=[h for h in history if 'validation' in h]
                row=dict(dataset=name,task=task,seed=int(seed),model=method,train_indices=train.tolist(),validation_indices=val.tolist(),
                         test_indices=test.tolist(),trials=history,selection_metric=metric)
                if valid:
                    chosen=(max if task=='classification' else min)(valid,key=lambda h:h['validation'][metric])['trial']
                    final=estimator(method,task,chosen,seed,framework);refit=np.r_[train,val]
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');final.fit(x[refit],y[refit]);clock=time.perf_counter();pred=final.predict(x[test]);infer=time.perf_counter()-clock
                        row.update(selected_trial=chosen,refit_indices=refit.tolist(),test=score(task,y[test],pred),
                                   test_predictions=np.asarray(pred).tolist(),test_targets=y[test].tolist(),
                                   total_selection_refit_seconds=time.perf_counter()-fit_start,inference_seconds=infer,
                                   final_resources=dict(getattr(final,'resources_',{})),warnings=[str(w.message) for w in caught])
                    except Exception as e:row['error']=f'{type(e).__name__}: {e}'
                else:row['error']='All validation candidates failed'
                rows.append(row)
            print(f'Study completed: {name}, seed {seed}',flush=True)
    summaries=[]
    for name in datasets:
        for model in models:
            group=[r for r in rows if r['dataset']==name and r['model']==model and 'test' in r]
            if not group:continue
            metric=group[0]['selection_metric'];values=np.array([r['test'][metric] for r in group]);rng=np.random.default_rng(123)
            boot=rng.choice(values,(2000,len(values))).mean(axis=1)
            summaries.append(dict(dataset=name,model=model,metric=metric,mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.,
                                  seed_count=len(values),seed_mean_bootstrap_interval_95=np.quantile(boot,[.025,.975]).tolist(),
                                  mean_selection_refit_seconds=float(np.mean([r['total_selection_refit_seconds'] for r in group]))))
    root=Path(__file__).resolve().parents[1]
    result=dict(protocol=dict(seeds=list(seeds),row_limit=limit,trials_per_model=trials,framework=framework,
        test_policy='test used once after validation selection and train+validation refit',
        preprocessing='fit independently within each training partition; regression targets scaled within fit',
        budget='equal candidate counts, not equal wall time or circuit counts; complete candidate resource logs retained',
        uncertainty='descriptive seed bootstrap on overlapping repeated splits; not independent-replication confidence or significance',
        claims='no quantum advantage, production readiness or general predictive superiority established',
        execution='ideal native SDK simulation, no hardware noise/QPU'),datasets=provenance,runs=rows,summary=summaries,
        elapsed_seconds=time.perf_counter()-started,versions={p:version(p) for p in ('numpy','scipy','scikit-learn',{'cirq':'cirq-core'}.get(framework,framework))},
        platform=platform.platform(),source_hashes={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')})
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/study.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)));p.add_argument('--limit',type=int,default=128)
    p.add_argument('--trials',type=int,default=3);p.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    a=p.parse_args();run_study(a.output,tuple(a.seeds),a.limit,a.trials,a.framework)

if __name__=='__main__':main()
