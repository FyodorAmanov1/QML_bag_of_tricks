"""Run an explicit exploratory portfolio; freeze selection before test prediction."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback
import warnings
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from .models.estimators import QuantumEstimator, METHODS, preprocessing, bounded_angles
from .project_data import load_data, split_indices

ROOT = Path(__file__).resolve().parent
CLASSICAL = ('dummy','linear','rbf','boosting','fourier_linear')
SCALABLE_CLASSICAL = ('dummy','linear','random_features','hist_boosting')


def fourier_features(x, scale=1.):
    angles = bounded_angles(x)*scale
    return np.concatenate([np.sin(angles),np.cos(angles)],axis=1)


def encoding_scales(config):
    scales = config.get('quantum',{}).get('encoding_scale_candidates',[.1,.5,1.])
    if not isinstance(scales,list) or len(scales)!=3 or any(type(value) not in (int,float) or not np.isfinite(value) or not 0<value<=4 for value in scales):
        raise ValueError('Declare three finite encoding_scale_candidates in (0,4]')
    return scales


def candidates(method, task, config):
    seed = config.get('seed',7)
    if method in METHODS:
        q = config.get('quantum',{})
        for scale in encoding_scales(config):
            options = dict(method=method,task=task,framework=config.get('framework','qiskit'),seed=seed,
                           qubits=q.get('qubits',2),layers=q.get('layers',1),steps=q.get('steps',12),scale=scale,
                           max_state_preparations=config.get('budget',{}).get('state_preparations_per_trial',50000),
                           max_seconds=config.get('budget',{}).get('seconds_per_operation',120.))
            if config.get('adapter') == 'batched_features':
                from .models.scalable import BatchedQuantumFeatures
                options.pop('method');options.pop('steps')
                options.update(regularization=.001,epochs=q.get('head_epochs',10),batch_size=q.get('batch_size',128),
                               max_feature_bytes=config.get('budget',{}).get('max_feature_bytes',64*1024*1024))
                model = BatchedQuantumFeatures(**options)
                settings = dict(scale=scale,regularization=.001,qubits=options['qubits'],layers=options['layers'],
                                head_epochs=options['epochs'],batch_size=options['batch_size'])
            else:
                if method=='trainable_local':
                    options.update(optimizer=q.get('optimizer','adam'),metric_regularization=q.get('metric_regularization',.001))
                model = QuantumEstimator(**options)
                settings = dict(scale=scale,regularization=1.,qubits=options['qubits'],layers=options['layers'],steps=options['steps'])
                if method=='trainable_local':
                    settings.update(optimizer=model.optimizer,metric_regularization=model.metric_regularization)
            yield settings,model
    else:
        values = (1.,) if method == 'dummy' else (encoding_scales(config) if method == 'fourier_linear' else (.1,1.,10.))
        for value in values:
            clf = task == 'classification'
            if method == 'dummy':
                head = DummyClassifier(strategy='most_frequent') if clf else DummyRegressor()
            elif method == 'linear' or method == 'fourier_linear':
                strength = 1. if method == 'fourier_linear' else value
                head = LogisticRegression(C=strength,max_iter=1500,random_state=seed) if clf else Ridge(alpha=strength)
            elif method == 'rbf':
                head = SVC(C=value,gamma='scale') if clf else SVR(C=value,gamma='scale')
            elif method == 'boosting':
                options = dict(n_estimators=100,max_depth=2,learning_rate=value/20,random_state=seed)
                head = GradientBoostingClassifier(**options) if clf else GradientBoostingRegressor(**options)
            elif method == 'random_features':
                head = LogisticRegression(C=1.,max_iter=1500,random_state=seed) if clf else Ridge(alpha=1.)
            elif method == 'hist_boosting':
                options = dict(max_iter=100,max_leaf_nodes=15,learning_rate=value/20,
                               l2_regularization=1.,early_stopping=False,random_state=seed)
                head = HistGradientBoostingClassifier(**options) if clf else HistGradientBoostingRegressor(**options)
            else:
                raise ValueError('Unknown method: '+method)
            steps = [preprocessing()]
            if method == 'fourier_linear':
                steps.append(FunctionTransformer(fourier_features,kw_args=dict(scale=value)))
            if method == 'random_features':
                steps.append(RBFSampler(gamma=value/max(1,len(config['data']['features'])),n_components=128,random_state=seed))
            settings = dict(candidate=value,head_parameters=head.get_params())
            if method == 'fourier_linear':
                settings['encoding_scale'] = value
            if method == 'random_features':
                settings.update(rbf_gamma=steps[-1].gamma,random_features=128)
            yield settings,make_pipeline(*steps,head)


def score(task,y,predictions):
    return float(balanced_accuracy_score(y,predictions) if task == 'classification' else np.sqrt(mean_squared_error(y,predictions)))


def run(config, output, base=Path('.')):
    if config.get('task') not in ('classification','regression'):
        raise ValueError('Runnable adapter supports numeric classification/regression only')
    adapter = config.get('adapter','reference')
    if adapter not in ('reference','batched_features'):
        raise ValueError('adapter must be reference or batched_features')
    methods = config.get('methods',['quantum_features'] if adapter == 'batched_features' else list(METHODS))
    if not isinstance(methods,list) or not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise ValueError('methods must be a nonempty list of supported quantum candidates')
    if adapter == 'batched_features' and methods != ['quantum_features']:
        raise ValueError('Batched adapter supports only fixed quantum_features; use reference for dense/trainable families')
    if config.get('allow_exploratory') is not True:
        raise ValueError('Set allow_exploratory=true for this engineering comparison; it is not a paper reproduction')
    if type(config.get('seed',7)) is not int:
        raise ValueError('seed must be an integer')
    encoding_scales(config)
    x,y,rows,ids,payload = load_data(config,base)
    splits = split_indices(y,rows,config['data']['split'],config.get('seed',7),config['task'],ids)
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new output directory; preserve earlier runs')
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    train,validation,test = (splits[key] for key in ('train','validation','test'))
    report = dict(schema_version=1,scope='single_holdout_exploratory_comparison',task=config['task'],
                  adapter=adapter,
                  paper_replication=False,advantage_established=False,
                  metric='balanced_accuracy' if config['task']=='classification' else 'rmse',
                  data_sha256=hashlib.sha256(payload).hexdigest(),features=config['data']['features'],
                  splits={k:[ids[i] for i in indices] for k,indices in splits.items()},
                  split_kind=config['data']['split']['kind'],
                  uncertainty='One split; no significance, equivalence or generalization claim. Use independent data units and a preregistered repeated/external evaluation.',
                  selection_policy='Three prespecified candidates per non-dummy family; train-only preprocessing; validation selection; no train+validation refit. Unequal compute costs are recorded.',
                  resource_scope='Exact native statevectors and classical Gram calculations. Zero physical shots and QPU jobs. Per-operation time limits are cooperative for quantum preparation, not OS process timeouts.',
                  versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','joblib')},
                  python=platform.python_version(),trials=[],models=[])
    try:
        report['versions'][config.get('framework','qiskit')] = importlib.metadata.version(
            'cirq-core' if config.get('framework') == 'cirq' else config.get('framework','qiskit'))
    except importlib.metadata.PackageNotFoundError:
        report['versions'][config.get('framework','qiskit')] = 'not installed'
    report['source_sha256'] = {}
    from qml_tricks import geometry
    library = Path(geometry.__file__).parent
    report['library_source_sha256'] = {}
    for name in ('geometry.py','__init__.py'):
        source_bytes = (library/name).read_bytes()
        report['library_source_sha256'][name] = hashlib.sha256(source_bytes).hexdigest()
        snapshot = output/'source/qml_tricks'/name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(source_bytes)
    excluded = {'.cache','tests','results','build','dist','.venv','venv','.git','__pycache__'}
    for directory,subdirectories,filenames in os.walk(ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in excluded and not name.endswith(('.egg-info','.dist-info'))]
        for filename in sorted(filenames):
            if not filename.endswith('.py'):
                continue
            path = Path(directory)/filename
            relative = path.relative_to(ROOT)
            source_bytes = path.read_bytes()
            report['source_sha256'][str(relative)] = hashlib.sha256(source_bytes).hexdigest()
            snapshot = output/'source'/'QML_utility'/relative
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            snapshot.write_bytes(source_bytes)
    winners = {}
    classical = SCALABLE_CLASSICAL if adapter == 'batched_features' else CLASSICAL
    report['classical_controls'] = list(classical)
    for method in classical+tuple(methods):
        best = None
        for index,(settings,model) in enumerate(candidates(method,config['task'],config)):
            record = dict(method=method,candidate=index,settings=settings,status='failed')
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    model.fit(x[train],y[train])
                    predictions = model.predict(x[validation])
                    record.update(status='completed',validation_score=score(config['task'],y[validation],predictions),
                                  validation_predictions=np.asarray(predictions).tolist())
                    better = best is None or (record['validation_score'] > best[0] if config['task']=='classification' else record['validation_score'] < best[0])
                    if better:
                        best = record['validation_score'],index,model
                except Exception as error:
                    record.update(error=str(error),traceback=traceback.format_exc())
            record.update(warnings=[str(w.message) for w in caught],fit_and_validation_seconds=time.perf_counter()-started,
                          resources=dict(getattr(model,'resource_',{})),training_trace=getattr(model,'trace_',[]),
                          diagnostics=getattr(model,'diagnostics_',{}))
            report['trials'].append(record)
        if best:
            winners[method] = best
    # Immutable selection snapshot is written before any test prediction/scoring.
    selected = {method:dict(validation_score=value[0],candidate=value[1]) for method,value in winners.items()}
    validation_champion = (max if config['task']=='classification' else min)(selected,key=lambda m:selected[m]['validation_score']) if selected else None
    report['validation_champion'] = validation_champion
    (output/'selection.json').write_text(json.dumps(dict(selected=selected,validation_champion=validation_champion),indent=2)+'\n')
    for method,(_,index,model) in winners.items():
        result = dict(method=method,selected_candidate=index,status='test_failed')
        started = time.perf_counter()
        try:
            predictions = model.predict(x[test])
            result.update(status='evaluated',test_score=score(config['task'],y[test],predictions),
                          test_predictions=np.asarray(predictions).tolist(),
                          test_targets=y[test].tolist(),test_ids=[ids[i] for i in test])
            artifact = output/(method+'.joblib')
            joblib.dump(dict(model=model,features=config['data']['features'],task=config['task'],scope=report['scope'],
                             data_sha256=report['data_sha256'],split_ids=report['splits']),artifact)
            result['artifact'] = artifact.name
            result['artifact_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        except Exception as error:
            result.update(error=str(error),traceback=traceback.format_exc())
        result.update(test_prediction_seconds=time.perf_counter()-started,resources=dict(getattr(model,'resource_',{})))
        report['models'].append(result)
    report['status'] = 'completed' if len(report['models']) == len(classical)+len(methods) and all(m['status']=='evaluated' for m in report['models']) and all(t['status']=='completed' for t in report['trials']) else 'partial_with_failures'
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# Project comparison','',f"Scope: {report['scope']}. Metric: {report['metric']}.",'',
             'Selection used validation only. This run does not certify quantum advantage.','',
             '| Family | Status | Validation-selected test score |','|---|---|---:|']
    lines += [f"| {m['method']} | {m['status']} | {m.get('test_score','unavailable')} |" for m in report['models']]
    lines += ['',f"Validation champion, selected before test: `{validation_champion}`.",'',
              'Inspect report.json for every trial, failure, warning, split, prediction, source hash and cost.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    report = run(json.loads(args.config.read_text()),args.output,args.config.resolve().parent)
    print(json.dumps(dict(status=report['status'],report=str(args.output/'REPORT.md'),
                          validation_champion=report['validation_champion'],advantage_established=False),indent=2))


if __name__ == '__main__':
    main()
