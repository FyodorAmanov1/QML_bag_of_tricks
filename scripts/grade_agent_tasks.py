"""Independently rerun five agent-produced solutions and check saved artifacts.

This measures bounded workflow feasibility, not general agent capability. Three
agents completed five tasks (two were follow-ups); no baseline/power study.
"""
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score

ROOT=Path(__file__).resolve().parents[1]
tasks=json.loads((ROOT/'agent_eval/tasks.json').read_text())


def disjoint(*groups):
    return all(not set(a)&set(b) for i,a in enumerate(groups) for b in groups[i+1:])


def grade(task,report,directory):
    if task=='csv_classification':
        assert report['configuration']['seeds']==[29]
        runs=report['runs'];q=[r for r in runs if not r['model'].startswith('classical_')]
        assert len(q)>=2 and len(runs)>=6
        for r in runs:
            assert disjoint(r['train_indices'],r['validation_indices'],r['test_indices'])
            assert len(r['test_predictions'])==len(r['test_indices'])
            assert r['test']['accuracy']==accuracy_score(r['test_targets'],r['test_predictions'])
            csv=np.genfromtxt(directory/'data.csv',delimiter=',',names=True)
            np.testing.assert_array_equal(np.array(r['test_targets']),csv['label'][r['test_indices']])
        return dict(split_isolation=True,metrics_recomputed=True,qml_models=len(q),controls=len(runs)-len(q))
    if task=='regression_artifact':
        assert report['seed']==31 and disjoint(report['train_indices'],report['test_indices'])
        assert report['reload_max_error']<1e-10
        np.testing.assert_allclose(report['preprocessing_train_mean'],report['actual_train_mean'])
        assert all(np.isfinite(v) for v in report['rmse'].values())
        assert list(directory.glob('*.joblib'))
        import joblib
        csv=np.genfromtxt(directory/'data.csv',delimiter=',',names=True)
        features=np.column_stack([csv[n] for n in csv.dtype.names if n!='target'])
        target=csv['target'];model=joblib.load(next(directory.glob('*.joblib')))
        pred=model.predict(features[report['test_indices']])
        rmse=float(np.sqrt(np.mean((pred-target[report['test_indices']])**2)))
        assert abs(rmse-report['rmse']['frozen_features'])<1e-10
        np.testing.assert_allclose(model.scaler_.mean_,features[report['train_indices']].mean(axis=0))
        return dict(train_only_preprocessing=True,reload_max_error=report['reload_max_error'])
    if task=='finite_shot_gradient':
        exact=report['exact'];sampled=report.get('sampled',report.get('finite_shot'))
        assert exact['configuration']['seed']==37 and exact['configuration']['shots'] is None
        assert sampled['configuration']['shots']==4096 and sampled['configuration']['steps']==0
        resources=sampled['resources']
        assert resources['simulated_measurement_shots']==4096*resources['sampled_objective_evaluations']
        assert resources['simulated_measurement_shots']>0 and report['limitations']
        return dict(shots_accounted=True,bias_disclosed=True)
    if task=='surrogate_deployment':
        assert report['validation']['passed'] and report['inference_quantum_calls']==0
        assert report['test_predictions']
        files=list(directory.glob('*.joblib'));assert files
        from qml_tricks.learning.offline_surrogate import ClassicalFeatureStudent
        obj=ClassicalFeatureStudent.load(files[0]);assert not hasattr(obj,'teacher_')
        return dict(validation_passed=True,teacher_free_artifact=True)
    if task=='failed_surrogate_gate':
        assert not report['validation']['passed'] and report['export_rejected']
        assert report['exception_type']=='ValueError' and report['limitations']
        assert not list(directory.glob('*.joblib'))
        return dict(failed_gate_enforced=True)
    raise ValueError(task)


rows=[]
for task in tasks:
    directory=ROOT/'agent_eval/runs'/task['id'];script=directory/'solution.py';started=time.perf_counter()
    proc=subprocess.run([sys.executable,str(script)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    result=dict(task=task['id'],replay_seconds=time.perf_counter()-started,returncode=proc.returncode,
                script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),activity_log=str((directory/'activity.md').relative_to(ROOT)))
    try:
        assert proc.returncode==0,proc.stderr[-2000:]
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        result['checks']=grade(task['id'],report,directory);result['passed']=True
    except Exception as e:result.update(passed=False,error=f'{type(e).__name__}: {e}')
    rows.append(result);print(f'{task["id"]}: {result["passed"]}',flush=True)
out=dict(protocol='Three fresh-context agents, five project tasks; two follow-ups reuse task context. Independent replay and artifact assertions.',
         human_scientific_review='deferred to repository owner',
         limitations='One attempt per scenario, no control condition, no measured token/tool counts; activity logs are self-reported. Not a general agent success-rate estimate.',
         tasks=rows,passed=sum(r['passed'] for r in rows),total=len(rows))
(ROOT/'results/agent_evaluation.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
if not all(r['passed'] for r in rows):raise SystemExit(1)
