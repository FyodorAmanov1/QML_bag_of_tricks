import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pytest
from QML_utility.models.estimators import QuantumEstimator,ResourceLimit
from QML_utility.models.scalable import BatchedQuantumFeatures
from QML_utility.offline_surrogate import OfflineFeatureStudent,classical_head_spec
from QML_utility.evaluate_project import evaluation_plan,evaluate,summarize
from QML_utility.project_data import split_indices
from QML_utility.run_project import run
from QML_utility.predict_project import predict_csv
from QML_utility.audit_project import audit
from QML_utility.recommend import recommend


def project(tmp_path,rows=60,task='classification',adapter='reference'):
    source = tmp_path/'data.csv'
    with source.open('w',newline='') as handle:
        writer = csv.writer(handle);writer.writerow(['id','x','z','target','group','time'])
        for i in range(rows):
            writer.writerow([i,np.sin(i),np.cos(i/3),i%3 if task=='classification' else np.sin(i)+.2*np.cos(i/3),
                             i//5,f'2025-01-01T{i//3600:02}:{(i//60)%60:02}:{i%60:02}'])
    return dict(data_kind='classical',modality='numeric_table',task=task,objective='iid_'+task,capabilities={},
                allow_exploratory=True,framework='qiskit',adapter=adapter,seed=7,methods=['quantum_features'],
                quantum=dict(qubits=1,layers=1,batch_size=17,head_epochs=2),
                budget=dict(max_rows=20000 if adapter=='batched_features' else 512,state_preparations_per_trial=10000,
                            seconds_per_operation=120,max_feature_bytes=1000000),
                data=dict(path=str(source),features=['x','z'],target='target',id_column='id',split=dict(kind='iid')))


@pytest.mark.parametrize('framework',['qiskit','pennylane','cirq'])
@pytest.mark.parametrize('task',['classification','regression'])
def test_batched_training_exceeds_reference_cap_without_dense_gram(framework,task,tmp_path):
    pytest.importorskip(framework)
    rng = np.random.default_rng(31);x=rng.normal(size=(600,2))
    y = (x[:,0]>0).astype(int) if task=='classification' else x[:,0]-.2*x[:,1]
    model = BatchedQuantumFeatures(task=task,framework=framework,qubits=1,batch_size=17,epochs=3,seed=7).fit(x,y)
    assert model.resource_['state_preparations']==600
    assert model.resource_['max_state_batch_rows']==17 and model.resource_['dense_train_gram_bytes']==0
    assert model.resource_['head_epochs']==3 and model.resource_['largest_feature_matrix_bytes']==600*3*8
    prediction = model.predict(x[:3]);assert np.isfinite(prediction.astype(float)).all()
    assert model.resource_['state_preparations']==603
    path=tmp_path/'model.joblib';joblib.dump(model,path)
    assert np.allclose(joblib.load(path).predict(x[:3]).astype(float),prediction.astype(float))


def test_batched_memory_preparation_limits_and_failed_refit():
    pytest.importorskip('qiskit')
    x=np.arange(40.).reshape(20,2);y=np.arange(20)%2
    with pytest.raises(ResourceLimit,match='cache'):
        BatchedQuantumFeatures(qubits=1,max_feature_bytes=64).fit(x,y)
    with pytest.raises(ResourceLimit):
        BatchedQuantumFeatures(qubits=1,max_state_preparations=1).fit(x,y)
    model=QuantumEstimator(qubits=1).fit(x,y)
    with pytest.raises(ValueError):model.fit(x,np.ones(20))
    assert not hasattr(model,'fitted_')


@pytest.mark.parametrize('task',['classification','regression'])
def test_large_project_adapter_runs_and_is_auditable(tmp_path,task):
    pytest.importorskip('qiskit')
    config=project(tmp_path,600,task,'batched_features')
    portfolio=recommend(dict(config,constraints=dict(n_samples=600)))
    assert [item['id'] for item in portfolio['method_candidates']]==['quantum_features']
    report=run(config,tmp_path/'run')
    assert report['status']=='completed' and len(report['trials'])==13 and len(report['models'])==5
    assert audit(tmp_path/'run',config)['evaluated_families']==5
    quantum=next(model for model in report['models'] if model['method']=='quantum_features')
    assert quantum['resources']['state_preparations']==600
    assert quantum['resources']['dense_train_gram_bytes']==0


@pytest.mark.parametrize('kind',['iid','group'])
def test_repeated_holdouts_freeze_plan_audit_every_run_and_pair_results(tmp_path,kind):
    pytest.importorskip('qiskit')
    config=project(tmp_path)
    config['data']['split']=dict(kind=kind,**({'group_column':'group'} if kind=='group' else {}))
    config['evaluation']=dict(kind='repeated_holdout',seeds=[7,17],baseline='linear',practical_margin=.01)
    result=evaluate(config,tmp_path/'evaluation')
    assert result['status']=='completed' and len(result['runs'])==2
    assert all(record['audit']['verified'] for record in result['runs'])
    assert result['paired_comparisons']['quantum_features']['missing_runs']==[]
    assert len(result['paired_comparisons']['quantum_features']['per_run'])==2
    assert result['test_overlap'] and 'overlap' in result['uncertainty']
    assert not result['advantage_established'] and result['no_test_selected_deployment_winner']
    plan=json.loads((tmp_path/'evaluation/evaluation_plan.json').read_text())
    assert plan['frozen_before_execution'] and plan['runs'][0]['seed']==7
    for entry in plan['runs']:
        child=json.loads((tmp_path/'evaluation'/entry['run_id']/'config.json').read_text())
        assert child['data']['split']['test_ids']==entry['split']['test']


def test_walk_forward_windows_have_disjoint_tests_and_two_embargoes(tmp_path):
    pytest.importorskip('qiskit')
    config=project(tmp_path)
    config['data']['split']=dict(kind='chronological',time_column='time',gap_rows=2)
    config['evaluation']=dict(kind='walk_forward',n_splits=3,test_rows=8,validation_rows=8,min_train_rows=10)
    plan=evaluation_plan(config)
    tests=[]
    for entry in plan['runs']:
        split={key:list(map(int,values)) for key,values in entry['split'].items()}
        assert max(split['train'])+2<min(split['validation'])
        assert max(split['validation'])+2<min(split['test'])
        tests.append(set(split['test']))
    assert all(not tests[i]&tests[j] for i in range(3) for j in range(i))
    result=evaluate(config,tmp_path/'rolling')
    assert result['status']=='completed' and all(row['shared_test_records']==0 for row in result['test_overlap'])


def test_prespecified_partitions_and_evaluation_errors_are_explicit(tmp_path):
    config=project(tmp_path)
    config['evaluation']=dict(seeds=[7,7])
    with pytest.raises(ValueError,match='distinct'):evaluation_plan(config)
    y=np.arange(20)%2;rows=[{}]*20;ids=list(map(str,range(20)))
    with pytest.raises(ValueError,match='overlap'):
        split_indices(y,rows,dict(kind='predefined',train_ids=ids[:12],validation_ids=ids[10:16],test_ids=ids[16:]),7,'classification',ids)
    records=[dict(run_id='a',report=dict(models=[dict(method='linear',status='evaluated',test_score=2.),dict(method='quantum_features',status='evaluated',test_score=1.)])),
             dict(run_id='b',report=dict(models=[dict(method='linear',status='evaluated',test_score=2.)]))]
    plan=dict(baseline='linear',practical_margin=.1,runs=[dict(run_id='a',split=dict(test=['a'])),dict(run_id='b',split=dict(test=['a']))])
    result=summarize(records,plan,'regression')
    assert result['paired_comparisons']['quantum_features']['mean_improvement']==1
    assert result['paired_comparisons']['quantum_features']['missing_runs']==['b']


def test_csv_size_budget_rejects_before_parsing_all_records(tmp_path):
    from QML_utility.project_data import load_data
    config=project(tmp_path)
    config['budget']['max_csv_bytes']=10
    with pytest.raises(ValueError,match='max_csv_bytes'):load_data(config)


def make_student(framework='qiskit',task='classification'):
    rng=np.random.default_rng(24);x=rng.normal(size=(368,2))
    y=(x.sum(axis=1)>0).astype(int) if task=='classification' else x.sum(axis=1)
    teacher=QuantumEstimator(framework=framework,task=task,qubits=1,scale=.05).fit(x[:48],y[:48])
    gates=dict(max_feature_rmse=.03,max_feature_relative_l2=.35,max_task_degradation=.1)
    gates.update(dict(max_prediction_disagreement=.1,min_teacher_balanced_accuracy=.5) if task=='classification' else dict(max_prediction_rmse=.2,max_teacher_rmse=3.))
    student=OfflineFeatureStudent(kind='rff_ridge',alpha=1e-6,gamma=.15,components=128)
    student.fit_from_teacher(teacher,x[48:304],x[304:],y[304:],distill_ids=list(range(48,304)),
                             gate_ids=list(range(304,368)),teacher_used_ids=list(range(48)),gates=gates)
    return teacher,student,x,y,gates


@pytest.mark.parametrize('framework',['qiskit','pennylane','cirq'])
@pytest.mark.parametrize('task',['classification','regression'])
def test_frozen_teacher_distillation_and_no_native_inference(framework,task,tmp_path,monkeypatch):
    pytest.importorskip(framework)
    teacher,student,x,y,gates=make_student(framework,task)
    assert student.accepted_,student.gate_report_
    count=teacher.resource_['state_preparations']
    theta=teacher.theta_.copy()
    monkeypatch.setattr(teacher,'transform_quantum_features',lambda *_:(_ for _ in ()).throw(AssertionError('Teacher queried during inference')))
    prediction=student.predict(x[:5])
    assert np.isfinite(prediction.astype(float)).all() and teacher.resource_['state_preparations']==count
    assert np.array_equal(theta,teacher.theta_)
    student.export(tmp_path/'student.joblib')
    assert np.allclose(joblib.load(tmp_path/'student.joblib').predict(x[:5]).astype(float),prediction.astype(float))
    assert student.gate_report_['student_inference_quantum_queries']==0


def test_failed_fidelity_gate_blocks_export_and_overlapping_gate_spends_no_queries(tmp_path):
    pytest.importorskip('qiskit')
    teacher,student,x,y,gates=make_student()
    count=teacher.resource_['state_preparations']
    with pytest.raises(ValueError,match='overlap'):
        student.fit_from_teacher(teacher,x[48:304],x[304:],y[304:],distill_ids=list(range(48,304)),
                                  gate_ids=list(range(64)),teacher_used_ids=list(range(48)),gates=gates)
    assert teacher.resource_['state_preparations']==count
    assert not student.accepted_
    strict=dict(gates,max_feature_rmse=0.)
    student.fit_from_teacher(teacher,x[48:304],x[304:],y[304:],distill_ids=list(range(48,304)),
                              gate_ids=list(range(304,368)),teacher_used_ids=list(range(48)),gates=strict)
    assert not student.accepted_
    with pytest.raises(ValueError,match='gate failed'):student.export(tmp_path/'blocked.joblib')
    assert not (tmp_path/'blocked.joblib').exists()
    with pytest.raises(ValueError,match='cover every teacher class'):
        student.fit_from_teacher(teacher,x[48:304],x[304:],np.zeros(64),distill_ids=list(range(48,304)),
                                  gate_ids=list(range(304,368)),teacher_used_ids=list(range(48)),gates=gates)


def test_student_loads_and_predicts_in_a_process_with_all_quantum_imports_blocked(tmp_path):
    pytest.importorskip('qiskit')
    teacher,student,x,y,gates=make_student()
    student.export(tmp_path/'student.joblib')
    code="import sys,joblib; sys.modules.update({name:None for name in ('qiskit','pennylane','cirq','QML_utility.models')}); model=joblib.load(sys.argv[1]); print(len(model.predict([[0.,0.],[.2,.1]])))"
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'student.joblib')],capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='2'


def test_prediction_cli_uses_original_features_and_records_queries(tmp_path):
    pytest.importorskip('qiskit')
    teacher,student,x,y,gates=make_student()
    student.export(tmp_path/'student.joblib')
    source=tmp_path/'new.csv'
    source.write_text('id,z,x\na,0.2,0.1\nb,-0.3,0.2\n')
    # The student's gate manifest already owns student.json; preserve it.
    with pytest.raises(ValueError,match='preserved'):
        predict_csv(tmp_path/'student.joblib',source,tmp_path/'student.csv',['x','z'],'id')
    manifest=predict_csv(tmp_path/'student.joblib',source,tmp_path/'predictions.csv',['x','z'],'id')
    assert manifest['simulated_native_preparations']==0 and manifest['physical_qpu_jobs']==0
    with (tmp_path/'predictions.csv').open() as handle:
        assert len(list(csv.DictReader(handle)))==2
    joblib.dump(dict(model=teacher,features=['x','z']),tmp_path/'teacher.joblib')
    manifest=predict_csv(tmp_path/'teacher.joblib',source,tmp_path/'native.csv',id_column='id',state_budget=2)
    assert manifest['simulated_native_preparations']==2
    with pytest.raises(ValueError,match='order'):
        predict_csv(tmp_path/'teacher.joblib',source,tmp_path/'wrong.csv',['z','x'])
