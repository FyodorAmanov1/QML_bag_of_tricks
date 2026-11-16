"""Train a native reference teacher, distill its features, and test classical inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from .models.estimators import QuantumEstimator
from .offline_surrogate import OfflineFeatureStudent


def demo(output,framework='qiskit'):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a fresh surrogate demo directory')
    output.mkdir(parents=True)
    rng = np.random.default_rng(19)
    x = rng.normal(size=(896,2))
    y = (x.sum(axis=1)>0).astype(int)
    # Four disjoint roles: teacher training, mimic training, export gate, final test.
    a,b,c,d = x[:128],x[128:640],x[640:768],x[768:]
    teacher = QuantumEstimator('trainable_local',framework=framework,qubits=1,scale=.15,
                               regularization=.01,steps=10,max_state_preparations=20000,max_seconds=300)
    teacher.fit(a,y[:128])
    gates = dict(max_feature_rmse=.02,max_feature_relative_l2=.15,max_prediction_disagreement=.05,
                 max_task_degradation=.03,min_teacher_balanced_accuracy=.65)
    student = OfflineFeatureStudent(kind='rff_ridge',alpha=1e-5,gamma=.15,components=256,seed=7)
    student.fit_from_teacher(teacher,b,c,y[640:768],distill_ids=list(range(128,640)),
                             gate_ids=list(range(640,768)),teacher_used_ids=list(range(128)),gates=gates)
    result = dict(scope='Synthetic integration demo, not proprietary DQFE or quantum advantage',
                   framework=framework,gate=student.gate_report_,teacher_trace=teacher.trace_,export=None)
    if student.accepted_:
        result['export'] = student.export(output/'student.joblib')
        q_test = teacher.transform_quantum_features(d)
        before = teacher.resource_['state_preparations']
        predicted = student.predict(d)
        result['final_test'] = dict(feature_rmse=float(np.sqrt(np.mean((q_test-student.predict_features(d))**2))),
                                    student_balanced_accuracy=float(balanced_accuracy_score(y[768:],predicted)),
                                    student_native_preparations=teacher.resource_['state_preparations']-before,
                                    role='Untouched evaluation after the export decision; not used to tune or gate')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    args = parser.parse_args(argv)
    result = demo(args.output,args.framework)
    print(json.dumps(dict(accepted=result['gate']['accepted'],report=str(args.output/'report.json'),
                          exported_student=str(args.output/'student.joblib') if result['export'] else None),indent=2))


if __name__ == '__main__':
    main()
