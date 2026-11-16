"""Offline feature-map distillation with an independent, export-gating holdout.

Kipu-style workflow reference, not proprietary DQFE or a peer-reviewed advantage
claim. Serialized students contain classical models/arrays, never a teacher.
"""
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge,LogisticRegression,SGDClassifier,SGDRegressor
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.validation import check_is_fitted


def _ids(values,count,name):
    if values is None or len(values) != count:
        raise ValueError(name+' must identify every record')
    result = [str(value) for value in values]
    if len(set(result)) != count:
        raise ValueError(name+' must contain unique record identities')
    return result


def _head_predict(spec,features):
    if spec['feature_scaler'] is not None:
        features = spec['feature_scaler'].transform(features)
    if spec['kind'] == 'logits':
        values = features@spec['weights']+spec['bias']
        predictions = values.argmax(axis=1) if spec['task']=='classification' else values[:,0]
    else:
        predictions = spec['head'].predict(features)
    if spec['task'] == 'classification':
        return spec['classes'][np.asarray(predictions,int)]
    return np.asarray(predictions,float)*spec['output_scale']+spec['output_mean']


def _teacher_head(teacher):
    if teacher.method not in ('quantum_features','projected_kernel','trainable_local'):
        raise ValueError('Local-feature distillation cannot replace an arbitrary fidelity-kernel Gram head')
    spec = dict(task=teacher.task,classes=copy.deepcopy(getattr(teacher,'classes_',None)),
                feature_scaler=copy.deepcopy(getattr(teacher,'feature_scaler_',None)),
                output_mean=0.,output_scale=1.)
    if teacher.method == 'trainable_local':
        spec.update(kind='logits',weights=teacher.weights_.copy(),bias=teacher.bias_.copy())
    else:
        allowed = (Ridge,LogisticRegression,SGDClassifier,SGDRegressor,SVC,KernelRidge)
        if type(teacher.head_) not in allowed or callable(getattr(teacher.head_,'kernel',None)) or getattr(teacher.head_,'kernel',None)=='precomputed':
            raise ValueError('Require a supported wholly classical local-feature head')
        spec.update(kind='sklearn',head=copy.deepcopy(teacher.head_))
    if teacher.task == 'regression' and (teacher.method=='trainable_local' or spec['feature_scaler'] is not None):
        spec.update(output_mean=teacher.target_mean_,output_scale=teacher.target_scale_)
    return spec


def classical_head_spec(head,task,classes=None,feature_scaler=None):
    """Describe a pre-fitted classical head for externally measured feature pairs.

    Classification heads must predict integer indices into the supplied classes.
    fit_from_pairs checks supported classical head/scaler types before use.
    """
    return dict(kind='sklearn',head=head,task=task,
                classes=None if classes is None else np.asarray(classes),
                feature_scaler=feature_scaler,output_mean=0.,output_scale=1.)


class OfflineFeatureStudent(BaseEstimator):
    def __init__(self,kind='ridge',alpha=1.,gamma=.2,components=256,seed=7,max_feature_bytes=64*1024*1024):
        self.kind,self.alpha,self.gamma = kind,alpha,gamma
        self.components,self.seed,self.max_feature_bytes = components,seed,max_feature_bytes

    def fit(self,x,y,**kwargs):
        """Fit input/teacher-feature pairs; holdout/head/gate arguments are mandatory."""
        return self.fit_from_pairs(x,y,**kwargs)

    def fit_from_teacher(self,teacher,x_distill,x_gate,y_gate,*,distill_ids,gate_ids,teacher_used_ids,gates):
        self.accepted_ = False
        if hasattr(self,'fitted_'):
            del self.fitted_
        check_is_fitted(teacher,'fitted_')
        a,b = np.asarray(x_distill,float),np.asarray(x_gate,float)
        # Validate identities before spending any teacher queries.
        first,second = _ids(distill_ids,len(a),'distill_ids'),_ids(gate_ids,len(b),'gate_ids')
        used = set(map(str,teacher_used_ids))
        if set(first)&set(second) or set(second)&used:
            raise ValueError('Gate records overlap distillation or prior teacher training/selection/evaluation records')
        spec = _teacher_head(teacher)
        before = dict(teacher.resource_)
        q_distill = teacher.transform_quantum_features(a)
        q_gate = teacher.transform_quantum_features(b)
        return self.fit_from_pairs(a,q_distill,b,q_gate,y_gate,head_spec=spec,
                                   distill_ids=first,gate_ids=second,teacher_used_ids=teacher_used_ids,
                                   gates=gates,teacher_acquisition=dict(
                                       state_preparations=teacher.resource_['state_preparations']-before['state_preparations'],
                                       physical_shots=teacher.resource_.get('measurement_shots',0)-before.get('measurement_shots',0),
                                       qpu_jobs=teacher.resource_.get('qpu_jobs',0)-before.get('qpu_jobs',0),
                                       circuit_parameter_sha256=hashlib.sha256(np.ascontiguousarray(teacher.theta_).tobytes()).hexdigest()))

    def fit_from_pairs(self,x_distill,q_distill,x_gate,q_gate,y_gate,*,head_spec,distill_ids,gate_ids,
                       teacher_used_ids,gates,teacher_acquisition=None):
        if hasattr(self,'fitted_'):
            del self.fitted_
        self.accepted_ = False
        if self.kind not in ('ridge','rff_ridge') or not np.isfinite(self.alpha) or self.alpha <= 0 or not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError('Require ridge/rff_ridge and positive alpha/gamma')
        if type(self.components) is not int or not 1 <= self.components <= 4096 or type(self.max_feature_bytes) is not int or self.max_feature_bytes < 1:
            raise ValueError('Require 1..4096 random features and positive memory limit')
        a,qa,b,qb = (np.asarray(value,float) for value in (x_distill,q_distill,x_gate,q_gate))
        if a.ndim != 2 or b.ndim != 2 or qa.ndim != 2 or qb.ndim != 2 or not 8 <= len(a) <= 20000 or not 4 <= len(b) <= 20000 or not 1 <= a.shape[1] <= 64 or a.shape[1] != b.shape[1] or len(qa)!=len(a) or len(qb)!=len(b) or qa.shape[1] != qb.shape[1] or not qa.shape[1]:
            raise ValueError('Require compatible numeric input/feature pairs and independent gate rows')
        if any(not np.isfinite(value).all() for value in (qa,qb)) or np.isinf(a).any() or np.isinf(b).any():
            raise ValueError('Require finite teacher features; missing raw features use train-only imputation')
        if max(qa.nbytes,qb.nbytes,len(a)*(self.components if self.kind=='rff_ridge' else a.shape[1])*8)>self.max_feature_bytes:
            raise ValueError('Distillation component matrix exceeds max_feature_bytes')
        first,second = _ids(distill_ids,len(a),'distill_ids'),_ids(gate_ids,len(b),'gate_ids')
        if set(first)&set(second) or set(second)&set(map(str,teacher_used_ids)):
            raise ValueError('Gate records overlap previously used records')
        spec = copy.deepcopy(head_spec)
        if spec.get('task') not in ('classification','regression') or spec.get('kind') not in ('sklearn','logits'):
            raise ValueError('Unsupported classical head specification')
        if spec['kind']=='sklearn' and (type(spec.get('head')) not in (Ridge,LogisticRegression,SGDClassifier,SGDRegressor,SVC,KernelRidge) or callable(getattr(spec['head'],'kernel',None)) or getattr(spec['head'],'kernel',None)=='precomputed'):
            raise ValueError('Head must be a supported wholly classical predictor')
        if spec.get('feature_scaler') is not None and type(spec['feature_scaler']) is not StandardScaler:
            raise ValueError('Feature scaling must be a fitted classical StandardScaler')
        if spec['task']=='classification':
            classes = np.asarray(spec.get('classes'))
            if classes.ndim!=1 or len(classes)<2 or len(np.unique(classes))!=len(classes):
                raise ValueError('Classification needs an ordered unique class-label mapping')
            spec['classes'] = classes
            if spec['kind']=='sklearn' and not np.array_equal(getattr(spec['head'],'classes_',None),np.arange(len(classes))):
                raise ValueError('Classical classification head must predict integer indices into classes')
        required = ['max_feature_rmse','max_feature_relative_l2','max_task_degradation']
        required += ['max_prediction_disagreement','min_teacher_balanced_accuracy'] if spec['task']=='classification' else ['max_prediction_rmse','max_teacher_rmse']
        if not isinstance(gates,dict) or any(key not in gates or type(gates[key]) not in (int,float) or not np.isfinite(gates[key]) or gates[key]<0 for key in required):
            raise ValueError('Provide explicit finite nonnegative feature, prediction and teacher/task quality gates')
        truth = np.asarray(y_gate)
        if truth.ndim != 1 or len(truth)!=len(b):
            raise ValueError('Provide gate targets for downstream utility checks')
        if spec['task']=='classification' and set(truth.tolist())!=set(spec['classes'].tolist()):
            raise ValueError('Gate labels must cover every teacher class and contain no unseen classes')
        stages = [SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler()]
        if self.kind == 'rff_ridge':
            stages.append(RBFSampler(gamma=self.gamma,n_components=self.components,random_state=self.seed))
        stages.append(Ridge(alpha=self.alpha))
        self.mapper_ = make_pipeline(*stages).fit(a,qa)
        estimate = np.asarray(self.mapper_.predict(b))
        teacher_predictions = _head_predict(spec,qb)
        student_predictions = _head_predict(spec,estimate)
        if teacher_predictions.shape!=(len(b),) or student_predictions.shape!=(len(b),):
            raise ValueError('Head must produce one scalar/label per record')
        error = float(np.linalg.norm(estimate-qb))
        variation = float(np.linalg.norm(qb-qa.mean(axis=0)))
        relative = error/variation if variation>1e-12 else (0. if error<=1e-12 else None)
        metrics = dict(feature_rmse=float(np.sqrt(np.mean((estimate-qb)**2))),feature_relative_l2=relative)
        if spec['task'] == 'classification':
            metrics.update(prediction_disagreement=float(np.mean(student_predictions!=teacher_predictions)),
                           teacher_score=float(balanced_accuracy_score(truth,teacher_predictions)),
                           student_score=float(balanced_accuracy_score(truth,student_predictions)))
            prediction_pass = metrics['prediction_disagreement'] <= gates['max_prediction_disagreement']
            quality_pass = metrics['teacher_score'] >= gates['min_teacher_balanced_accuracy']
            degradation = metrics['teacher_score']-metrics['student_score']
        else:
            truth = truth.astype(float)
            if not np.isfinite(truth).all():
                raise ValueError('Gate regression targets must be finite')
            metrics.update(prediction_rmse=float(np.sqrt(np.mean((teacher_predictions-student_predictions)**2))),
                           teacher_score=float(np.sqrt(np.mean((truth-teacher_predictions)**2))),
                           student_score=float(np.sqrt(np.mean((truth-student_predictions)**2))))
            prediction_pass = metrics['prediction_rmse'] <= gates['max_prediction_rmse']
            quality_pass = metrics['teacher_score'] <= gates['max_teacher_rmse']
            degradation = metrics['student_score']-metrics['teacher_score']
        checks = dict(feature_rmse=metrics['feature_rmse']<=gates['max_feature_rmse'],
                      feature_relative_l2=relative is not None and relative<=gates['max_feature_relative_l2'],
                      prediction_agreement=prediction_pass,teacher_quality=quality_pass,
                      downstream_degradation=degradation<=gates['max_task_degradation'])
        self.head_spec_,self.n_features_in_ = spec,a.shape[1]
        self.accepted_,self.fitted_ = bool(all(checks.values())),True
        self.gate_report_ = dict(accepted=self.accepted_,checks=checks,metrics=metrics,thresholds=copy.deepcopy(gates),
                                 distillation_rows=len(a),gate_rows=len(b),gate_ids=second,
                                 distillation_feature_sha256=hashlib.sha256(np.ascontiguousarray(qa).tobytes()).hexdigest(),
                                 gate_feature_sha256=hashlib.sha256(np.ascontiguousarray(qb).tobytes()).hexdigest(),
                                 teacher_acquisition=teacher_acquisition or {'source':'caller-supplied measured feature pairs; acquisition cost must be supplied'},
                                 student_inference_quantum_queries=0,
                                 scope='Held-out empirical feature/prediction fidelity, not a universal or confidence-certified guarantee')
        return self

    def predict_features(self,x):
        check_is_fitted(self,'fitted_')
        x = np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=self.n_features_in_ or not len(x) or np.isinf(x).any():
            raise ValueError('Require matching numeric features')
        return np.asarray(self.mapper_.predict(x))

    def predict(self,x):
        check_is_fitted(self,'fitted_')
        return _head_predict(self.head_spec_,self.predict_features(x))

    def export(self,path):
        check_is_fitted(self,'fitted_')
        if not self.accepted_:
            raise ValueError('Held-out fidelity/utility gate failed; student export is blocked')
        path = Path(path)
        if path.exists() or path.with_suffix('.json').exists():
            raise ValueError('Preserve prior student artifacts; use a new path')
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(self,path)
        manifest = dict(self.gate_report_,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        return manifest
