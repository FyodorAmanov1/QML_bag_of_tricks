"""Offline quantum-feature distillation following Kipu's public protocol.

This implements the offline surrogate protocol, not Kipu's DQFE circuit or SDK.
Actual Kipu outputs can be supplied directly through fit_from_pairs.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error


class ClassicalFeatureStudent:
    """Deployable classical artifact; no quantum teacher or backend reference."""
    def __init__(self, surrogate, head, classes, validation):
        self.surrogate=surrogate;self.head=head;self.classes_=classes
        self.validation_=validation

    def transform(self,x):return self.surrogate.predict(np.asarray(x,float))

    def predict_proba(self,x):
        if self.classes_ is None:raise ValueError('Classification only')
        return softmax(self.head.predict(self.transform(x)),axis=1)

    def predict(self,x):
        scores=self.head.predict(self.transform(x))
        return self.classes_[np.argmax(scores,axis=1)] if self.classes_ is not None else scores

    def save(self,path):
        import joblib
        joblib.dump(self,path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


class OfflineFeatureSurrogate:
    """Fit on quantum input/output pairs; validate against a frozen teacher head.

    Ridge matches the surrogate family named in Kipu's public example.
    Optional polynomial expansion is a local engineering extension. Neither
    configuration carries a blanket high-fidelity generalization guarantee.
    """
    def __init__(self,task='classification',alpha=1e-3,degree=1,
                 min_agreement=.95,max_accuracy_drop=.02,max_relative_feature_rmse=.2,student_kind='ridge'):
        if task not in ('classification','regression') or alpha<=0 or degree not in (1,2,3):raise ValueError('Invalid configuration')
        if not 0<=min_agreement<=1 or max_accuracy_drop<0 or max_relative_feature_rmse<0:raise ValueError('Invalid validation thresholds')
        self.task,self.alpha,self.degree=task,alpha,degree
        if student_kind not in ('ridge','rbf_ridge'):raise ValueError('Unknown classical student')
        self.student_kind=student_kind
        self.min_agreement,self.max_accuracy_drop,self.max_relative_feature_rmse=min_agreement,max_accuracy_drop,max_relative_feature_rmse

    def fit_from_pairs(self,x_train,q_train,y_train,x_validation,q_validation,y_validation):
        """All validation pairs must be disjoint from fitting inputs.

        q arrays can be finite-shot features obtained from Kipu or another
        frozen teacher. They are feature vectors, not density matrices.
        """
        x,q=np.asarray(x_train,float),np.asarray(q_train,float)
        xv,qv=np.asarray(x_validation,float),np.asarray(q_validation,float)
        if x.ndim!=2 or q.ndim!=2 or xv.ndim!=2 or qv.ndim!=2 or len(x)!=len(q) or len(xv)!=len(qv) or x.shape[1]!=xv.shape[1] or q.shape[1]!=qv.shape[1]:raise ValueError('Incompatible input/feature matrices')
        if len(x)<2 or len(xv)<2 or not all(np.isfinite(a).all() for a in (x,q,xv,qv)):raise ValueError('Need finite train/validation pairs')
        train_rows={tuple(row) for row in x}
        if any(tuple(row) in train_rows for row in xv):raise ValueError('Overlapping train/validation inputs; supply independent held-out pairs')
        yt,yv=np.asarray(y_train),np.asarray(y_validation)
        if len(yt)!=len(x) or len(yv)!=len(xv):raise ValueError('Mismatched labels')
        self.classes_=None
        if self.task=='classification':
            self.classes_,encoded=np.unique(yt,return_inverse=True)
            if len(self.classes_)<2 or not np.isin(yv,self.classes_).all():raise ValueError('Invalid classification labels')
            targets=np.eye(len(self.classes_))[encoded]
        else:targets=yt.astype(float)
        # First train the downstream teacher predictor on actual quantum features.
        self.head_=Ridge(alpha=self.alpha).fit(q,targets)
        # Then learn only x -> q. Validation and test outputs never enter fitting.
        if self.student_kind=='ridge':
            self.surrogate_=make_pipeline(StandardScaler(),PolynomialFeatures(self.degree,include_bias=False),Ridge(alpha=self.alpha)).fit(x,q)
        else:
            from sklearn.kernel_ridge import KernelRidge
            self.surrogate_=make_pipeline(StandardScaler(),KernelRidge(alpha=self.alpha,kernel='rbf',gamma=1/x.shape[1])).fit(x,q)
        self.validation_=self.audit_pairs(xv,qv,yv)
        self.resources_=dict(teacher_training_pairs=len(x),teacher_validation_pairs=len(xv),feature_dimension=q.shape[1],
                             quantum_queries_at_student_inference=0,teacher_feature_access='supplied measured pairs or frozen native teacher',
                             validation_fitted=False,degree=self.degree,student_kind=self.student_kind)
        return self

    def fit_teacher(self,teacher,x_train,y_train,x_validation,y_validation):
        """Distill an ALREADY FITTED QMLModel; never refit its quantum angles.

        The reference adapter obtains local Z features using its native backend.
        It does not implement Kipu's proprietary DQFE teacher.
        """
        if not hasattr(teacher,'theta_'):raise ValueError('Fit the quantum teacher before distillation')
        self.teacher_=deepcopy(teacher)
        before=self.teacher_.resources_['state_preparations']
        q=self.teacher_.quantum_features(x_train)
        qv=self.teacher_.quantum_features(x_validation)
        self.fit_from_pairs(x_train,q,y_train,x_validation,qv,y_validation)
        self.resources_['native_teacher_acquisition_simulations']=self.teacher_.resources_['state_preparations']-before
        self.resources_['teacher_initial_training_resources']=deepcopy(teacher.resources_)
        self.resources_['teacher_parameters_frozen']=True
        return self

    def audit_pairs(self,x,q,y):
        """Evaluation only; never fits any model or changes acceptance thresholds."""
        q=np.asarray(q,float);pred=self.surrogate_.predict(np.asarray(x,float))
        rmse=float(np.sqrt(np.mean((q-pred)**2)))
        # Relative to per-feature centered variation, not the feature offsets.
        scale=max(float(np.sqrt(np.mean((q-q.mean(axis=0))**2))),1e-12)
        result=dict(feature_rmse=rmse,relative_feature_rmse=rmse/scale,
                    feature_r2=float(r2_score(q,pred,multioutput='uniform_average')),
                    acceptance_thresholds=dict(min_agreement=self.min_agreement,max_accuracy_drop=self.max_accuracy_drop,
                                               max_relative_feature_rmse=self.max_relative_feature_rmse),
                    confidence_statement='held-out empirical validation, no universal accuracy guarantee')
        original=self.head_.predict(q);student=self.head_.predict(pred)
        if self.classes_ is not None:
            a=self.classes_[np.argmax(original,axis=1)];b=self.classes_[np.argmax(student,axis=1)]
            agreement=float(np.mean(a==b));ta=float(accuracy_score(y,a));sa=float(accuracy_score(y,b))
            result.update(teacher_student_agreement=agreement,teacher_accuracy=ta,student_accuracy=sa,accuracy_drop=ta-sa,
                          passed=bool(agreement>=self.min_agreement and ta-sa<=self.max_accuracy_drop and rmse/scale<=self.max_relative_feature_rmse))
        else:
            result.update(teacher_mae=float(mean_absolute_error(y,original)),student_mae=float(mean_absolute_error(y,student)),
                          prediction_rmse=float(np.sqrt(np.mean((original-student)**2))),
                          passed=bool(rmse/scale<=self.max_relative_feature_rmse))
        return result

    def student(self):
        """Return the strictly classical model only after the validation gate passes."""
        if not self.validation_['passed']:raise ValueError('Surrogate failed held-out fidelity requirements; improve acquisition/model and revalidate on a fresh holdout')
        return ClassicalFeatureStudent(deepcopy(self.surrogate_),deepcopy(self.head_),self.classes_,deepcopy(self.validation_))

    def export_student(self,path):
        self.student().save(path)
        Path(str(path)+'.validation.json').write_text(json.dumps(self.validation_,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    def predict(self,x):return self.student().predict(x)

    def predict_proba(self,x):return self.student().predict_proba(x)
