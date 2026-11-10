"""Small complete estimators. Full-state access is explicitly simulator-only."""
from importlib import import_module
import time
import numpy as np
from scipy.special import expit, softmax
from .methods import parameter_shift, spsa

MODELS = ('reuploading','frozen_features','fidelity_kernel','projected_kernel',
          'classical_pretraining','classical_warm_start','nystrom','kernel_surrogate','spsa')


class QMLModel:
    def __init__(self, method='frozen_features', task='classification', framework='qiskit',
                 n_qubits=2, depth=1, seed=7, shots=None, steps=5, regularization=.1, landmarks=8):
        if method not in MODELS or task not in ('classification','regression'): raise ValueError('Unsupported method/task')
        if not 1 <= n_qubits <= 6 or not 1 <= depth <= 4 or steps < 0: raise ValueError('Invalid bounded learning configuration')
        if shots is not None and shots < 2: raise ValueError('shots must be None or >=2')
        if regularization <= 0 or landmarks < 1: raise ValueError('Positive regularization and landmarks required')
        self.method,self.task,self.framework = method,task,framework
        self.n,self.depth,self.seed,self.shots,self.steps = n_qubits,depth,seed,shots,steps
        self.regularization,self.landmarks = regularization,landmarks

    def _state(self, x):
        self.resources_['state_preparations'] += 1
        return import_module(f'qml_tricks.learning.backends.{self.framework}').state(self.theta_,x,self.n,self.depth)

    def _features(self, x):
        out=[]
        z = 1-2*((np.arange(2**self.n)[:,None] >> np.arange(self.n-1,-1,-1)) & 1)
        for row in x:
            key=tuple(row)
            if key not in self.cache_:
                state=self._state(row); p=np.abs(state)**2; p/=p.sum()
                if self.shots:
                    p=self.rng_.multinomial(self.shots,p)/self.shots
                    self.resources_['measurement_shots'] += self.shots
                self.cache_[key]=p@z
            out.append(self.cache_[key])
        return np.asarray(out)

    def _density(self,x):
        rows=[]
        for row in x:
            psi=self._state(row); rho=np.outer(psi,psi.conj()).reshape(-1)
            rows.append(np.r_[rho.real,rho.imag])
        return np.array(rows)

    def _kernel(self,x,y):
        a=np.array([self._state(row) for row in x]); b=np.array([self._state(row) for row in y])
        k=np.clip(np.abs(a.conj()@b.T)**2,0,1)
        self.resources_['kernel_pair_evaluations'] += k.size
        if self.shots:
            k=self.rng_.binomial(self.shots,k)/self.shots
            self.resources_['measurement_shots'] += self.shots*k.size
        return k

    def _transform(self,x):
        z=self.scaler_.transform(np.asarray(x,float))
        if self.method=='classical_pretraining':
            z=np.tanh(z@self.extractor_.coefs_[0]+self.extractor_.intercepts_[0])
        else: z=self.pca_.transform(z)
        return np.tanh(z)*np.pi

    def _output(self,theta,x,exact=False):
        state_fn=import_module(f'qml_tricks.learning.backends.{self.framework}').state
        z=1-2*((np.arange(2**self.n) >> (self.n-1)) & 1)
        result=[]
        for row in x:
            self.resources_['warm_start_state_preparations' if exact else 'state_preparations'] += 1
            p=np.abs(state_fn(theta,row,self.n,self.depth))**2; p/=p.sum()
            if self.shots and not exact:
                p=self.rng_.multinomial(self.shots,p)/self.shots
                self.resources_['measurement_shots'] += self.shots
            result.append(p@z)
        return np.array(result)

    def _loss_gradient(self,theta,x,y,exact=False):
        f=self._output(theta,x,exact)
        if self.task=='classification':
            # Logistic loss on a quantum expectation, with the classical chain rule.
            prob=expit(2*f)
            loss=np.mean(np.logaddexp(0,2*f)-y*2*f)
            chain=2*(prob-y)/len(y)
        else:
            loss=np.mean((f-y)**2);chain=2*(f-y)/len(y)
        derivative=parameter_shift(lambda t:self._output(t,x,exact),theta)
        return float(loss), derivative@chain

    def fit(self,x,y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier,MLPRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.svm import SVC
        start=time.perf_counter();x=np.asarray(x,float);y=np.asarray(y)
        if x.ndim!=2 or not np.isfinite(x).all() or len(x)!=len(y):raise ValueError('Expected finite X matrix and matching labels')
        if len(x)>512:raise ValueError('Reference learners are capped at 512 training examples')
        if min(x.shape)<self.n:raise ValueError('n_qubits exceeds train samples or feature dimension')
        self.rng_=np.random.default_rng(self.seed);self.theta_=self.rng_.normal(0,.1,self.n*self.depth)
        self.cache_={};self.resources_=dict(state_preparations=0,measurement_shots=0,kernel_pair_evaluations=0,warm_start_state_preparations=0)
        self.scaler_=StandardScaler().fit(x)
        if self.task=='classification':
            self.classes_,yi=np.unique(y,return_inverse=True)
            if len(self.classes_)!=2:raise ValueError('Current learning recipes support binary classification')
            targets=np.eye(2)[yi]
        else:
            yi=np.asarray(y,float);self.y_mean_=yi.mean();self.y_scale_=max(yi.std(),1e-8)
            targets=(yi-self.y_mean_)/self.y_scale_
        if self.method=='classical_pretraining':
            cls=MLPClassifier if self.task=='classification' else MLPRegressor
            self.extractor_=cls(hidden_layer_sizes=(self.n,),activation='tanh',solver='lbfgs',max_iter=200,random_state=self.seed)
            self.extractor_.fit(self.scaler_.transform(x),yi)
        else:self.pca_=PCA(n_components=self.n).fit(self.scaler_.transform(x))
        tx=self._transform(x)
        if self.method=='classical_pretraining':
            self.classical_control_=KernelRidge(alpha=self.regularization,kernel='linear').fit(tx,targets)
        if self.method in ('reuploading','classical_warm_start','spsa'):
            ty=yi if self.task=='classification' else targets
            if self.method=='classical_warm_start':
                for _ in range(self.steps):
                    _,g=self._loss_gradient(self.theta_,tx,ty,exact=True);self.theta_-=.1*g
            if self.method=='spsa':
                def loss(t):
                    f=self._output(t,tx)
                    return np.mean(np.logaddexp(0,2*f)-ty*2*f) if self.task=='classification' else np.mean((f-ty)**2)
                self.theta_=spsa(loss,self.theta_,self.steps,self.seed)
            else:
                for _ in range(self.steps):
                    _,g=self._loss_gradient(self.theta_,tx,ty);self.theta_-=.1*g
        else:
            if self.method in ('fidelity_kernel','nystrom'):
                size=len(tx) if self.method=='fidelity_kernel' else min(self.landmarks,len(tx))
                self.landmarks_=tx[self.rng_.choice(len(tx),size,replace=False)]
                w=self._kernel(self.landmarks_,self.landmarks_);w=(w+w.T)/2
                vals,vecs=np.linalg.eigh(w);keep=vals>1e-9
                if not keep.any():raise ValueError('Landmark Gram matrix has no positive eigenspace')
                self.projection_=vecs[:,keep]/np.sqrt(vals[keep])
                self.resources_['gram_negative_eigenvalue_mass']=float(-vals[vals<0].sum())
                features=self._kernel(tx,self.landmarks_)@self.projection_
            elif self.method=='kernel_surrogate':
                if self.shots:raise ValueError('Density-embedding teacher is exact-simulator-only; use shots=None')
                teacher=self._density(tx)
                self.student_=KernelRidge(alpha=self.regularization,kernel='rbf',gamma=1/self.n).fit(tx,teacher)
                features=self.student_.predict(tx)
            else:features=self._features(tx)
            if self.task=='classification':
                self.head_=SVC(C=1/self.regularization,kernel='rbf' if self.method=='projected_kernel' else 'linear',gamma=1/self.n)
                if self.method in ('frozen_features','classical_pretraining'):
                    self.head_=KernelRidge(alpha=self.regularization,kernel='linear');self.head_.fit(features,targets)
                else:self.head_.fit(features,yi)
            else:
                self.head_=KernelRidge(alpha=self.regularization,kernel='rbf' if self.method=='projected_kernel' else 'linear',gamma=1/self.n).fit(features,targets)
        self.resources_.update(training_seconds=time.perf_counter()-start,statevector_bytes=16*2**self.n,
            state_preparations_meaning='native simulator executions, not hardware state copies',
            full_density_teacher=self.method=='kernel_surrogate',
            density_embedding_bytes=(len(x)*16*4**self.n if self.method=='kernel_surrogate' else 0),
            simulator_only=True, data_access='classical feature vectors or known simulator preparation descriptions')
        return self

    def _readout(self,x):
        if not hasattr(self,'theta_'):raise ValueError('Call fit first')
        tx=self._transform(x)
        if self.method in ('reuploading','classical_warm_start','spsa'):return self._output(self.theta_,tx)
        if self.method in ('fidelity_kernel','nystrom'):features=self._kernel(tx,self.landmarks_)@self.projection_
        elif self.method=='kernel_surrogate':features=self.student_.predict(tx)
        else:features=self._features(tx)
        if self.task=='classification' and hasattr(self.head_,'decision_function'):
            score=self.head_.decision_function(features);return np.column_stack([-score,score])
        return self.head_.predict(features)

    def predict_proba(self,x):
        if self.task!='classification':raise ValueError('Classification only')
        scores=self._readout(x)
        if scores.ndim==1:
            p=expit(2*scores);return np.column_stack([1-p,p])
        return softmax(scores,axis=1)  # scores, not calibrated probabilities

    def predict(self,x):
        if self.task=='classification':return self.classes_[np.argmax(self.predict_proba(x),axis=1)]
        return self._readout(x)*self.y_scale_+self.y_mean_

    def quantum_features(self,x):
        """Measured local-Z features of this fitted, frozen native encoder.

        Acquisition can execute the quantum backend. This is not the classical
        student inference path; feature ordering is fixed by wire number.
        """
        if not hasattr(self,'theta_'):raise ValueError('Fit the teacher first')
        return self._features(self._transform(x))

    def audit_surrogate(self,x):
        """Separate held-out teacher acquisition, never used to refit the student."""
        if self.method!='kernel_surrogate':raise ValueError('Surrogate only')
        tx=self._transform(x);before=self.resources_['state_preparations']
        target=self._density(tx);student=self.student_.predict(tx)
        delta=np.linalg.norm(target-student,axis=1)
        return dict(teacher_audit_preparations=self.resources_['state_preparations']-before,
                    embedding_error_max=float(delta.max()),kernel_rmse=float(np.sqrt(np.mean((target@target.T-student@student.T)**2))),
                    conditional_kernel_error_bound=float(2*delta.max()+delta.max()**2),
                    out_of_distribution_guarantee=False)

    def predict_classical_control(self,x):
        """Classical head on the SAME frozen pretrained hidden representation."""
        if self.method!='classical_pretraining':raise ValueError('Pretrained model only')
        values=self.classical_control_.predict(self._transform(x))
        return self.classes_[np.argmax(values,axis=1)] if self.task=='classification' else values*self.y_scale_+self.y_mean_

    def save(self,path):
        """Pickle-based artifact: load only trusted locally produced models."""
        import joblib
        joblib.dump(self,path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)
