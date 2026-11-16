"""Fit/predict references for explicit exploratory project comparisons.

Exact native simulation, full feature coverage, local readouts. Not replicas of
the ten success cases and not a theorem that the chosen circuit will train well.
"""
import importlib
import time
import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import softmax, logsumexp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils.validation import check_is_fitted
from .portable_contract import local_features

METHODS = ('quantum_features', 'projected_kernel', 'fidelity_kernel', 'trainable_local')


class ResourceLimit(RuntimeError):
    pass


def preprocessing():
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())


def bounded_angles(x):
    return np.pi*np.tanh(np.asarray(x)/3.)


class QuantumEstimator(BaseEstimator):
    """Numeric binary/multiclass classification or scalar regression.

    Missing values are train-only median imputed. No PCA is performed. Counter
    limits include validation and test inference, not only fitting.
    """
    def __init__(self, method='quantum_features', task='classification', framework='qiskit',
                 qubits=2, layers=1, scale=1., regularization=1., steps=12,
                 learning_rate=.03, seed=7, entangle=True,
                 max_state_preparations=50000, max_seconds=120.,optimizer='adam',metric_regularization=.001):
        self.method, self.task, self.framework = method,task,framework
        self.qubits, self.layers, self.scale = qubits,layers,scale
        self.regularization, self.steps = regularization,steps
        self.learning_rate, self.seed, self.entangle = learning_rate,seed,entangle
        self.max_state_preparations, self.max_seconds = max_state_preparations,max_seconds
        self.optimizer,self.metric_regularization = optimizer,metric_regularization

    def _circuit_metric(self,angles,theta):
        from qml_tricks.geometry import fubini_study_metric
        before = self.resource_['state_preparations']
        metric = np.zeros((theta.size,theta.size))
        for row in angles:
            metric += fubini_study_metric(lambda flat:self._states([row],flat.reshape(theta.shape))[0],theta.ravel())
        self.resource_['metric_state_preparations'] += self.resource_['state_preparations']-before
        return metric/len(angles)

    def _states(self, angles, theta=None):
        native = importlib.import_module('QML_utility.models.portable_'+self.framework)
        states = []
        for row in angles:
            if self.resource_['state_preparations'] >= self.max_state_preparations or time.perf_counter()-self._started > self.max_seconds:
                raise ResourceLimit('Per-trial state-preparation or cooperative runtime budget exhausted')
            self.resource_['state_preparations'] += 1
            states.append(native.state(row,self.theta_ if theta is None else theta,self.scale,self.entangle))
        return np.asarray(states)

    def _features(self, angles, theta=None):
        return np.array([local_features(state) for state in self._states(angles,theta)])

    def _loss_gradient(self, angles, targets, parameters):
        p = self.theta_.size
        theta = parameters[:p].reshape(self.theta_.shape)
        weights = parameters[p:p+self.weights_.size].reshape(self.weights_.shape)
        bias = parameters[-len(self.bias_):]
        features = self._features(angles,theta)
        logits = features@weights+bias
        if self.task == 'classification':
            residual = (softmax(logits,axis=1)-targets)/len(angles)
            loss = np.mean(logsumexp(logits,axis=1)-np.sum(targets*logits,axis=1))
        else:
            residual = 2*(logits-targets)/len(angles)
            loss = np.mean((logits-targets)**2)
        # L2 acts on the classical head. Pauli shifts differentiate expectations,
        # then the nonlinear loss uses this classical chain rule.
        loss += self.regularization*np.sum(weights**2)/2
        dfeatures = residual@weights.T
        dtheta = np.zeros(p)
        for j in range(p):
            plus, minus = theta.copy(),theta.copy()
            plus.flat[j] += np.pi/2; minus.flat[j] -= np.pi/2
            derivative = (self._features(angles,plus)-self._features(angles,minus))/2
            dtheta[j] = np.sum(dfeatures*derivative)
        gradient = np.concatenate([dtheta,(features.T@residual+self.regularization*weights).ravel(),residual.sum(axis=0)])
        return float(loss),gradient

    def fit(self, x, y):
        if hasattr(self,'fitted_'):
            del self.fitted_
        if self.method not in METHODS or self.task not in ('classification','regression') or self.framework not in ('qiskit','pennylane','cirq'):
            raise ValueError('Unsupported method/task/framework')
        if self.optimizer not in ('adam','gd','qng') or not np.isfinite(self.metric_regularization) or self.metric_regularization<=0:
            raise ValueError('Require adam/gd/qng and positive metric_regularization')
        if self.method!='trainable_local' and self.optimizer!='adam':
            raise ValueError('Circuit optimizers apply only to trainable_local, not fixed-map heads')
        if type(self.qubits) is not int or not 1 <= self.qubits <= 6 or type(self.layers) is not int or not 1 <= self.layers <= 4:
            raise ValueError('Reference caps: 1..6 qubits, 1..4 layers')
        if type(self.steps) is not int or not 1 <= self.steps <= 200 or not np.isfinite(self.regularization) or self.regularization <= 0 or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError('Require positive regularization, 1..200 steps and 0 < learning_rate <= 1')
        if type(self.max_state_preparations) is not int or self.max_state_preparations < 1 or not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('Require positive state-preparation and runtime budgets')
        x, y = np.asarray(x,float),np.asarray(y)
        if x.ndim != 2 or not 1 <= x.shape[1] <= 64 or not 4 <= len(x) <= 512 or np.isinf(x).any() or y.ndim != 1 or len(y) != len(x):
            raise ValueError('Require 4..512 numeric rows, 1..64 features, matching targets; NaN features are imputed')
        self._started = time.perf_counter()
        self.resource_ = dict(state_preparations=0,measurement_shots=0,qpu_jobs=0,
                              classical_kernel_pairs=0,statevector_bytes=16*2**self.qubits,
                              physical_shots_estimated=False,metric_state_preparations=0)
        self.preprocessor_ = preprocessing()
        angles = bounded_angles(self.preprocessor_.fit_transform(x))
        self.n_features_in_ = x.shape[1]
        self.theta_ = np.random.default_rng(self.seed).normal(0,.01 if self.method == 'trainable_local' else .3,
                                                           (self.layers,self.qubits,2))
        if self.task == 'classification':
            self.encoder_ = LabelEncoder().fit(y)
            self.classes_ = self.encoder_.classes_
            if len(self.classes_) < 2:
                raise ValueError('Classification needs at least two training classes')
            target = self.encoder_.transform(y)
        else:
            target = y.astype(float)
            if not np.isfinite(target).all():
                raise ValueError('Regression targets must be finite')
        self.trace_ = []
        self.diagnostics_ = {}
        if self.method == 'trainable_local':
            if self.task == 'classification':
                target = np.eye(len(self.classes_))[target]
            else:
                self.target_mean_, self.target_scale_ = float(target.mean()),max(float(target.std()),1e-12)
                target = ((target-self.target_mean_)/self.target_scale_)[:,None]
            self.weights_ = np.random.default_rng(self.seed+1).normal(0,.1,(3*self.qubits,target.shape[1]))
            self.bias_ = np.zeros(target.shape[1])
            parameters = np.concatenate([self.theta_.ravel(),self.weights_.ravel(),self.bias_])
            first,second = np.zeros_like(parameters),np.zeros_like(parameters)
            for step in range(1,self.steps+1):
                loss,gradient = self._loss_gradient(angles,target,parameters)
                self.trace_.append(dict(step=step,training_loss=loss,gradient_norm=float(np.linalg.norm(gradient)),
                                       circuit_gradient_norm=float(np.linalg.norm(gradient[:self.theta_.size]))))
                if not np.isfinite(loss) or not np.isfinite(gradient).all():
                    raise ValueError('Non-finite optimization trace')
                if self.optimizer=='qng':
                    from qml_tricks.geometry import natural_gradient_direction
                    metric = self._circuit_metric(angles,parameters[:self.theta_.size].reshape(self.theta_.shape))
                    direction = natural_gradient_direction(metric,gradient[:self.theta_.size],self.metric_regularization)
                    self.trace_[-1].update(optimizer='qng',metric_minimum_eigenvalue=float(np.linalg.eigvalsh(metric)[0]),
                                           regularized_condition_number=float(np.linalg.cond(metric+self.metric_regularization*np.eye(len(metric)))))
                    parameters[:self.theta_.size] -= self.learning_rate*direction
                elif self.optimizer=='gd':
                    parameters[:self.theta_.size] -= self.learning_rate*gradient[:self.theta_.size]
                adam_gradient = gradient.copy()
                if self.optimizer!='adam':
                    adam_gradient[:self.theta_.size]=0
                first = .9*first+.1*adam_gradient; second = .999*second+.001*adam_gradient**2
                parameters -= self.learning_rate*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
            p,w = self.theta_.size,self.weights_.size
            self.theta_ = parameters[:p].reshape(self.theta_.shape)
            self.weights_ = parameters[p:p+w].reshape(self.weights_.shape)
            self.bias_ = parameters[p+w:]
            features = self._features(angles)
            self.diagnostics_ = dict(local_feature_variance=np.var(features,axis=0).tolist(),
                                     local_feature_rank=int(np.linalg.matrix_rank(features-features.mean(axis=0))))
        elif self.method == 'fidelity_kernel':
            self.train_states_ = self._states(angles)
            kernel = np.abs(self.train_states_.conj()@self.train_states_.T)**2
            self.resource_['classical_kernel_pairs'] += len(x)**2
            self.resource_['cached_train_state_bytes'] = self.train_states_.nbytes
            self.resource_['dense_train_gram_bytes'] = kernel.nbytes
            off_diagonal = kernel[~np.eye(len(x),dtype=bool)]
            self.diagnostics_ = dict(kernel_off_diagonal_std=float(off_diagonal.std()),
                                     gram_participation_rank=float(np.trace(kernel)**2/np.sum(kernel**2)))
            self.head_ = SVC(C=1/self.regularization,kernel='precomputed') if self.task == 'classification' else KernelRidge(alpha=self.regularization,kernel='precomputed')
            self.head_.fit(kernel,target)
        else:
            features = self._features(angles)
            self.diagnostics_ = dict(local_feature_variance=np.var(features,axis=0).tolist(),
                                     local_feature_rank=int(np.linalg.matrix_rank(features-features.mean(axis=0))))
            self.resource_['training_feature_matrix_bytes'] = features.nbytes
            if self.method == 'projected_kernel':
                self.head_ = SVC(C=1/self.regularization,kernel='rbf',gamma='scale') if self.task == 'classification' else KernelRidge(alpha=self.regularization,kernel='rbf',gamma=1/features.shape[1])
            else:
                self.head_ = LogisticRegression(C=1/self.regularization,max_iter=1000) if self.task == 'classification' else Ridge(alpha=self.regularization)
            self.head_.fit(features,target)
        self.fitted_ = True
        self.resource_['feature_upload_rotations_per_state'] = 2*x.shape[1]*self.layers
        self.resource_['entangling_gates_per_state'] = int(self.entangle)*(self.qubits-1)*int(np.ceil(x.shape[1]/self.qubits))*self.layers
        self.resource_['local_measurement_settings_if_sampled'] = 0 if self.method == 'fidelity_kernel' else 3
        return self

    def transform_quantum_features(self, x):
        """Acquire local features from a frozen fitted map; consumes preparation budget."""
        check_is_fitted(self,'fitted_')
        if self.method == 'fidelity_kernel':
            raise ValueError('Fidelity-kernel heads do not consume this local feature map')
        self._started = time.perf_counter()
        x = np.asarray(x,float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not len(x) or np.isinf(x).any():
            raise ValueError('Require matching numeric input features')
        return self._features(bounded_angles(self.preprocessor_.transform(x)))

    def predict(self, x):
        check_is_fitted(self,'fitted_')
        self._started = time.perf_counter()  # new operation; idle/serialization time is not compute
        x = np.asarray(x,float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not len(x) or np.isinf(x).any():
            raise ValueError('Prediction requires matching numeric features')
        angles = bounded_angles(self.preprocessor_.transform(x))
        if self.method == 'fidelity_kernel':
            states = self._states(angles)
            kernel = np.abs(states.conj()@self.train_states_.T)**2
            self.resource_['classical_kernel_pairs'] += len(x)*len(self.train_states_)
            predictions = self.head_.predict(kernel)
        else:
            features = self._features(angles)
            if self.method == 'trainable_local':
                logits = features@self.weights_+self.bias_
                predictions = logits.argmax(axis=1) if self.task == 'classification' else logits[:,0]*self.target_scale_+self.target_mean_
            else:
                predictions = self.head_.predict(features)
        return self.encoder_.inverse_transform(np.asarray(predictions,int)) if self.task == 'classification' else np.asarray(predictions,float)
