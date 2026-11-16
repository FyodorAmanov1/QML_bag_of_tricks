"""Batched native fixed features and a cached, incremental classical head.

Uses the existing native circuit unchanged. This is an engineering adapter,
not a newly advantageous quantum model or an out-of-core CSV implementation.
"""
import time
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.validation import check_is_fitted
from .estimators import QuantumEstimator, ResourceLimit, preprocessing, bounded_angles
from .portable_contract import local_features


class BatchedQuantumFeatures(QuantumEstimator):
    def __init__(self, task='classification', framework='qiskit', qubits=2, layers=1,
                 scale=1., regularization=.001, epochs=10, batch_size=128, seed=7,
                 max_state_preparations=100000, max_seconds=300.,
                 max_feature_bytes=64*1024*1024, entangle=True):
        super().__init__(method='quantum_features',task=task,framework=framework,
                         qubits=qubits,layers=layers,scale=scale,regularization=regularization,
                         seed=seed,max_state_preparations=max_state_preparations,
                         max_seconds=max_seconds,entangle=entangle)
        self.epochs,self.batch_size = epochs,batch_size
        self.max_feature_bytes = max_feature_bytes

    def _features(self, angles, theta=None):
        required = len(angles)*3*self.qubits*8
        if required > self.max_feature_bytes:
            raise ResourceLimit('Local-feature matrix exceeds max_feature_bytes')
        features = np.empty((len(angles),3*self.qubits))
        for start in range(0,len(angles),self.batch_size):
            block = angles[start:start+self.batch_size]
            states = self._states(block,theta)
            features[start:start+len(block)] = [local_features(state) for state in states]
            self.resource_['max_state_batch_rows'] = max(self.resource_['max_state_batch_rows'],len(block))
        self.resource_['largest_feature_matrix_bytes'] = max(self.resource_['largest_feature_matrix_bytes'],required)
        return features

    def fit(self, x, y):
        if hasattr(self,'fitted_'):
            del self.fitted_
        if self.task not in ('classification','regression') or self.framework not in ('qiskit','pennylane','cirq'):
            raise ValueError('Unsupported task/framework')
        if type(self.qubits) is not int or not 1 <= self.qubits <= 6 or type(self.layers) is not int or not 1 <= self.layers <= 4:
            raise ValueError('Require 1..6 qubits and 1..4 layers')
        if type(self.epochs) is not int or not 1 <= self.epochs <= 200 or type(self.batch_size) is not int or not 1 <= self.batch_size <= 1024:
            raise ValueError('Require 1..200 head epochs and 1..1024 state batch rows')
        if type(self.max_feature_bytes) is not int or self.max_feature_bytes < 1 or type(self.max_state_preparations) is not int or self.max_state_preparations < 1:
            raise ValueError('Require positive feature-memory and preparation limits')
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0 or not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError('Require positive runtime and regularization')
        x,y = np.asarray(x,float),np.asarray(y)
        if x.ndim != 2 or not 4 <= len(x) <= 20000 or not 1 <= x.shape[1] <= 64 or np.isinf(x).any() or y.ndim != 1 or len(y) != len(x):
            raise ValueError('Require 4..20000 rows, 1..64 numeric features and matching targets')
        if len(x)*3*self.qubits*8 > self.max_feature_bytes:
            raise ResourceLimit('Training feature cache exceeds max_feature_bytes')
        self._started = time.perf_counter()
        self.resource_ = dict(state_preparations=0,measurement_shots=0,qpu_jobs=0,
                              classical_kernel_pairs=0,dense_train_gram_bytes=0,
                              statevector_bytes=16*2**self.qubits,physical_shots_estimated=False,
                              max_state_batch_rows=0,largest_feature_matrix_bytes=0,
                              head_epochs=0,feature_acquisition_passes=1)
        self.preprocessor_ = preprocessing()
        angles = bounded_angles(self.preprocessor_.fit_transform(x))
        self.n_features_in_ = x.shape[1]
        self.theta_ = np.random.default_rng(self.seed).normal(0,.3,(self.layers,self.qubits,2))
        raw_features = self._features(angles)
        self.feature_scaler_ = StandardScaler().fit(raw_features)
        features = self.feature_scaler_.transform(raw_features)
        if self.task == 'classification':
            self.encoder_ = LabelEncoder().fit(y)
            self.classes_ = self.encoder_.classes_
            if len(self.classes_) < 2:
                raise ValueError('Need at least two training classes')
            targets = self.encoder_.transform(y)
            self.head_ = SGDClassifier(loss='log_loss',alpha=self.regularization,random_state=self.seed,average=True)
        else:
            targets = y.astype(float)
            if not np.isfinite(targets).all():
                raise ValueError('Require finite regression targets')
            self.target_mean_,self.target_scale_ = float(targets.mean()),max(float(targets.std()),1e-12)
            targets = (targets-self.target_mean_)/self.target_scale_
            self.head_ = SGDRegressor(alpha=self.regularization,random_state=self.seed,average=True)
        self.trace_ = []
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            order = rng.permutation(len(x))
            for start in range(0,len(x),self.batch_size):
                if time.perf_counter()-self._started > self.max_seconds:
                    raise ResourceLimit('Cooperative training-operation budget exhausted')
                indices = order[start:start+self.batch_size]
                if self.task == 'classification':
                    self.head_.partial_fit(features[indices],targets[indices],classes=np.arange(len(self.classes_)))
                else:
                    self.head_.partial_fit(features[indices],targets[indices])
            prediction = self.head_.predict(features)
            error = np.mean(prediction != targets) if self.task == 'classification' else np.mean((prediction-targets)**2)
            self.trace_.append(dict(epoch=epoch+1,training_error=float(error)))
            self.resource_['head_epochs'] += 1
        self.diagnostics_ = dict(local_feature_variance=np.var(raw_features,axis=0).tolist(),
                                 local_feature_rank=int(np.linalg.matrix_rank(raw_features-raw_features.mean(axis=0))))
        self.resource_.update(feature_upload_rotations_per_state=2*x.shape[1]*self.layers,
                              entangling_gates_per_state=int(self.entangle)*(self.qubits-1)*int(np.ceil(x.shape[1]/self.qubits))*self.layers,
                              local_measurement_settings_if_sampled=3)
        self.fitted_ = True
        return self

    def predict(self, x):
        check_is_fitted(self,'fitted_')
        raw = self.transform_quantum_features(x)
        prediction = self.head_.predict(self.feature_scaler_.transform(raw))
        if self.task == 'classification':
            return self.encoder_.inverse_transform(prediction.astype(int))
        return prediction*self.target_scale_+self.target_mean_
