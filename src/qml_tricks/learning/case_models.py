"""Runnable, reduced adaptations of three published model families.

These classes are reference engineering models, not reproductions of the paper
scores. Exact native simulator states are used; no hardware jobs are submitted.
"""
from __future__ import annotations

from importlib import import_module
import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TOPOLOGIES = ("full", "linear", "circular", "x", "y", "z")


def _backend(framework):
    if framework not in ("qiskit", "pennylane", "cirq"):
        raise ValueError("Choose qiskit, pennylane or cirq")
    return import_module(f"qml_tricks.learning.case_backends.{framework}")


def _matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite two-dimensional feature matrix")
    return x


def _labels(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("One label per example is required")
    classes, encoded = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("At least two classes are required")
    return classes, encoded


def _spsa(loss, initial, steps, learning_rate, seed):
    """Two perturbed evaluations plus one diagnostic evaluation per step."""
    if steps < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps must be nonnegative and learning_rate positive")
    rng = np.random.default_rng(seed)
    theta = initial.copy()
    best = theta.copy()
    best_loss = float(loss(theta))
    trace = [best_loss]
    for step in range(steps):
        perturbation = rng.choice([-1., 1.], size=theta.shape)
        c = .1 / (step + 1)**.101
        gradient = (loss(theta+c*perturbation)-loss(theta-c*perturbation)) / (2*c) * perturbation
        theta -= learning_rate / (step + 1)**.602 * np.clip(gradient, -5., 5.)
        value = float(loss(theta))
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite training objective")
        trace.append(value)
        if value < best_loss:
            best, best_loss = theta.copy(), value
    return best, trace


def classical_product_kernel(x, y):
    """Exact cheap control for RX/RY/RZ-on-|+> product-state kernels.

    Inputs are rotation angles, so the half-angle convention is explicit.
    """
    x, y = _matrix(x), _matrix(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    return np.prod(np.cos((x[:, None, :] - y[None, :, :])/2)**2, axis=2)


class MulticlassQuantumKernel:
    """One-vs-rest SVM with a selected IQP or product fidelity kernel.

    Related to Ding et al. (PRA 111, 062410). Bounded preprocessing, LIBSVM,
    regularization and split protocol are explicitly adaptation choices.
    """

    def __init__(self, framework="qiskit", topology="full", scale=.5, C=1.):
        if topology not in TOPOLOGIES or scale <= 0 or C <= 0:
            raise ValueError("Invalid kernel topology, encoding scale or C")
        self.framework, self.topology, self.scale, self.C = framework, topology, scale, C

    def angles(self, x):
        return np.pi*self.scale*np.tanh(self.scaler_.transform(_matrix(x)))

    def _states(self, x):
        angles = self.angles(x)
        self.state_evaluations_ += len(angles)
        return np.asarray([_backend(self.framework).kernel_state(row, self.topology) for row in angles])

    def fit(self, x, y):
        x = _matrix(x)
        _labels(y, len(x))
        self.n_qubits_ = x.shape[1]
        if self.n_qubits_ > 12:
            raise ValueError("Reference statevector model is limited to 12 features/qubits")
        self.scaler_ = StandardScaler().fit(x)
        self.state_evaluations_ = 0
        self.states_ = self._states(x)
        gram = np.abs(self.states_.conj() @ self.states_.T)**2
        self.model_ = OneVsRestClassifier(SVC(kernel="precomputed", C=self.C)).fit(gram, y)
        return self

    def predict(self, x):
        states = self._states(x)
        return self.model_.predict(np.abs(states.conj() @ self.states_.T)**2)


class AmplitudeQVCClassifier:
    """Amplitude encoding, RZ-RY-RZ/CZ layers and local-Z class scores.

    Related to West et al. (PRR 5, 023186). This reduced SPSA-trained model is
    not the paper's deep, Adam-trained image classifier or robustness result.
    """

    def __init__(self, framework="qiskit", depth=1, steps=8, learning_rate=.15, seed=7):
        if depth < 1:
            raise ValueError("depth must be positive")
        self.framework, self.depth, self.steps = framework, depth, steps
        self.learning_rate, self.seed = learning_rate, seed

    def _probabilities(self, x, flat_theta):
        x = _matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Feature dimension differs from training")
        theta = flat_theta.reshape(self.depth, self.n_qubits_, 3)
        scores = []
        indices = np.arange(2**self.n_qubits_)
        signs = np.asarray([1-2*((indices >> (self.n_qubits_-1-i)) & 1) for i in range(len(self.classes_))])
        for row in x:
            amplitudes = np.zeros(2**self.n_qubits_)
            amplitudes[:len(row)] = row
            norm = np.linalg.norm(amplitudes)
            if norm == 0:
                amplitudes[0] = 1.  # Explicit blank-input convention.
            else:
                amplitudes /= norm
            state = _backend(self.framework).qvc_state(amplitudes, theta)
            self.state_evaluations_ += 1
            scores.append(signs @ np.abs(state)**2)
        return softmax(np.asarray(scores), axis=1)

    def fit(self, x, y):
        x = _matrix(x)
        self.classes_, encoded = _labels(y, len(x))
        self.n_features_in_ = x.shape[1]
        self.n_qubits_ = max(1, (x.shape[1]-1).bit_length())
        if self.n_qubits_ > 12 or len(self.classes_) > self.n_qubits_:
            raise ValueError("Need one readout qubit per class and at most 12 reference qubits")
        self.state_evaluations_ = 0
        initial = np.random.default_rng(self.seed).normal(0, .15, size=self.depth*self.n_qubits_*3)
        def loss(theta):
            p = self._probabilities(x, theta)
            return -np.log(np.maximum(p[np.arange(len(x)), encoded], 1e-12)).mean()
        self.theta_, self.training_loss_ = _spsa(loss, initial, self.steps, self.learning_rate, self.seed)
        return self

    def predict_proba(self, x):
        return self._probabilities(x, self.theta_)

    def predict(self, x):
        return self.classes_[self.predict_proba(x).argmax(axis=1)]


class TfidfQuantumMatcher:
    """TF-IDF triage plus a jointly trained small classical/quantum head.

    Related to Bischof et al. (Scientific Reports 15, 4318). Three explicit
    pair descriptors replace the paper's learned token embedding. No author
    dataset, checkpoint or published score is replicated by this adaptation.
    """

    def __init__(self, framework="qiskit", n_qubits=3, steps=8, learning_rate=.15,
                 regularization=.001, seed=7):
        if not 2 <= n_qubits <= 12 or regularization < 0:
            raise ValueError("Need 2-12 qubits and nonnegative regularization")
        self.framework, self.n_qubits_, self.steps = framework, n_qubits, steps
        self.learning_rate, self.regularization, self.seed = learning_rate, regularization, seed

    @staticmethod
    def _pairs(pairs):
        pairs = list(pairs)
        if not pairs or any(len(p) != 2 or any(not isinstance(s, str) or not s.strip() for s in p) for p in pairs):
            raise ValueError("Expected nonempty text pairs")
        return pairs

    def features(self, pairs):
        pairs = self._pairs(pairs)
        left = self.vectorizer_.transform([p[0] for p in pairs])
        right = self.vectorizer_.transform([p[1] for p in pairs])
        similarity = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        analyze = self.vectorizer_.build_analyzer()
        rows = []
        for (a,b), score in zip(pairs, similarity):
            aa, bb = set(analyze(a)), set(analyze(b))
            rows.append([score, abs(len(aa)-len(bb))/max(len(aa),len(bb),1), len(aa & bb)/max(len(aa | bb),1)])
        return np.asarray(rows)

    def _unpack(self, theta):
        n = self.n_qubits_
        edge_count = 1 if n == 2 else n
        width = 4*n
        mixing_end = width+edge_count+n
        return (theta[:width].reshape(4,n), theta[width:mixing_end].reshape(1,edge_count+n), theta[mixing_end:].reshape(1,n,3))

    def _quantum_probability(self, features, theta):
        encoder, mixing, strong = self._unpack(theta)
        angles = np.pi*np.tanh(np.column_stack([features, np.ones(len(features))]) @ encoder)
        p = []
        for row in angles:
            state = _backend(self.framework).entity_state(row, mixing, strong)
            self.state_evaluations_ += 1
            p.append(np.sum(np.abs(state[2**(self.n_qubits_-1):])**2))
        return np.asarray(p)

    def fit(self, pairs, y):
        pairs = self._pairs(pairs)
        classes, encoded = _labels(y, len(pairs))
        if len(classes) != 2 or not np.array_equal(classes, [0,1]):
            raise ValueError("Entity labels must be 0 (nonmatch) or 1 (match)")
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1,2)).fit([s for pair in pairs for s in pair])
        features = self.features(pairs)
        grid = np.linspace(0,1,101)
        self.threshold_ = float(grid[np.argmax([np.mean((features[:,0]>=t)==encoded) for t in grid])])
        mistakes = features[(features[:,0]>=self.threshold_) != encoded, 0]
        self.uncertain_interval_ = tuple(np.quantile(mistakes,[.25,.75])) if len(mistakes) else (self.threshold_,self.threshold_)
        self.state_evaluations_ = 0
        n = self.n_qubits_
        count = 8*n+(1 if n == 2 else n)
        initial = np.random.default_rng(self.seed).normal(0,.15,size=count)
        def loss(theta):
            p = np.clip(self._quantum_probability(features,theta),1e-12,1-1e-12)
            return -np.mean(encoded*np.log(p)+(1-encoded)*np.log(1-p))+self.regularization*np.mean(theta**2)
        self.theta_, self.training_loss_ = _spsa(loss, initial, self.steps, self.learning_rate, self.seed)
        return self

    def predict_proba(self, pairs):
        features = self.features(pairs)
        lo, hi = self.uncertain_interval_
        mask = (features[:,0]>=lo) & (features[:,0]<=hi)
        p = (features[:,0]>=self.threshold_).astype(float)
        if mask.any():
            p[mask] = self._quantum_probability(features[mask],self.theta_)
        return np.column_stack([1-p,p])

    def predict(self, pairs):
        return self.predict_proba(pairs).argmax(axis=1)


def finite_difference_input_gradient(predict_proba, x, y, delta=1e-5):
    """Simulator input gradient of cross-entropy, before amplitude normalization."""
    x = _matrix(x)
    y = np.asarray(y,dtype=int)
    if delta <= 0 or y.shape != (len(x),) or np.any(y < 0):
        raise ValueError("Invalid gradient labels or difference step")
    gradient = np.empty_like(x)
    for j in range(x.shape[1]):
        plus, minus = x.copy(), x.copy()
        plus[:,j] += delta; minus[:,j] -= delta
        pp, pm = predict_proba(plus), predict_proba(minus)
        if np.any(y >= pp.shape[1]):
            raise ValueError("Class index outside predictor output")
        gradient[:,j] = (-np.log(np.maximum(pp[np.arange(len(x)),y],1e-12))+
                         np.log(np.maximum(pm[np.arange(len(x)),y],1e-12)))/(2*delta)
    return gradient


def projected_gradient_attack(predict_proba, x, y, epsilon=.2, steps=3, seed=7):
    """Small L-infinity PGD diagnostic; not a complete robustness assessment."""
    x = _matrix(x)
    if epsilon < 0 or steps < 1 or np.any((x < 0) | (x > 1)):
        raise ValueError("Need nonnegative epsilon, positive steps and pixels in [0,1]")
    rng = np.random.default_rng(seed)
    attacked = np.clip(x+rng.uniform(-epsilon,epsilon,x.shape),0,1)
    for _ in range(steps):
        gradient = finite_difference_input_gradient(predict_proba,attacked,y)
        attacked += (2*epsilon/steps)*np.sign(gradient)
        attacked = np.clip(np.clip(attacked,x-epsilon,x+epsilon),0,1)
    return attacked


if __name__ == "__main__":
    from qml_tricks.learning.case_benchmarks import main
    main()
