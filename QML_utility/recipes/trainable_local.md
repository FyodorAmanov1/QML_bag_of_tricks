# Trainable local re-uploading features

The [unified setup](../UNIFIED_SETUP.md) makes the main library's QNG optimizer
available with `optimizer="qng"` and positive `metric_regularization`.
It preconditions circuit coordinates using the mean per-input FS metric; the
classical head remains on Adam. Use `optimizer="gd"` for the matched Euclidean
circuit-update control. Default `optimizer="adam"` preserves the original
reference. Metrics use exact native simulator states and are charged separately.

Use this to test whether adapting circuit features to labels helps enough to
justify its derivative cost. The [re-uploading paper](https://doi.org/10.22331/q-2020-02-06-226)
motivates repeated encoding and expressivity; it does not establish convergence
or superiority of this circuit, local readout, loss or optimizer.

Jointly optimize the [reference circuit's](CIRCUIT_CONTRACT.md) angles, a local
XYZ linear head and a bias with full-batch Adam. Classification uses softmax
cross-entropy, including more than two classes. Regression standardizes targets
inside training data and uses squared loss, then inverts that scaling at predict.
L2 regularization acts on the classical head. No early stopping/test selection
occurs; the declared step count is fixed per candidate.

```python
model = QuantumEstimator(method="trainable_local", task="classification",
                         framework="qiskit", qubits=2, steps=20,
                         max_state_preparations=50000)
model.fit(X_train, y_train)
print(model.trace_)
```

Import the class from `QML_utility.models.estimators`. Each gate coordinate occurs
once. Its expectation derivative is half the difference at plus/minus pi/2,
following the [Pauli-shift identity](https://doi.org/10.1103/PhysRevA.99.032331).
The classical chain rule differentiates the nonlinear loss; shifting the whole
cross-entropy would be incorrect. Tests compare the joint circuit/head/bias
gradient against independent central differences for both task types.

Record loss and circuit/full gradient norms, all trials and the state-preparation
count. Compare frozen features and classical nonlinear heads. Weak encoding and
local measurements are diagnostic design choices, not a proof of plateau
avoidance. If gradients vanish, inspect state/readout dependence and consult
[the companion guide](../COMPOSING_METHODS.md). If training fits but validation
does not, test regularization and data structure rather than just training longer.
