# State-overlap kernel

Use this to test state-overlap similarity when moderate sample counts permit a
Gram matrix. The [published quantum kernel estimator](https://doi.org/10.1038/s41586-019-0980-2)
is the method foundation, not an assertion that this reference circuit wins.

With the [fixed reference states](CIRCUIT_CONTRACT.md), compute
`K(x,y)=abs(vdot(psi(x),psi(y)))**2` and fit a precomputed SVC or kernel ridge.
The ideal fidelity kernel is PSD. Training is convex conditional on the fixed
map and chosen regularization, but the map's usefulness is empirical.

```python
model = QuantumEstimator(method="fidelity_kernel", task="classification",
                         framework="cirq", qubits=2)
model.fit(X_train, y_train)
```

Import the class from `QML_utility.models.estimators`. This implementation caches
full native simulator statevectors and calculates Gram entries classically.
It is an exact classical simulation baseline for a quantum model, not evidence
of quantum computational savings. Record the off-diagonal standard deviation,
Gram participation rank, cached states, Gram memory and pair counts.

If K approaches the identity or an all-ones matrix, examine scale and feature
geometry before increasing circuit depth. Include classical RBF and Fourier
controls. Hardware needs a separate overlap-estimation/noise/PSD-extension
adapter. [Nyström](../../docs/learning/nystrom.md) reduces training acquisition
but retains new-input quantum queries; offline feature distillation is a different
deployment contract.
