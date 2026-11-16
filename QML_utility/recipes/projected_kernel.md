# Local quantum projection followed by an RBF head

Use this to test whether nonlinear processing of the same local quantum features
helps beyond a linear head. The [published projected-kernel work](https://doi.org/10.1038/s41467-021-22539-9)
contains task/access-specific results; the bundled local XYZ map is an engineering
adaptation and does not reproduce its advantageous data construction.

Extract the [fixed local features](CIRCUIT_CONTRACT.md), then fit an RBF SVC for
classification or kernel ridge for regression. A Gaussian kernel over fixed real
feature vectors is PSD. This mathematical property is not evidence of better
predictions. The kernel head still incurs quadratic training storage/work.

```python
model = QuantumEstimator(method="projected_kernel", task="regression",
                         framework="pennylane", scale=.5)
model.fit(X_train, y_train)
```

Import `QuantumEstimator` from `QML_utility.models.estimators`. Compare the fixed
feature linear head and classical RBF model. The initial runner sweeps scale;
bandwidth and regularization extensions require a new declared inner search.
Report collapsing features and same-feature head gains separately. At hardware
inference, new inputs still require quantum features; this is not an offline
classical surrogate. Finite-shot features should be acquired/cached consistently
and their uncertainty evaluated before deployment.
