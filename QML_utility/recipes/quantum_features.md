# Fixed local quantum features

Use this as an inexpensive-to-optimize representation baseline when the input
is an explicit numeric matrix and local measurement acquisition is affordable.
The [feature-space foundation](https://doi.org/10.1103/PhysRevLett.122.040504)
motivates quantum feature maps; it does not prove these features are useful.

Prepare the [reference state](CIRCUIT_CONTRACT.md), extract local XYZ expectations,
then fit an L2 logistic head for binary/multiclass classification or ridge for
scalar regression. The quantum map is fixed, so no variational quantum gradient
is required. Fixed quantum features can still collapse or be uninformative.

```python
from QML_utility.models.estimators import QuantumEstimator
model = QuantumEstimator(method="quantum_features", task="classification",
                         framework="qiskit", qubits=2, layers=1, scale=.5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Use `framework="pennylane"` or `"cirq"` for independent native constructions.
The same API supports `task="regression"`. The project runner selects among three
encoding scales using validation, and records feature variance/rank. Compare
full-feature regularized/RBF/boosting controls and the same-angle Fourier head.
If rank is low, test scale, readout or structure changes prospectively; more
classical head training alone cannot add information absent from the fixed map.
