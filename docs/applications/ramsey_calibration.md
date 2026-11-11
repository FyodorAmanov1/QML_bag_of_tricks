# Periodic sensor calibration with a frozen quantum feature bank

Use this example when the input is a physical phase, the period is known from
the instrument, labels are scarce, and prediction must extend to later periods.
It is a synthetic harmonic calibration response, not measured sensor data or
a demonstration that quantum hardware beats a suitable classical implementation.

## Model and implementation

Prepare `H|0>` and apply `RZ(k*phase)` for `k=1,2,3`. The X and Y expectations
are respectively `cos(k*phase)` and `sin(k*phase)`. Fit a regularized Ridge
readout on these six measured features. The circuit is frozen: no variational
quantum parameters or barren-plateau-sensitive optimization are required. The
downstream squared loss plus positive L2 regularization is convex.

The connection between angle encoding and accessible Fourier frequencies is
established in [Schuld, Sweke and Meyer, Physical Review A 103, 032430 (2021)](https://doi.org/10.1103/PhysRevA.103.032430).
This benchmark's particular signal, split and feature bank are local engineering
choices; the paper does not guarantee their predictive superiority.

The fixed target is

$$f(\phi)=0.7\cos\phi+0.35\sin(2\phi)-0.2\cos(3\phi)+0.15.$$

Labels have independent Gaussian noise with standard deviation 0.05. Train and
validate on independent phases in `[0,2*pi)` and evaluate on new phases in
`[2*pi,6*pi)`. This is explicitly an extrapolation task with a known periodic
prior. It is inappropriate to advertise it as an ordinary IID tabular test.

For an agent implementing this in a project:

1. Establish the phase units and period before inspecting held-out labels.
   If frequencies are unknown, estimate/select them using training/validation
   data and include that selection budget; do not copy the answer from test data.
2. Acquire X and Y in separate measurement settings for each frequency. One
   statevector simulation can provide exact expectations, but a device needs
   independent copies for the noncommuting settings.
3. Fit preprocessing and the regularized readout on training rows. Select the
   regularization from validation error; refit on train+validation and test once.
4. Compare Ridge, Elastic Net, RBF-SVR, L2-regularized MLP and boosting. Pool the
   best family using validation only. Keep the **classical Fourier-Ridge** control:
   it implements exactly the same features at negligible classical cost.
5. Compare exact and finite-shot features and report acquisition at deployment.
   Caching the acquisition across readout candidates is allowed; caching test
   labels or fitting preprocessing on test inputs is not.

## Run and interpret

```bash
python -m pip install -e ".[qiskit,learning]"
python -m qml_tricks.learning.opportunities --application ramsey --framework qiskit --output results/local/ramsey.json
```

Switch the framework to `pennylane` or `cirq` after installing its SDK extra.
Native constructions: [Qiskit](../source/applications/qiskit.py),
[PennyLane](../source/applications/pennylane.py), [Cirq](../source/applications/cirq.py).

Each input requires six measurement settings. At 1024 shots per setting,
that is 6144 state preparations per input, including each deployment input.
Exact simulation is labeled separately, with zero simulated measurement shots.
The one-qubit statevector occupies 32 bytes before SDK/array overhead.

A lower error than generic regularized models demonstrates the benefit of the
periodic representation in this setting. If Fourier-Ridge matches or beats the
quantum pipeline, use it for a classical phase-input problem. This example
does not provide a hardware or memory incentive by itself. It is useful for
choosing an inductive bias and avoiding an unnecessary QPU deployment.

All learning-curve, shot-mode and seed results are retained in the
[application report](results.md).
