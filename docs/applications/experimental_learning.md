# Learn a quantum-state property with a late Pauli query

This is an application prototype for characterizing quantum experiments when
independent state copies are available, two copies can be stored coherently,
and the observable of interest is revealed **after** acquisition. It is a
synthetic physical-state family, not a measured industrial dataset.

The underlying two-copy measurement protocol comes from
[Huang et al., Science 376, 1182–1186 (2022)](https://doi.org/10.1126/science.abn7293).
That publication proves separations under precise data-access assumptions.
The implementation here tests a restricted ensemble and specific acquisition
protocols; its binary classifier and finite numerical comparison are engineering
companions, not a reproduction of every theorem or the hardware experiment.

## State, query and prediction target

The simulator supplies independent copies of

$$\rho_{P,a}=2^{-n}(I+aP),$$

where `P` is a uniformly drawn full-weight X/Y/Z Pauli word. For a balanced
binary task, `a=0` in class 0 and `a=+0.8` or `-0.8` with equal probability in
class 1. Neither classifier gets the amplitude or state description. The Pauli
word is supplied as a query after each state's measurement record is complete.
It is independent of the class label. Labels on training/validation states are
available for supervised readout fitting; test labels are withheld from it.

The question is whether the magnitude of that Pauli expectation is zero or
nonzero. This is a measurement-and-learning problem with quantum inputs. Feeding
a classical model a full density matrix would define another access model.

## Quantum acquisition and a regularized classical readout

Apply CNOT from each qubit of copy A to the corresponding qubit of copy B,
then Hadamard on each A qubit and measure both copies. For each pair of output
bits `(u,v)`, the XX, ZZ and YY eigenvalues are respectively
`(-1)^u`, `(-1)^v` and `-(-1)^(u+v)`. Multiply the appropriate values for the
queried Pauli word and average over Bell rounds:

$$\mathbb E[g_P]=\operatorname{Tr}[(P\otimes P)(\rho\otimes\rho)]
  =\operatorname{Tr}(P\rho)^2=a^2.$$

The minus sign for YY is essential. A Bell round consumes **two independent
copies** of the state. The acquired record can answer any Pauli query without
another acquisition; it must not be measured using a query-specific setting.
Fit an L2-regularized logistic classifier on the scalar record mean, selecting
its regularization using validation data only. This logistic objective is convex
and has no variational quantum circuit training.

## Strong conventional controls

Each conventional pipeline receives the same number of state copies, measured
one at a time in uniformly random local X/Y/Z bases before the query arrives.
For a weight-`n` query, only a fraction `3**(-n)` of records have all requested
bases. The benchmark supplies both positive/negative matching counts and useful
derived statistics to L2-logistic regression, RBF-SVC and an L2-regularized MLP.
The best conventional learned family is chosen using validation only.

It also includes an **optimal likelihood-ratio classifier for these measurement
records**. Let `h+` and `h-` count matching positive/negative parities. With
known nonzero magnitude `m`, its likelihood ratio against `a=0` is

$$L=\tfrac12[(1+m)^{h_+}(1-m)^{h_-}
 +(1-m)^{h_+}(1+m)^{h_-}].$$

Unmatched records have identical distributions under both classes and cancel.
The implementation evaluates this in log space. This control knows the signal
family and does not need to learn its parameters; it is stronger than merely
choosing a weakly regularized model. Better classical regularization cannot
recover information absent from the selected acquisition record.

A second control learns the query **before** acquisition and measures that
Pauli directly on every copy. It changes the information order and usually
removes the observed gap. Keep it visible: advance knowledge of the query is
exactly a scenario in which the two-copy machinery may be unnecessary.

## Implementation and cost contract

```bash
python -m pip install -e ".[qiskit,learning]"
python -m qml_tricks.learning.opportunities --application pauli --framework qiskit --output results/local/pauli.json
```

[Qiskit](../source/applications/qiskit.py), [PennyLane](../source/applications/pennylane.py)
and [Cirq](../source/applications/cirq.py) independently construct the Bell
measurement. Their small density-matrix Born probabilities are checked against
the analytic law. For this special family, the larger benchmark efficiently
samples that law classically; **it does not execute its thousands of records
as native SDK jobs or QPU shots**. The simulator's hidden generator is not a
permitted learner input. This is not evidence that the generator is hard to
sample classically when its hidden parameters are supplied.

For `B` physical copies of an `n`-qubit state, compare `B` local rounds with
`B/2` Bell rounds. Bell acquisition needs `2*n` simultaneously coherent qubits,
`n` CNOTs and `n` Hadamards per round. Both protocols read `B*n` bits in total.
The implementation stores bytes per bit and records actual array storage.
State preparation, sensor-to-memory transfer, device gate errors and SDK overhead
are not free and are not modeled by the reported record sizes.

The noise sweep independently flips each measured bit with probability 0.01;
it is a toy readout-noise model, not a complete hardware noise forecast. For a
word containing `y` Y factors, the Bell statistic attenuates by
`(1-2*error)**(n+y)`. The single-copy matching parity attenuates by
`(1-2*error)**n`. The optimal local control accounts for that attenuation.

Use this strategy when copy/sample efficiency matters, the query is late, and
coherent two-copy operations are practical. Prefer ordinary Pauli measurements
when the observable is known early, and reconsider if coherent memory or gate
noise consumes the sample benefit. The local numerical benchmark does not
exhaust adaptive/global single-copy POVMs; any broader lower-bound claim must
be checked against the original theorem. Read the [complete results](results.md).
