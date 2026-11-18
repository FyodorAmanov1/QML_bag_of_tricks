# Implementation contracts for an agent

The new [practical workflow](PRACTICAL_WORKFLOW.md) provides reusable fit/predict
estimators and a CSV comparison adapter. Its [circuit contract](recipes/CIRCUIT_CONTRACT.md)
is distinct from the publication-specific references below. See
[the repository design](REPOSITORY_DESIGN.md) for how the original mechanism
library supplies canonical runtime implementations in the same distribution.

1. Fill in `examples/project`-style JSON with capabilities actually available.
   A missing capability is not inferred from a dataset name.
2. Run `select_case`; read the returned guide, primary result and protocol.
3. Use or adapt the author implementation or a scoped local recipe. An original
   paper rerun is optional; follow the exact upstream task only when claiming
   an independent replication. Use an isolated environment for legacy
   dependencies. Record the immutable commit, code changes, data hashes and trials.
4. Use full task-relevant data. Include resource-matched classical controls and
   an evaluation plan frozen before test inspection.
5. Save raw predictions/trajectories, independent seeds/task orders, uncertainty,
   wall time, memory, loading, readout and query counts. For a continual model,
   measure adaptation and retention separately.
6. A failed replication is a result. Do not weaken a baseline or rename a toy
   result to obtain a successful status. See `EVIDENCE_POLICY.md` for promotion.
7. Phrase a non-win in terms of the tested configuration, data and budget.
   Generate a prospective follow-up with `plan_project --profile ... --output ...`
   and [the workflow](AGENT_WORKFLOW.md); alternative training and structure are
   testable hypotheses rather than guaranteed remedies.

## Native continual-model interface

```python
import numpy as np
from QML_utility.models.numerics import amplitude_input, last_z
from QML_utility.models.qiskit_impl import continual_state

amplitudes = amplitude_input(my_feature_vector)  # pad to 2**n, normalize; no PCA
n = int(np.log2(len(amplitudes)))
theta = np.zeros((2, n - 1, 15))  # 2 brickwall layers; reference construction only
state = continual_state(amplitudes, theta)
logit = last_z(state)
```

The other two SDK modules expose the same signature. Parameters index all 15
nonidentity two-qubit Paulis in lexicographic `(I,X,Y,Z)` tensor order. A block is
`exp(-i sum(theta_j P_j)/2)`. Each layer traverses even pairs then odd pairs.
The reference caps explicit statevectors at ten qubits. Amplitude preparation
can require O(2**n) operations for an arbitrary classical vector; logical register
width alone is not a loading or memory advantage. The matrix exponential has
noncommuting generators; a naive two-point parameter-shift rule is not assumed.

This interface is an inference/construction building block. It supplies no
trained checkpoint or complete paper-scale optimizer/training trajectory.

## Fetch inspected source files without executing them

```bash
python -m QML_utility.fetch_upstream --case plasticity
python -m QML_utility.fetch_upstream --case graphon
python -m QML_utility.fetch_upstream --case attention
python -m QML_utility.fetch_upstream --case photonic_kernel
python -m QML_utility.fetch_upstream --case volatility
```

The downloader verifies recorded hashes, writes only under `QML_utility/.cache/`,
and prints the pin. It does not install dependencies, fetch pickle datasets,
unpack archives, run author code or modify the original repository.

## Signed channel component

```python
from QML_utility.models.qiskit_impl import pauli_channel_probabilities
from QML_utility.models.pauli_channel import eigenvalues_from_bell_probabilities

probs = pauli_channel_probabilities([0.7, 0.1, 0.15, 0.05])  # I,X,Y,Z errors
words, eigenvalues = eigenvalues_from_bell_probabilities(probs)
# words: I,X,Y,Z; eigenvalues: 1.0,0.6,0.7,0.5
```

PennyLane and Cirq construct their own native Bell-pair preparation, channel and
Bell readout. The input error probabilities use lexicographic IXYZ tensor order;
the output bit order is all system bits then all reference bits, most significant
first. An X error maps to 01, Y to 11, Z to 10 for one pair. Decoding estimates
`P tensor transpose(P)`, with signed coefficients, unlike the two-copy state
function's `P tensor P` square estimator. The references enumerate 1–3 system
qubits with exact simulation and zero shots; density matrices use O(16**n) memory.
They do not implement device SPAM-depth fits, calibration acquisition, drift
handling or the full published EMEEL experiment. The known-attenuation helper
rejects small divisors and does not clip estimates; covariance remains to be fitted.

## Projected attention component

`models/projected_attention.gaussian_projected_attention(q, k, v, key_mask)`
accepts scalar Pauli query/key expectations and a matrix of value features.
It returns aggregated values and normalized Gaussian weights; masked keys receive
zero weight. All query rows are returned, so callers must also mask padded query
positions in pooling and loss. This is classical postmeasurement computation
with O(query_count * key_count) attention storage and no quantum queries. Native
token encoders, tokenization, training and trained checkpoints remain upstream
replication work. Swap quantum and classical features while keeping this same
aggregation when testing which component contributes any observed gain.
