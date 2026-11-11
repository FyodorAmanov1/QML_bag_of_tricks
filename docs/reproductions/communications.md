# Wireless channel equalization: Hu et al., Nature Communications (2024)

Primary paper: [Overcoming the coherence time barrier in quantum machine
learning on temporal data](https://doi.org/10.1038/s41467-024-51162-7), including
its publisher corrections. The authors demonstrate recovery of classical
communication symbols with a reset-based quantum reservoir and report gains
over direct memoryless logistic regression. Its hardware experiment is distinct
from the numerical replication here. Source code is at
[NISQRC-Codes](https://github.com/skhanCC/NISQRC-Codes).

## The fixed reproduction target

The target is the connected two-memory/four-readout-qubit Hamiltonian simulation
in Figure 3 at 20 dB. Classical four-PAM messages take values `{-3,-1,1,3}`.
An eight-tap filter, cubic nonlinearity and Gaussian noise generate the received
stream. Thus this is a simulated communications application, not measured
network traffic and not a quantum-data learning problem.

The pinned notebook publishes a reference error of 0.0021 for the ideal
connected reservoir and 0.0446 for direct logistic regression at this SNR.
Its random message generation is unseeded, and its complete connectivity graph
is overwritten by a split graph in an earlier cell. The local port explicitly
selects the paper's connected branch and generates three new seeded datasets.
An exact replay of the notebook's printed numbers is therefore not claimed.

## Model and implementation

For each received scalar `u`, form the prescribed Hamiltonian `H0 + u*H1`.
`H0` contains random local X fields and complete-graph ZZ couplings; `H1`
contains local Z fields. Evolution is `exp(-i*(H0+u*H1))` for unit time.
Initialize readout qubits to zero, evolve with the retained memory state,
record all 16 readout probabilities, trace out the readout and reset it to zero.
Start each message from the null-input fixed point.

The implementation keeps the 4-by-4 memory density matrix and contracts a
64-by-4 unitary isometry, rather than storing a full 64-by-64 density matrix at
every update. This is an algebraically equivalent classical simulation of the
mean channel. Tests verify partial traces, positivity, normalization and native
[Qiskit](../source/replications/qiskit.py),
[PennyLane](../source/replications/pennylane.py) and
[Cirq](../source/replications/cirq.py) one-step evolution.

Fit the paper-style L2-logistic readout with `C=1e6` on 100 independent training
messages of 100 symbols, and test on 100 new messages. Fit direct logistic
regression to current scalar inputs as the original classical comparator.
Retain 20 additional validation messages for the stronger controls: tuned
causal polynomial-delay logistic regression and an L2-regularized delay-window
MLP, with nine candidates each. These controls use eight past/current samples;
no window crosses a message boundary or uses future samples.

## When to use and how to interpret

Use this architecture as an experimental nonlinear equalizer when the task
needs fading memory and reset/readout are available. Its fixed quantum channel
and convex classical readout avoid training a large variational quantum circuit.
The mean channel has no measurement-trajectory noise in this reproduction.

Improvement over a memoryless classifier supports the reservoir's nonlinear
temporal representation; it does not establish superiority to classical models
with comparable memory. Check the stronger controls in the [results](results.md).
The controlled models additionally refit on train+validation messages; that
training-data difference is explicitly recorded.

A separate post-run fairness audit refits those same selected classical
configurations on the original 100 training messages only, matching the quantum
replay's fit budget. It changes no candidate grid or seed and retains the earlier
train+validation fits. Its rows are labeled `train_only`; the extra audit is
identified separately from the pre-specified numerical run.

This is one SNR, three fresh datasets and exact unconditional dynamics. It does
not reproduce the six-SNR sweep, correlated finite-shot trajectories, device
noise, the no-reset ablation or the seven-qubit hardware experiment. A physical
implementation needs repeated message executions to estimate probabilities;
zero hardware shots in the report means exact simulation, not free acquisition.
