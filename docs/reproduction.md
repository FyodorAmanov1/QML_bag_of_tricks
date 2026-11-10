# Reproducing and extending an experiment

## Installation

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,learning,notebooks,dev,docs]"
```

On Linux/macOS use `source .venv/bin/activate` after the same `venv` command.
The SDK versions in the project extras define the supported reference combination.
`requirements-reference.txt` records its resolved packages after validation.

## Commands

```bash
python -m qml_tricks run local_loss --framework cirq --seed 7 --output results/local/run.json
python -m qml_tricks run local_loss --framework cirq --shots 4096 --initializations 16
python -m qml_tricks sweep local_loss --framework cirq --sizes 2 4 8 --steps 0 --initializations 32 --output results/local/sweep.json
python -m pytest
python scripts/run_notebooks.py
python scripts/grade_agent_tasks.py
python -m mkdocs build --strict
```

`run` selects one control or strategy. `sweep` runs both variants at each size.
Quantum copies are not free: sampled measurement counts include each sampled
objective evaluation. The additional exact reference evaluations are reported
but do not add simulated shots. Finite-shot MMD uses squared empirical moments;
it has a plug-in bias rather than an unbiased U-statistic estimator.

## Reading results

`configuration` records the experimental inputs. `training_trace` starts before
the first update. `gradient_statistics` describes the selected coordinate (zero by default, a final-layer rotation for non-unital noise, and the first forward Y angle for identity blocks), central
differences, bootstrap intervals, and measurement-noise separation. A zero result
can reflect an irrelevant parameter rather than a plateau.

Some architecture examples pad parameter storage to a common size. Stored
parameter count is consequently an upper bound on independent active parameters.
Framework depth means each SDK's native logical schedule; it is not a common
hardware latency benchmark. Two-qubit Haar matrix gates are counted as logical
two-qubit operations before synthesis. The dense simulator storage estimate does
not include host overhead or circuit objects.

## Public Python interface

```python
from qml_tricks import RunConfig
from qml_tricks.strategies.local_loss.cirq_impl import run

result = run(RunConfig(n_qubits=4, seed=7, steps=3))
print(result.to_dict())
```

Native modules also expose `build_circuit`, `probabilities`, and `resources`.
Pure-state modules expose `statevector` for simulator geometry and validation;
channel modules reject this operation. QNG uses these amplitudes for its dense
FS metric and reports their separate computational cost.
PennyLane's builder queues operations inside a QNode; Qiskit/Cirq return circuit
objects. Probability vectors use qubit zero as the first, most significant bit.

## Resource guards and failures

`--encoding-scale` and `--features` configure the near-identity data encoder;
features must lie in [-1, 1] and match the qubit count. `--metric-regularization`
sets QNG's positive diagonal regularizer. These two simple encoder/optimizer
examples have fixed gate layouts, so `--depth` does not change them. Identity
blocks use `--depth` as the number of shallow matched blocks.

Defaults are intentionally small. Pure-state examples accept 2–12 qubits; mixed
state examples cap at 8. Hierarchies require powers of two. Unsupported settings
raise errors. Very small ensembles give poor confidence intervals; increase
initializations and repetitions before drawing statistical conclusions.

## Extending a recipe

Implement the requested data preparation in each SDK. Validate normalization,
output ordering, objective equivalence, gradient conventions, and the theorem's
data-dependent factors. Replacing preparation with an arbitrary amplitude vector
requires a new loading-cost analysis. Re-run parity and structural tests, then
describe the exact difference from the published construction.
