# Find applications where quantum structure matters

The original [real-data study](../study.md) remains a useful negative control.
Two-qubit frozen features on compressed generic tabular data do not test every
QML architecture or application. They do not establish a reason to choose QML
for those recorded datasets either.

These additional experiments ask more specific questions:

| Application | Potential benefit | Control that tests the explanation |
|---|---|---|
| [Periodic sensor calibration](ramsey_calibration.md) | A known periodic representation can extrapolate with few labels | Classical Fourier-Ridge implements the same features cheaply |
| [Late-query quantum-state characterization](experimental_learning.md) | Two-copy acquisition can retain more useful information per state copy | Tuned regularized models, optimal local-record inference, and a query-known acquisition control |

The [full results](results.md) retain every pre-specified configuration, learning
curve, copy/shot budget and seed. A gain against a stated control is distinct
from a gain against the best possible classical method, a wall-time speedup,
or a demonstrated production application. No tests assert that QML must win.

## Reproduce or adapt

```bash
python -m pip install -e ".[qiskit,learning,notebooks]"
python -m qml_tricks.learning.opportunities
python scripts/build_opportunity_report.py
```

The installed `qml_tricks/learning/opportunity_protocol.json` fixes seeds,
sample sizes, input/state families and the configuration grid before evaluation.
Every learned family has nine actual tuning candidates. The conventional
champion pools 45 regression or 27 classification candidates using validation
scores; the quantum readout has nine. Candidate counts, times, warnings,
failures, predictions, preprocessing statistics and source hashes are retained.

To benchmark a new project, first specify the data access and intended benefit,
then freeze its splits, candidate grids and resource budget. Include regularized
and domain-informed controls. Preserve unsuccessful configurations and collect
new independent test data after changing a hypothesis using observed outcomes.
Do not tune a benchmark until its desired headline appears.

## A real-world lead, without an invented reproduction

[Wang et al., Advanced Science (2025)](https://doi.org/10.1002/advs.202506213)
report quantum kernel regression gains on 159 GaN HEMT fabrication samples,
including comparison with Elastic Net. The paper states that the supporting
data are available from its corresponding author on request, and acknowledges
that a more comprehensive hyperparameter search remains future work. This is
a relevant semiconductor-modeling lead, not a locally reproduced result or
proof that it beats systematically tuned regularized models. No substitute
synthetic data are labeled as those experiments, and no author contact has been
sent. Obtain the data and freeze a leakage-safe tuning/validation protocol before
attempting an independent replication.
