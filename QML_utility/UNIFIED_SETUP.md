# One project, starting with QML Utility

`QML_bag_of_tricks` is one installable project. `QML_utility` is its practical
entry point; `qml_tricks` is the canonical research/strategy implementation
library. One root installation provides both. Utility files stay in this
directory; the existing generated strategies keep their original authoring
sources and independent native SDK construction.

## Command map

One root install provides every route below. The `project` namespace dispatches
directly to QML Utility; the aliases share parsers, defaults and execution code.

| Task | Unified command | Equivalent Utility command |
|---|---|---|
| Check environment | `qml-tricks project doctor` | `qml-utility doctor` |
| Prepare a contract | `qml-tricks project setup --output results/local/setup` | `qml-utility setup --output results/local/setup` |
| Fit a bundled comparison | `qml-tricks project quickstart --output results/local/example` | `qml-utility quickstart --output results/local/example` |
| Fit your numeric data | `qml-tricks project run --config project.json --output results/local/run` | `qml-utility run --config project.json --output results/local/run` |
| Evaluate/audit/predict | `qml-tricks project evaluate --help` (also `audit`, `predict`) | `qml-utility evaluate --help` (also `audit`, `predict`) |
| Core strategy checks | `qml-tricks project strategy ngd --compare` | `qml-utility strategy ngd --compare` |
| Advanced research commands | `qml-tricks run local_loss --framework qiskit` | `qml-utility research run local_loss --framework qiskit` |

Prefix the commands with `python -m qml_tricks` or `python -m QML_utility`
instead of the corresponding console-script name if needed. Use Python 3.12,
the same environment, SDK extras and `--framework` names throughout.

The research `run` accepts a strategy name and circuit options; project `run`
accepts a JSON data/split/budget contract. Those are different scientific tasks,
so their argument schemas and output schemas remain explicit. Strategy checks
test mechanisms; project reports evaluate predictive quality. They share the
canonical strategy implementations and QNG geometry, without treating a
mechanism demonstration as a trained-model performance result. There is no
second installation, copied circuit library or circuit-conversion layer.

```bash
python -m pip install -e ".[qiskit]"
qml-utility setup --output QML_utility/results/local/my_setup --check-essentials
```

Choose the `pennylane`, `cirq` or `all` extra as appropriate. The setup creates
a local numeric example, an editable project contract, the publication catalog,
the core training toolkit, and optional executed strategy/control checks. It
does not train a classifier or infer a predictive advantage merely by running
those checks. Follow the generated README to train the candidate portfolio.

## Start from an actual published case

Peer-reviewed findings with documented author code are sufficient starting
points; independent paper reruns are optional. See [evidence levels and source
access](EVIDENCE_LEVELS.md). New case setups use `intent="adapt"` and record
source access for every case alongside its published and local scopes.

```bash
qml-utility setup --case continual_plasticity --output QML_utility/results/local/continual_setup
qml-utility setup --profile my_project.json --output QML_utility/results/local/my_project_setup
```

`setup.json` includes all ten paper-scoped records: original data/access
conditions, observed success, implementation status, controls and blockers.
Case selection starts with required capabilities unconfirmed; inspect the data
and fill them in rather than assuming a dataset resemblance is sufficient.
An available case protocol is copied to `case_protocol.json`. The strict case
selector and the exploratory numeric portfolio retain their separate roles.

The papers document successful comparisons in their original settings. They
do not establish a recipe guaranteed to beat classical models on a new project.
Some records concern predictive quality, others quantum acquisition or conditional
theory. See [case selection](CASE_SELECTION.md) and [evidence policy](EVIDENCE_POLICY.md).

## Core training toolkit

| Tool | Executable through the entry point | What is available for a project |
|---|---|---|
| Quantum natural gradient / QNG | `qml-utility strategy ngd --compare` | Set `quantum.optimizer="qng"` for `trainable_local`; regularized FS update on circuit coordinates, Adam on the classical head |
| Local cost functions | `qml-utility strategy local_cost --compare` | Canonical local/global warm-up; derive a compatible task loss before transferring its bound |
| Identity-block / near-identity initialization | `qml-utility strategy near_identity_initialization --compare` | Canonical independent forward/inverse blocks initialized to cancel; transfer needs an ansatz with those independent parameters |
| Near-identity encoding | `qml-utility strategy near_identity_encoding --compare` | Weak-angle candidates, configurable via `quantum.encoding_scale_candidates` |

Pass `--framework qiskit`, `pennylane` or `cirq`. These commands invoke the main
library's actual `RunConfig`/`experiment` implementation. They do not copy a
second implementation or convert circuits between SDKs. Use `qml-utility strategies`
for the core toolkit or `qml-utility strategies --all` for the full research
catalog. The existing `qml-tricks` CLI remains available for advanced sweeps.

## Initialization and encoding are different

Identity-block initialization uses `B(b).dagger() @ B(a)` with independent arrays
matched only at initialization. Tying them throughout training removes those
trainable directions. The existing canonical check verifies this construction.
The portable supervised circuit's small initial RY/RZ angles are a different
engineering choice: its data uploads and CNOTs do not make the whole circuit an
identity block. Setup does not silently claim that construction was applied.

Similarly, local XYZ measurements in a supervised classifier do not by themselves
turn nonlinear cross-entropy into the local linear objective analyzed by the
local-cost theorem. Use the canonical check, inspect the assumptions, then test
a declared task-specific composition. See the primary guides for
[initialization](../docs/strategies/identity_block_init.md),
[local costs](../docs/strategies/local_loss.md),
[weak encoding](../docs/strategies/near_identity_encoding.md) and
[QNG](../docs/strategies/natural_gradient.md).

## Supervised QNG implementation

For a batch of inputs, the adapter averages the pure-state Fubini–Study metrics
of the individual encoded states. This is not the metric of their averaged mixed
state. For circuit gradient g it solves `(G + lambda*I) direction = g`, using
the main library's geometry routines, then applies a Euclidean step in that
direction. The classical readout retains its Adam update. Select `optimizer="gd"`
to compare Euclidean circuit updates with the same classical-head update.

The metric uses central derivatives of exact native statevectors, all training
inputs and the FS convention (QFIM = 4*FS). With P circuit coordinates it adds
`N*(2*P+1)` state preparations per step, on top of the gradient evaluations.
Reports retain metric preparation counts, minimum eigenvalue and regularized
condition number. Geometry source hashes/snapshots accompany utility sources.
Account for solve cost and statevector memory as well as preparations. A hardware
metric, finite-shot uncertainty and a generic plateau-avoidance claim are not
implemented or established by this adapter.

```json
{"quantum":{"optimizer":"qng","metric_regularization":0.001,"encoding_scale_candidates":[0.1,0.5,1.0]}}
```

These settings are explicit project choices, selected without test inspection.
QNG needs resolvable gradient information; local costs and initialization have
their own assumptions. The integrated setup makes all of them available without
presenting a combination as a universal guarantee.
