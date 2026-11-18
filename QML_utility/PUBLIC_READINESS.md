# Public repository: practical scope and audit

The public project has one root installation, one dependency definition and two
connected layers. Start with [the setup and command map](UNIFIED_SETUP.md), then
[the bundled comparison](START_HERE.md) or [your own data](PRACTICAL_WORKFLOW.md).

## What goes where

| Location | Responsibility | Authoring rule |
|---|---|---|
| `QML_utility/` | Project contracts, routing, four fit/predict families, evaluation, offline students and publication-scoped evidence | Utility code and guides are maintained here |
| `src/qml_tricks/` | Canonical strategy mechanisms, numerical geometry and existing learning/application references | Generated strategies use `scripts/assemble.py`; learning recipes use `scripts/assemble_learning.py` |
| `docs/`, `overrides/` | Website presentation and original research guides | Utility website pages are generated from their canonical files |
| `tests/`, `QML_utility/tests/` | Numerical conventions, SDK parity, workflows, evidence and command contracts | Root `python -m pytest` discovers both |
| `results/`, `reproductions/` | Preserved studies and provenance | Keep comparison scope and historical evidence intact |

Do not install from inside `QML_utility`; it is a package in the same distribution.
Console commands and Python module commands call the same entry functions.
There is no required directory move or migration for existing users.

## Audit performed for public publication

- Parsed every tracked Python file and every tracked JSON/notebook document.
- Scanned Git history with Gitleaks, with redacted output. One precisely scoped
  historical finding was reviewed as ordinary vocabulary/tokenization prose;
  its exact fingerprint is documented in `.gitleaksignore`.
- Added shared command routing and parity checks, preserving existing commands.
- Reviewed package contents, examples, ignored run/model directories and source
  archive assets. Font license and text handoffs accompany the website assets.
- Updated source-access instructions to the public clone workflow.
- Ran the numerical/workflow suite, strict documentation build, source/result
  audits and package/installed-workflow checks. GitHub checks cover Linux and Windows.

The execution receipt for this publication is reported by the maintainer and
GitHub Actions; these bullets describe checks, not a security certification.
No automated scan proves that a repository contains no sensitive information.

## Boundaries users need to know

- Python 3.12 is the tested reference environment. Install one SDK extra first.
- Project adapters currently accept explicit numeric classification or scalar
  regression contracts. Text, graphs, long continual tasks and physical quantum
  acquisition require their own adapters/protocols; they are not generic CSV tasks.
- Exact simulation is the runnable reference. Simulator memory grows
  exponentially with qubit count. Gradient and metric preparation costs are
  recorded; a hardware execution backend is not implied by a native SDK circuit.
- Research strategy `run` and project `run` serve different purposes and retain
  distinct schemas, explicitly separated by the `project` namespace.
- Four runnable model families and ten publication-scoped cases are different
  inventories. The ten cases do not certify ten local paper-performance wins.
- Setup diagnostics, bounded initialization results and smoke tests establish
  neither full-training success nor a general quantum advantage.
- Preserve group/time splits, train-only preprocessing, validation-only
  selection, tuned classical controls and every failed or non-winning trial.
- Use fresh output directories; load only trusted serialized model artifacts.
- Peer-reviewed evidence and documented author implementations support the
  published-method collection; independent paper reruns are optional. Review
  [evidence labels](EVIDENCE_LEVELS.md), [validation](VALIDATION.md),
  [evidence policy](EVIDENCE_POLICY.md) and [contextual results](CONTEXTUAL_RESULTS.md).
