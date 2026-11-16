# Prototype validation — 16 November 2026

Current integration: the root project is the single distribution and default
test entry point. Earlier standalone-package commands below document past checks;
use `python -m build` from the repository root for the unified artifact and
`qml-utility setup` for the combined workflow. See [unified setup](UNIFIED_SETUP.md).

Executed with the existing Python 3.12 repository environment:

```bash
python -m pytest tests QML_utility/tests -q
python -m mkdocs build --strict --site-dir .cache/qml-utility-expansion-doccheck
python scripts/build_showcase.py --check
python -m QML_utility.report_results
```

**Observed:** 401 tests passed, including 89 utility tests. Strict MkDocs
and canonical showcase checks passed. Every local Markdown link in the new
folder resolves. Original tracked repository files remain unchanged.

The new checks cover rejection of an unmatched generic CSV, incomplete capability
profiles, absence of evidence promotion, amplitude-coordinate preservation,
SU(4) half-angle convention, noncommuting-coordinate wrapping, independent
three-SDK state agreement and readout derivatives, Bell probability normalization,
YY signs, copy accounting, noise and the early-query ablation.

Two executed audit reports contain 40 runs and 80,000 labeled state examples.
Independent binomial-PMF analysis reproduces every saved prediction and metric.
The source hash is verified before regenerating the human-readable result tables.
All five source pins were verified: graphon, plasticity, attention, photonic kernel
and volatility. The three newly added download routes were executed successfully;
inspected files matched their pinned hashes and no author code was executed.

## Published-case expansion checks

The original prototype run had 326 passing tests. The expansion adds 12 cases
to the test count, covering ten-case evidence routing, explicit capability
requirements, contextual follow-up contracts and preserved original-result hashes.
The native channel checks distinguish signed Choi coefficients from squared-state
signals, verify the I/X/Y/Z to Bell-bit mapping independently in all three SDKs,
and cover a correlated two-system-qubit channel. Calibration division rejects
unstable factors and leaves out-of-range estimates visible rather than clipping.
Gaussian attention tests check the mathematical rule, normalization, masking and
key/value permutation consistency.

Both planner examples were executed. The molecular draft selects the intended
case and reports missing data provenance and budget; the wine follow-up includes
its old bounded outcome, preserves the no-match status, and requests a budget.
Neither command trains a model or certifies a performance gain. All 24 utility
Markdown files passed a local file-link check, and all five cached source pins
passed hash rechecking. Direct native pure-Pauli output decoding passed in all
three SDKs without a probability-order adapter.

The new paper-specific protocols have not been executed. Text paper/code
differences, molecular SI inconsistencies/selection scope, volatility selection
history and missing channel data artifacts remain unresolved scientific
replication requirements. The strict MkDocs check covers the existing website;
this standalone utility's links were checked separately and it was not published.

These checks establish implementation consistency and the stated restricted
measurement comparison. They do not establish a paper-scale classical-data
training win, hardware performance, asymptotic speedup or production readiness.

## Practical portfolio verification

The practical layer adds 17 tests to the 338-test case-expansion state, for a
total of **355 passing tests**, including **43 utility tests**. Its coverage is
recorded in [implementation_coverage.json](implementation_coverage.json).

- All three SDKs pass an independent analytical XYZ/half-angle fixture and
  expectation-gradient finite differences. Full states agree for seven uploaded
  coordinates, three qubits and two layers; a tail-coordinate change affects
  the state, checking against silent feature truncation.
- Joint nonlinear-loss derivatives are checked independently for classification
  and regression, including the circuit/head/bias chain rule.
- All four QML families complete fitting, held-out prediction, model reload and
  artifact audit for classification/regression in each of the three SDKs.
- Group and chronological splitting, embargo, repeated-time rejection, train-only
  preprocessing, target exclusion, budget failures and failed-trial retention
  have executable checks.
- Routing responds to gradient, dense-Gram memory, preparation and deployment
  constraints while preserving separate publication matching and companion roles.

Two real-observation integration runs also completed with all features retained:
72-record three-class wine/Qiskit and 72-record diabetes regression/PennyLane.
Each fitted nine families across 25 candidates. Independent audits verified the
saved metrics, original targets, validation-selected candidates/champion and
dataset/source-snapshot/model hashes. See [the executed checks](results/practical_smoke.md).
These are integration records, not successful application-paper reproductions.

```bash
python -m QML_utility.validate_docs
python -m QML_utility.audit_project --run QML_utility/results/local/practical_wine_run_v2 --config QML_utility/results/local/practical_wine_v1/project.json
python -m QML_utility.audit_project --run QML_utility/results/local/practical_diabetes_run_v2 --config QML_utility/results/local/practical_diabetes_v1/project.json
```

The utility's local links and JSON syntax were checked separately from the strict
MkDocs build of the existing website. Existing canonical website/showcase checks
also passed. No tracked main-repository file, study or generated strategy changed.

## Scaling, multi-run evaluation, packaging and offline deployment

The next implementation round adds 23 tests: **378 total, 66 in the utility**.
The native circuit definitions remain unchanged. New checks cover batched
600-row acquisition for both tasks in all three SDKs, bounded state batches,
single-pass acquisition despite repeated head epochs, memory/preparation failures,
larger-data runner audits, explicit predefined row IDs, repeated IID/group
holdouts, expanding chronological windows and failed/missing paired outcomes.

Offline-student checks cover both tasks and three SDK teachers, independent gate
identities, complete classification gate coverage, export rejection, frozen heads,
serialization and prediction after blocking every quantum SDK and the native-model
package in a fresh process. Prediction CSV checks retain the fitted feature order,
record simulator counts and protect existing artifact/gate sidecars.

Executed checks outside the unit suite include all-record 569-row batched Qiskit
classification, three repeated Cirq regression runs, the trained-teacher synthetic
distillation demo and an isolated installed-wheel PennyLane quickstart. See
[the exact record](results/workbench_checks.md). The wheel includes evidence,
protocol and immutable reference-result resources and requires no `qml_tricks`
runtime import. The package preserves the existing repository license.

```bash
python -m QML_utility quickstart --preset scalable --output QML_utility/results/local/batched_quickstart_v1
python -m QML_utility evaluate --config QML_utility/results/local/repeated_diabetes_v1.json --output QML_utility/results/local/repeated_diabetes_evaluation_v1
python -m QML_utility surrogate_demo --output QML_utility/results/local/offline_demo_v2
python -m build --no-isolation --wheel QML_utility --outdir QML_utility/.cache/wheels
```

These paths name the executed local runs; choose new destinations when repeating.
The loader's 20,000-row cap is an implementation limit, not a measured benchmark
at that size. Reported feature/cache/state sizes are component accounting, not
peak RSS. Hardware performance and application-paper replication remain separate
uncompleted evidence gates.

## Unified project verification

Twenty-three integration checks bring the default root suite to **401 passing
tests**. They cover one distribution/two entry points, canonical strategy/control
execution for all four essentials in all three SDKs, case setup without invented
capabilities, and supervised QNG for both tasks in all three SDKs. The QNG update
is compared against independently derived one-qubit state derivatives and an
explicit regularized linear solve, with separate metric preparation accounting.

The root wheel and sdist build successfully; archive checks confirm both code
layers, both commands and exclusion of generated private runs/caches. The strict
website build includes utility docs from their canonical sources and the homepage
links to unified setup. Root showcase consistency checks and utility links pass.
No generated research circuit was changed or relabeled by the integration.

An isolated installation of the single root wheel loaded both code layers from
the same installed target, executed all four essential strategy/control checks,
and completed a nine-family/25-candidate classification comparison with supervised
QNG enabled. Encoding-scale customization also keeps the Fourier control on the
same declared scale grid. This verifies integration, not a new predictive win.
