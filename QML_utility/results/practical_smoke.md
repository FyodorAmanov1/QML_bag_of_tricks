# Executed practical integration checks

These checks establish a runnable software path, not a favorable application
benchmark. Dataset rows were chosen with a fixed seed before any model result,
and every feature was retained. No setting was selected because QML won.

| Check | Native SDK | Data | Fitted families | Candidate trials | Independent audit |
|---|---|---|---:|---:|---|
| Three-class classification | Qiskit | 72 wine records, all 13 features | 9 | 25 | Passed |
| Scalar regression | PennyLane | 72 diabetes records, all 10 features | 9 | 25 | Passed |

Both use two qubits, one layer, six trainable steps where applicable and the
same three encoding-scale candidates as the Fourier representation control.
All families completed fitting, validation selection, test prediction and artifact
export. The final validation-selected champion is linear for wine and Fourier
linear for diabetes. These observations belong to these integration configurations;
they do not rank method families generally or invalidate the published cases.

The per-run audit independently recalculates every saved validation/test metric
from original CSV targets, checks candidate/champion selection and verifies data,
source-snapshot and model-artifact hashes. No author paper was relabeled as
locally reproduced. The datasets are small illustrative research observations;
the diabetes check is not a validated medical application.

Local artifacts are under `results/local/practical_wine_run_v2` and
`results/local/practical_diabetes_run_v2`, with their input configs in
`results/local/practical_wine_v1` and `results/local/practical_diabetes_v1`.
Local outputs and serialized models are intentionally excluded from version
control. Recreate your own artifacts using [the workflow](../PRACTICAL_WORKFLOW.md).
The checked-in [coverage manifest](../implementation_coverage.json) records the
specific test coverage, including end-to-end training tests for all three SDKs.
