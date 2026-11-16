# Executed scaling, evaluation and deployment checks

These records verify software behavior. They do not establish a new predictive
or computational quantum advantage, validate a medical application or reproduce
a full application paper.

| Path | Executed check | Outcome |
|---|---|---|
| Batched fixed features | Qiskit, all 569 breast-cancer records and all 30 features; five families, 13 candidates | Completed; independent metric/selection/source/model audit passed |
| Repeated evaluation | Cirq, fixed 72-record diabetes example; seeds 7/17/27; seven families and 19 candidates per run | Three runs completed and audited; paired differences and record overlap retained |
| Frozen-teacher distillation | Qiskit, trained one-qubit reference teacher; disjoint 512-row mimic fit, 128-row gate and 128-row final test | All feature/prediction/quality gates passed; classical student exported |
| Student fidelity | Same synthetic reference | Gate feature RMSE 0.00006012, relative L2 0.0004623 and zero teacher/student label disagreements on the gate |
| Final student inference | Final data inspected after export decision | Feature RMSE 0.00005121; zero native preparations during student inference |
| Installed package | Locally built wheel installed into an isolated target; PennyLane quickstart executed from that installation | Evidence resources loaded; seven-family/19-candidate quickstart and audit passed |

The teacher/student fidelity example uses a small, weakly encoded, classically
simulable synthetic teacher. Its high imitation accuracy is an integration result,
not evidence that arbitrary large hardware feature maps admit a high-fidelity
student. The fixed-feature and repeated real-observation examples are retained
even though their tested classical controls are stronger on these data.

The batched execution's configured loader cap is 20,000 rows; this record does
not claim a 20,000-row benchmark or peak process-memory measurement. Separate
tests acquire 600 rows in batches for both tasks in all three SDKs, preserve
serialization behavior and check the absence of a dense quantum Gram matrix.

Local outputs are in `results/local/batched_quickstart_v1`,
`results/local/repeated_diabetes_evaluation_v1` and `results/local/offline_demo_v2`.
The installed-package check is under `.cache/installed_run_v1`. Generated
artifacts are excluded from version control; [start here](../START_HERE.md)
recreates the workflows. Specific test coverage is in
[implementation_coverage.json](../implementation_coverage.json).
