# Agent implementation evaluation

Five project tasks were assigned to three fresh-context coding agents. Two
agents subsequently completed a second task. They were allowed to read repository
instructions and docs, write only within their assigned task folder, and execute
their solutions. They did not perform a scientific review of the repository.

The root task independently re-executed all five saved scripts and graded the
resulting artifacts. All five passed the recorded acceptance checks. The scripts,
synthetic CSVs, prompts and activity logs are retained in agent_eval/. Model
artifacts can be regenerated and are excluded from the source distribution.

| Task | Independent acceptance evidence |
|---|---|
| CSV classification | Two quantum models and four controls, disjoint splits, metrics recomputed and targets checked against the CSV |
| Regression artifact | Train-only scaler statistics, held-out RMSE independently recomputed from the reloaded model, identical predictions |
| Finite-shot diagnostics | Seed/configuration, sampled evaluation count, shot accounting and estimator-bias disclosure |
| Offline surrogate deployment | Passed held-out gate and reloaded teacher-free student artifact |
| Rejected poor surrogate | Failed fidelity check, ValueError on export and no deployment artifact |

## Reproduce

```bash
python scripts/grade_agent_tasks.py
```

The machine-readable report is results/agent_evaluation.json. Replay times are
script execution times, not original agent development times. Activity logs
describe commands and corrections but are self-reported; token and tool counts
were not instrumented. Do not infer them from replay times.

This is a bounded integration evaluation with one solution per task and no
control condition. It does not estimate a general agent success rate, compare
agent models, or prove the docs caused success. Independent human/scientific
review is reserved for the repository owner, as requested.
