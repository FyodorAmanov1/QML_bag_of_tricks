# Regression artifact activity

- Read `AGENTS.md` and `agent_eval/tasks.json` with `Get-Content`.
- Located learning documentation and APIs with `rg --files | rg 'learning|regress|README|artifact'`.
- Read `docs/learning/frozen_features.md`, `docs/learning/index.md`, and `src/qml_tricks/learning/models.py` with `Get-Content`.
- Wrote a standalone `solution.py` using the repository's frozen-feature regression and save/load APIs, with a train-only classical ridge control. No library, docs, tests, or grader were inspected or changed beyond the listed API/documentation reads.
- Ran `.\.venv-clean\Scripts\python.exe agent_eval/runs/regression_artifact/solution.py` from the repository root. Exit code: 0. No execution errors or corrections were needed.
- Observed 60 training rows and 20 disjoint test rows, frozen-feature test RMSE `1.2298766840128206`, classical ridge test RMSE `0.1065822346201551`, and reload maximum prediction error `0.0`. Exact prediction equality and train-only preprocessing mean assertions passed.
- Outputs: `report.json`, `frozen_features.joblib`, and `data.csv` in this directory. The artifact was saved before test predictions, so the reload check did not reuse test-feature cache entries.
