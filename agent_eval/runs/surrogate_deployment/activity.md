# Surrogate deployment activity

- Read `AGENTS.md` and `agent_eval/tasks.json` in the preceding regression task; the current task is `surrogate_deployment`, seed 41.
- Ran `Get-Content docs/learning/kipu_offline_surrogate.md; Get-Content docs/learning/kipu_workflow.md; Get-Content src/qml_tricks/learning/offline_surrogate.py` to inspect the intended protocol, gate, artifact and API.
- Wrote only this directory's standalone `solution.py` and outputs. Did not inspect a grader, edit library/docs/tests, or delegate.
- Generated independent synthetic recorded pairs under a known linear map with 96 train, 48 validation and 48 test inputs. Used the repository's default fidelity thresholds, exporting only after the gate passes.
- Ran `.\.venv-clean\Scripts\python.exe agent_eval/runs/surrogate_deployment/solution.py` from the repository root: exit code 0. No errors or corrections were needed.
- Observed validation gate passed, relative feature RMSE `9.797027627796393e-09`, teacher/student agreement `1.0`, separate test accuracy `1.0`, and identical predictions after artifact reload. Restored inference produced no quantum backend import attempts and made zero quantum calls.
- Saved `report.json`, `offline_student.joblib`, its `.validation.json` metadata, and `recorded_pairs.npz`. These results concern this synthetic linear map only.
