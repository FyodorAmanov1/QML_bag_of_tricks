# Activity

- Read `AGENTS.md` and `agent_eval/tasks.json` using `Get-Content`.
- Listed learning files with `rg --files | Select-String -Pattern 'learning|classification|README|pyproject'`.
- Read `docs/learning/index.md`, `frozen_features.md`, `projected_kernel.md`, and the learning `benchmark.py`, `models.py`, and `__init__.py` APIs using `Get-Content`.
- Ran `.\.venv-clean\Scripts\python.exe -c "import sys; import qiskit, numpy, sklearn; from qml_tricks.learning.benchmark import benchmark; print(sys.executable); print('Qiskit', qiskit.__version__); print('learning API imports succeeded')"`. Imports succeeded; Qiskit 2.3.1.
- Added standalone `solution.py` using repository benchmark API, two QML methods and the API's four classical controls.
- Errors or corrections before execution: none. Console displayed mojibake for parts of Windows paths; file access and imports succeeded.
- Ran `.\.venv-clean\Scripts\python.exe agent_eval/runs/csv_classification/solution.py`: exit code 0; generated `data.csv` (80 rows) and `report.json`; disjoint split sizes 48/16/16.
- Ran `.\.venv-clean\Scripts\python.exe -c "import json, pathlib, runpy; p=pathlib.Path('agent_eval/runs/csv_classification/report.json'); before=json.loads(p.read_text()); runpy.run_path('agent_eval/runs/csv_classification/solution.py', run_name='__main__'); after=json.loads(p.read_text()); assert before['test_predictions']==after['test_predictions']; assert before['split_indices']==after['split_indices']; print('Repeated-run predictions and splits identical'); print(json.dumps(after['resource_counts'], indent=2))"`: exit code 0; repeat predictions and splits identical. Each QML method used 48 training and 80 total simulator preparations, 16 test preparations, zero shots, and zero hardware jobs.
- Observed test accuracies: frozen features 0.5625; projected kernel 0.5; classical linear/RBF 0.9375 each; MLP/boosting 0.875 each. No advantage claim.
- Execution errors or corrections: none. No library, docs, tests, or grader files changed or inspected.
