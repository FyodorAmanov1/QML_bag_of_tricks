# Three complete project walkthroughs

Start with one notebook and one SDK. The notebooks are saved with executed
outputs, interpretations, plots and checks. Actual execution times are recorded
in results/notebook_execution.json; they are not universal runtime promises.

| Walkthrough | What you learn | Notebook |
|---|---|---|
| Real-data classification | Train-only preprocessing, quantum features, classical control and saved-model parity | [Open notebook](source/notebooks/01_classification.ipynb) |
| Real-data regression | Target scaling, original-unit RMSE/MAE and error plots | [Open notebook](source/notebooks/02_regression.ipynb) |
| Offline quantum-feature distillation | Train teacher first, gate fidelity, export a classical-only student | [Open notebook](source/notebooks/03_offline_surrogate.ipynb) |

## Run locally in your browser

```bash
python -m venv .venv
# Activate the environment using your shell.
python -m pip install -e ".[qiskit,learning,notebooks]"
python -m pip install jupyterlab
python -m jupyter lab notebooks
```

The same notebooks can run in Colab after a public repository URL is configured.
Until that destination exists, no working public Colab URL is claimed. Set the
first cell's REPO_URL or QML_TRICKS_REPO_URL to that destination. The release
preparation script generates repository-specific Colab links once supplied.

## Troubleshooting and adapting

- Use Python 3.12 and the matching SDK extra; begin with Qiskit only.
- Set FRAMEWORK to pennylane or cirq after installing its extra.
- A failed student export means the validation gate rejected fidelity. Preserve
  the failure and improve acquisition/model quality using fresh validation data.
- Synthetic distillation fidelity is separate from teacher task usefulness.
- Group/time-dependent CSV data need project-specific splits before fitting.
- Saved models use Python serialization: load only artifacts you trust.

For automated verification, run `python scripts/run_notebooks.py`. This executes
the notebook code in an explicit local kernel and retains outputs and timings.
