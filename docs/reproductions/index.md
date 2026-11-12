# Reproduce QML papers on classical application data

Explore QML through two published applications with classical inputs: materials
screening and nonlinear communication-channel recovery. Fresh numerical runs
recover the original comparator gains: **89.12% vs 87.73% accuracy** in the
ten-qubit alloy replay, and **0.313% vs 4.437% symbol error** in equalization.
Use these as concrete starting hypotheses for a project.

| Study | Application and accessible artifacts | Local reproduction scope |
|---|---|---|
| [Pei et al., Matter (2024)](alloys.md) | Real alloy descriptors and phase labels; author code and 300-split score archives | Archive recalculation, fresh native circuit replay on ten splits per architecture, and a leakage-free tuned-baseline extension |
| [Hu et al., Nature Communications (2024)](communications.md) | Simulated classical communication symbols through a noisy nonlinear channel; author notebook and numerical reference curves | Six-qubit connected-reservoir simulation at 20 dB with three seeded datasets, original comparator and stronger temporal controls |

Stronger tuned classical alternatives lead in the follow-up comparisons, also
included in the [measured results](results.md). A successfully recomputed archived
number, a new numerical replication, and superiority to newly tuned baselines
are distinct evidence levels. Neither study is reproduced end-to-end here:
alloy-candidate discovery, language-model screening and quantum hardware jobs
are excluded. No result is required to favor QML.

## Run from a source checkout

```bash
python -m pip install -e ".[qiskit,learning,notebooks]"
# Optional refresh of byte-hashed artifacts from pinned author commits:
python scripts/fetch_paper_sources.py
python -m qml_tricks.learning.paper_reproductions --study alloys
python -m qml_tricks.learning.paper_reproductions --study communications
python scripts/match_communication_training_budget.py
python scripts/build_paper_report.py
python scripts/audit_paper_results.py
```

The alloy data, license and small score archives are included under
`reproductions/alloys/`. Sources and byte hashes are listed in
`reproductions/sources.json`; the configuration was fixed in
`reproductions/protocol.json` before the experiments. Downloaded upstream code
stays in the ignored cache and is not executed by the fetch script. The NISQRC
implementation was written from the mathematical model and inspected protocol;
its upstream notebook is not redistributed as licensed repository code.

Use `--framework pennylane` or `--framework cirq` for the alloy state preparation
after installing that extra. The reservoir production path uses SciPy exact
linear algebra, with one-step native SDK equivalence checked independently;
changing this flag does not turn that path into a hardware experiment.
