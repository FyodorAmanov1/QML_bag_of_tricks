# Validation record

Verified locally on Windows with CPython 3.12.3 on 2026-11-09.

| Check | Observed result |
|---|---|
| Clean virtual environment, pinned dependency closure | All reference dependencies installed; `pip check` reports no broken requirements |
| Native SDK suite | 168 tests passed, including probability agreement across Qiskit, PennyLane and Cirq |
| Analytical checks | Product-state objective/gradient formulas, exact variance quadrature, tied-parameter derivative |
| Structural checks | Excitation conservation and deliberate control leakage; permutation-equivariant output |
| Sampling and reporting | Finite-shot consistency, exact/sampled evaluation accounting, serializable results |
| Ensemble sampling | Fixed Haar circuit factors are resampled along with initialized angles |
| Fundamental choices | Near-identity encoder bound; independent-copy identity cancellation and gradient formula; FS metric, global-phase invariance, QNG update and singular-direction regularization |
| Documentation | `python -m mkdocs build --strict` passed; 19 circuit guides plus 14 learning guides; 99 native entrypoints total |
| Reference runs | 57 one-step four-qubit runs; configurations, SDK versions, and source hashes recorded |
| Learning reference | 42 additional native entrypoints; classification, regression, known-preparation quantum and generative benchmarks with three seeds |
| Surrogates and pretraining | Saved-model parity, zero quantum-backend inference for the student, held-out kernel error, same-hidden-feature classical control and explicit warm-start costs |
| Kipu-style offline protocol | Frozen trained teachers in all three SDKs, real-feature-pair input, fidelity-gated export, classical-only reload and separate teacher/student task accuracy |
| Executed notebooks | All three classification, regression and offline-distillation notebooks executed without cell errors; timings in `results/notebook_execution.json` |
| Small real-data study | 240 selected test evaluations: three datasets, ten seeds, eight models and three validation candidates; zero failed final evaluations |
| Agent implementation tasks | Five scripts independently re-executed and passed artifact checks; three agents, two follow-up tasks; not a general capability estimate |
| Packaging | 0.2.0 wheel/source archive built; wheel imported independently of the source tree and ran a native circuit plus learning model; source archive contains three notebooks and five agent solutions |

Reference versions: Qiskit 2.3.1, Qiskit Aer 0.17.2, PennyLane 0.45.1, Cirq Core
1.7.0, NumPy 2.2.4, SciPy 1.15.2, scikit-learn 1.9.1. See `requirements-reference.txt` for the
platform-specific installed dependency closure.

The clean environment was created without system-site-packages. Pip caching
was disabled because its cache writer stalled in this workspace's restricted
environment. Reproduction commands:

```powershell
python -m venv .venv-clean
.\.venv-clean\Scripts\python.exe -m pip install --no-cache-dir setuptools wheel
.\.venv-clean\Scripts\python.exe -m pip install --disable-pip-version-check --no-cache-dir --no-build-isolation -c requirements-reference.txt -e ".[all,learning,notebooks,dev,docs]"
.\.venv-clean\Scripts\python.exe scripts/run_notebooks.py
.\.venv-clean\Scripts\python.exe -m pytest -q
.\.venv-clean\Scripts\python.exe scripts/grade_agent_tasks.py
.\.venv-clean\Scripts\python.exe -m pip check
.\.venv-clean\Scripts\python.exe -m mkdocs build --strict
```

The real-data study uses small fixed subsets and ideal simulation. The tested
classical configurations outperform these quantum-feature configurations on mean
scores; it establishes no general ranking of QML. Its repeated splits overlap,
so bootstrap intervals are descriptive rather than independent-replication
confidence guarantees. Full trial histories and warnings are retained.

Linux CI is configured but was not executed locally. Hardware jobs, production
validation, exhaustive literature coverage and full theorem reproductions are
not established by these checks. Scientific review is pending with the
repository owner; see `docs/owner_review.md` and `docs/coverage.md`.
The GitHub repository is private for owner review. No live Pages site or public
release is claimed.
