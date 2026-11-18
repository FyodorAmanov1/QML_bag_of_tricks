# Agent brief for a specific user project

Copy the block below into a coding-agent session with an authorized local clone.
Fill in the actual project facts; an agent should inspect available project files
before asking for information they already contain.

```text
Build and evaluate a QML candidate portfolio for this project:
- Data source/access and inference-time inputs: [fill in]
- Target, primary metric and minimum practically useful improvement: [fill in]
- Independent unit and deployment split: [IID/entity/time/other]
- Runtime, RAM, native SDK and physical shot/query budgets: [fill in]
- Deployment constraint: [native circuit/simulator/classical-only inference]

Read QML_utility/AGENTS.md, README.md and PRACTICAL_WORKFLOW.md. Keep all new
utility code/configs/results in QML_utility. Preserve the original repository's
strategies, generated sources and studies. Do not publish anything.

Read EVIDENCE_LEVELS.md. Peer-reviewed evidence and documented author code are
valid starting points; do not rerun a whole paper unless I request replication.
Default to adapting the implementation and testing my project. Keep the source
publication, code access, local verification and my new results separate.

Create an explicit project JSON. Run QML_utility.recommend. Distinguish exact
published-case matches from exploratory transfer candidates; do not convert a
method foundation or construction test into a published predictive win.

For supported numeric classification/regression, select several justified model
families and run QML_utility.run_project with fresh output paths and explicit
budgets. Preserve all feature coordinates unless a new declared representation
experiment warrants compression. Use the correct IID/group/time split, train-only
preprocessing and regularized/domain-specific classical controls. If the task is
graphs, raw text/images, continual learning or quantum acquisition, implement and
test an explicit adapter using EXTENDING.md; do not force it into a numeric CSV.

Inspect every candidate, failed trial, warning, feature/kernel diagnostic and
training trace. Consult COMPOSING_METHODS.md and companion_map.json for encoding,
local objectives, QNG, initialization, pretraining, shot budgets and surrogates.
Explain the diagnostic and assumption behind each proposed composition. Near-
identity encoding, local measurements and QNG are not universal success guarantees.

Freeze selection using validation, then inspect the held-out report. Audit saved
metrics and artifacts with QML_utility.audit_project. Return the selected model,
feature order, environment, source/data hashes, predictions, all tuning history
and costs. Keep a classical model if it better satisfies the declared objective.
Do not choose a different winner by looking at the test scores.

If a change was motivated by earlier test results, preserve that outcome and
use fresh uninspected data for confirmation, or label the follow-up exploratory.
For classical-only deployment, evaluate the existing offline feature-distillation
companion and QML_utility/OFFLINE_DISTILLATION.md; require held-out fidelity,
task accuracy/error and zero-query inference. For larger numeric datasets select
the explicit batched adapter, not a larger dense kernel by default. Prespecify
repeated/group holdouts or walk-forward windows using EVALUATION.md; retain all runs.
Never call Nyström a zero-query surrogate or claim to reproduce proprietary DQFE.

Report separately: verified implementation, observed predictive utility, physical
measurement/resource costs, and unresolved limitations. A single exploratory
holdout is not a general quantum-advantage claim.
```
