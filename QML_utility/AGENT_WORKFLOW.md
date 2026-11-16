# From a paper success to a project experiment

For a new user project, begin with [the practical workflow](PRACTICAL_WORKFLOW.md)
and `recommend.py`. It combines publication matches with separately labeled
exploratory model portfolios. The instructions below describe the exact
paper-replication track, whose unresolved artifacts remain explicit blockers.

Use the selector to identify a replication candidate, then generate a reviewable
contract. Neither command fits a model or certifies performance.

Start with [the task-to-case map](CASE_SELECTION.md) to judge which candidate is
worth the replication cost. The example profiles intentionally leave unknown
data provenance or compute budgets blank; the planner reports those omissions.
Even a complete draft requires resolution of the paper-specific artifact blockers.

```bash
python -m QML_utility.plan_project --profile QML_utility/examples/molecular_project.json --output QML_utility/results/local/molecular_plan.json
python -m QML_utility.plan_project --profile QML_utility/examples/wine_followup.json --output QML_utility/results/local/wine_plan.json
```

## Required project decisions

Specify data provenance and access; target and metric; IID/group/time split;
sample/feature counts; relevant structure; SDK or analog hardware; memory, time,
shots and tuning budget. A capability claim should have supporting artifacts,
not just a dataset name. Missing conditions mean *no matched evidence yet*.

## Execution sequence

1. Read the evidence card and exact source version. Separate published settings,
   local references, unresolved details and proposed extensions.
2. Recover original artifacts and first replay the paper comparator tuple. A
   construction test or short smoke run is not a paper-scale performance run.
3. Prespecify a new-project comparison. Fit preprocessing and model selection
   inside the training partition; use validation to select, then freeze.
4. Include classical controls that match information, architecture components
   and tuning access. Make the budget policy explicit rather than assuming the
   same epoch count means the same compute.
5. Save all trials, learning curves, split/record IDs, seeds, source/environment
   hashes, warnings, held-out predictions and resource estimates.
6. If no gain is observed, retain a scoped outcome and follow
   `CONTEXTUAL_RESULTS.md`. If a gain is observed, check uncertainty, leakage,
   ablations and transfer before promoting a recommendation.

## Deliverables

Provide a runnable script, a frozen experiment configuration, the source/artifact
manifest, all selection history, test predictions or physical estimates, a report
with effect sizes/uncertainty and a limitation statement. Distinguish predictive
performance, parameters, sample complexity, memory and end-to-end runtime.

Use `models/projected_attention.py` and native Pauli-channel references only for
their documented component contracts. They do not contain complete trained NLP
models, molecular reservoirs or large-device EMEEL pipelines.
