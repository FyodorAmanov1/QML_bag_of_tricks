# Prespecify the evaluation, not just the model

A useful result must survive the deployment split and a credible comparator.
`evaluate_project.py` wraps the existing inner training/validation/test workflow;
it freezes the complete run/split plan before the first fit. Each completed run
is independently audited against its original CSV and saved artifacts.

## Repeated IID or group holdouts

```json
{
  "data":{"split":{"kind":"group","group_column":"entity_id"}},
  "evaluation":{"kind":"repeated_holdout","seeds":[7,17,27],"baseline":"linear","practical_margin":0.01}
}
```

This fragment supplements a full project config. Choose 2–20 distinct seeds
before reading results. Every run reconstructs a disjoint inner train/validation/
test split. Group identities remain disjoint within each run; repeated runs can
reuse groups. Fitted preprocessing is local to that run's training data, and
candidate selection uses only its validation partition.

## Expanding chronological windows

```json
{
  "data":{"split":{"kind":"chronological","time_column":"date","gap_rows":3}},
  "evaluation":{"kind":"walk_forward","n_splits":3,"test_rows":12,"validation_rows":12,"min_train_rows":24,"baseline":"linear","practical_margin":0.0}
}
```

Each fold trains on a growing prefix, validates on a later window and tests on
the next window, with the specified row gap on both boundaries. Test windows
do not overlap. Previous test observations may enter a later fold's training
history when they would have become available in deployment. There is no
automatic lag engineering, target-overlap purge or feature-vintage recovery.
Set the embargo from the target horizon and information availability. Repeated
timestamps/panel data require a dedicated grouped temporal adapter.

The window unit is a record, not necessarily a calendar month. This backtest
does not recreate the published volatility protocol, which has its own target,
rolling-window length, candidate history and source-recovery obligations.

## Inspect every outcome

`evaluation_plan.json` freezes seeds, row IDs, baseline and practical margin.
Each run retains all trial history, selected predictions and source snapshots.
`summary.json` and `REPORT.md` report per-family means, descriptive sample SDs,
per-run paired differences, missing pairs and test-record overlap. A failed run
or family is retained and the evaluation is marked partial. The displayed mean
may use fewer runs; its completed/expected count must accompany it.

Positive paired improvement favors the row method: QML-minus-baseline for balanced
accuracy, baseline-minus-QML for RMSE. Report the complete vector, not just how
often it is positive. The default linear control is a reference, not necessarily
the strongest domain model; prespecify a stronger comparison where appropriate.

Overlapping resamples are not independent experimental replications. Nonoverlapping
time windows can remain dependent. This tool deliberately does not label a
bootstrap over seeds as an independent confidence interval, provide a naive
p-value or select a deployment winner from outer test scores. For a confirmatory
claim, use a justified independent unit, appropriate paired/cluster/temporal
analysis, a multiple-comparison policy and fresh external observations.

For an externally defined split inside one CSV, use `kind="predefined"` with
`train_ids`, `validation_ids` and `test_ids` referencing original row IDs. IDs
must exist, be unique and not overlap. The adapter validates record disjointness;
you remain responsible for entity/time independence of a manually supplied split.

The evaluation design follows the distinction between model selection and test
evaluation in the [official scikit-learn guidance](https://scikit-learn.org/stable/modules/cross_validation.html).
No repeated experiment changes the original bounded observations or establishes
a universal statement about a method family. See [contextual results](CONTEXTUAL_RESULTS.md).
