# Positive findings that still need stronger replication evidence

These are not silently promoted into the agent selector.

## Clinical quantum-kernel stacking

[Daga et al., Scientific Reports (2026)](https://www.nature.com/articles/s41598-026-56928-1)
reports Parkinson's specificity 0.813 versus 0.585 for its classical RBF control,
with recall above 0.95, and a diabetes recall gain. Breast-cancer comparisons do
not show a significant gain. The proposed method uses nested out-of-fold stacking
of several encodings and classifiers; it is not a generic VQC. The small named
datasets and metric trade-offs are the tested conditions, not sufficient
conditions for a new medical dataset. A matched classical stacking pipeline,
threshold tuning, group-independent splits, raw out-of-fold predictions and
complete circuit costs are replication obligations. This is benchmark research,
not clinical validation or a recommendation to deploy a diagnostic tool.

## Existing repository application leads

Satellite-image multitask learning offers a specific reported small-sample
comparison; ADMET has a positive endpoint among otherwise unfavorable endpoints.
Neither supports an unconditional "small-data QML" recipe. Hybrid entity matching
requires a classical-routing ablation, and transferred-attack resilience requires
a declared threat model and quantum white-box controls. Exact classical product-
kernel equivalents must remain visible.

The old reports are retained. The new workflow prioritizes task-specific evidence
and full replication rather than using their generic reduced adaptations as
default training recipes.
