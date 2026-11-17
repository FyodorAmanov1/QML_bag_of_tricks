# What a non-winning experiment does and does not tell an agent

A result belongs to its dataset, split, feature map, training procedure, budget,
measurement model, comparator and metric. It does not belong to a method name
for all time. `experiment_outcomes.json` retains five such bounded episodes.

For pinned text sources, `source_sha256` retains the originally audited byte
checksum. The additional `source_sha256_lf` records the same content with CRLF
converted to LF, so Linux and Windows checkouts can verify it consistently.
Verification allows only that line-ending conversion; result content and the
original provenance hashes remain preserved.

For example, the old real-data study used 128-row subsets of wine, breast-cancer
and diabetes data; two PCA coordinates, two qubits, one layer and three candidate
settings per quantum model. Its unfavorable outcomes apply to those pipelines.
They do not rule out better-trained models, task-aligned encodings or other data.
They also do not establish that simply increasing training would fix them.

## A responsible follow-up

1. **Diagnose before scaling.** Compare training and validation curves. Check
   gradients, observable sensitivity, kernel variation and input information loss.
2. **Name the hypothesis.** Distinguish optimization limitation, representation
   mismatch, statistical overfitting and information/access limitations.
3. **Change a declared axis.** Examples include training budget, learning rate,
   initialization, encoding scale, depth, readout, retained features or a new
   structurally motivated dataset. Record the alteration as a new experiment.
4. **Match classical controls.** Keep both same-feature and full-feature controls,
   regularization and a disclosed tuning/compute policy.
5. **Protect evaluation.** Select using training/validation, freeze the decision,
   and use an untouched test. Preserve every configuration, not only the winner.
6. **Report the bounded conclusion.** Use the exact data and protocol in the
   statement and list untested alternatives separately.

If earlier test scores motivated the redesign, reuse of that same test is
exploratory. Keep it for comparability, but reserve fresh, uninspected observations
for confirmation. A new random seed on repeatedly inspected data does not erase
that selection history. If fresh data are unavailable, disclose that limitation
instead of labeling the redesigned comparison confirmatory.

More effective training may help optimization-limited fitting. It cannot recover
coordinates discarded by PCA, add temporal information absent from an input,
or break an exact classical equivalence of an unchanged kernel. More training
can also overfit or increase hardware cost. These possibilities need diagnostics.

## Useful wording

> No advantage was observed for these configurations on these data under this
> budget and comparator set. Modified training or differently structured data
> remains untested. The next experiment will test [specific hypothesis], with
> [prespecified controls and acceptance criterion].

Avoid both "this method is useless" and "this will work with enough training".
Likewise, a positive result should name the same scope. Absence of significance
does not prove equivalence, and an observed mean gain does not prove universality.

[Prospective follow-up protocol](protocols/contextual_followup.json) ·
[Agent experiment workflow](AGENT_WORKFLOW.md).
