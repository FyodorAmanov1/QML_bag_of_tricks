# First full classical-data replication brief

Copy this into an agent session with an isolated training environment and an
explicit compute budget. It is a project specification, not an already completed run.

```text
Replicate the Split CIFAR-100 continued-adaptation comparison described in
QML_utility/evidence/continual_plasticity.md and protocols/continual_cifar100.json.
Budget: [GPU/CPU, memory, maximum hours]. Do not publish anything.

Read QML_utility/AGENTS.md, EVIDENCE_POLICY.md and SOURCE_AUDIT.md first.
Use the immutable author commit in upstream_manifest.json. Inspect the original
paper and code before executing; reconcile the stated and implemented logits,
SU(4) parameterization, task count, train/test data and training schedules.

Preserve original grayscale/L2 inputs with all 1024 coordinates, original
train/test identities, sequential binary class-pair tasks and persistent weights.
First reproduce the original trajectory at the target scale. A short smoke run
is allowed but must be labeled a smoke run and cannot establish the paper claim.

Then add separately tuned L2 MLP, orthogonal/unitary classical and plasticity-aware
controls under declared data/tuning/compute budgets. Freeze selection without
test inspection. Report both new-task adaptation and old-task retention.

Save commits/diffs, environment, data hashes, task orders, every trial and seed,
held-out predictions, trajectory statistics, uncertainty, loading/readout/shots,
time and memory. Keep failures. Do not replace a failed result with weaker controls.
If the original artifacts or budget are insufficient, report the precise blocker.
Only mark a local success after independently checking the full claimed comparison.
```
