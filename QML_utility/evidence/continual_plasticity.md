# Continued adaptation: the strongest classical-data replication target here

**Primary result:** Chen & Zhang, PRX Quantum 7, 033003 (2026),
[paper](https://journals.aps.org/prxquantum/pdf/10.1103/ry52-85ll),
Figure 2 and Appendix G/Figure 11. No graph-digitized accuracy is asserted.

The reported benefit is preserving new-task learnability across a long random
binary CIFAR-100 sequence. Images become 1024 grayscale coordinates, L2-normalized
for both model families. The paper uses ten-qubit amplitude input and deep
two-qubit-unitary circuits; tested MLPs include L2 weight-decay variants.
Both families still forget old tasks without additional stabilization.

## Implementation contract

Use the [original-data protocol](../protocols/continual_cifar100.json). Retain
task order and original train/test images, preserve all coordinates and carry
parameters between tasks. Follow Appendix A.2 for the matched five-epoch Adam
schedule. The L2 ablation has a different schedule; do not merge its settings.

`models/*_impl.py` implements the paper's exponential-map SU(4) block and
last-logical-qubit Z readout. It is a circuit reference, not a trained model.
Read [the pinned source audit](../SOURCE_AUDIT.md): upstream logits and the
paper's description need reconciliation before trajectory comparison.

Measure accuracy on each new task, prior-task retention, weight/gradient norms,
loading, finite-shot estimation and wall time. Add separately tuned L2 MLPs,
classical orthogonal/unitary networks, replay/reset and continual-backprop controls.

**Use when:** the target is learning through sustained distribution changes.
**Do not infer:** superior one-task accuracy, universal plateau avoidance, or
that every classical continual learner is beaten. The geometry explanation is
a mechanism hypothesis for transfer, not a dataset-level guarantee.
