# Coverage ledger

**Cutoff:** 9 November 2026. **Status:** initial curated implementation, not an
exhaustive systematic review or a complete set of theorem reproductions.

The catalog contains 19 guides: 17 trainability strategies, one explicitly
bounded encoding-design example, and one QNG optimization companion. The QNG
entry is not a claimed general barren-plateau remedy. Most native
examples are explicitly bounded mechanism demonstrations. Their presence in the
catalog does not certify the published theorem's applicability to every execution.
The local-loss and correlated-product special cases admit direct analytical
checks. Each guide states remaining obligations for the other implementations.

## Discovery and screening

Discovery consulted the reference trails and mitigation taxonomies in
[Larocca et al.](https://arxiv.org/abs/2405.00781),
[Gelman](https://arxiv.org/abs/2406.14285), and
[Cerezo et al.](https://doi.org/10.1038/s41467-025-63099-6), with targeted searches
for Gaussian, symmetry, isometric tensor, finite-local-depth, dissipation, and
2025–2026 noise results. Guides link original publications. This was targeted
searching rather than an exported, exhaustive forward-citation search; coverage
cannot be certified complete through the cutoff.

Admission requires a peer-reviewed theoretical result for a specified construction.
The actual code has a separate implementation-status field. Unverified extrapolation
from an architecture name to a full learning pipeline is not admitted as a proof.

## Adjacent methods and unresolved coverage

| Candidate | Disposition and reason |
|---|---|
| Identity-block initialization | Now implemented with [independent parameter copies](strategies/identity_block_init.md) and a directly derived initial-gradient special case. It was deferred in the initial release's incomplete audit. |
| Layerwise learning, architecture search and ADAPT | Useful empirical methods; each needs its own qualifying bound and task-specific transfer audit. Not implemented as proven remedies here. |
| Classical pretraining and simulator warm starts | Now explicit [learning companions](learning/pretraining_surrogates.md), requested by the user; not promoted to general plateau remedies. |
| Kipu-style offline surrogate | Now a [separate teacher-first workflow](learning/kipu_workflow.md). Accepts measured feature pairs, freezes the teacher, validates fidelity and exports a classical-only artifact. Native reference teachers are not proprietary DQFE replicas. |
| Quantum natural gradient | Now included as an [optimization companion](strategies/natural_gradient.md), with an exact-simulator FS metric and metric-cost accounting. It was previously excluded from standalone proven remedies; no general plateau cure is claimed. |
| Generic SPSA, higher-order or gradient-free optimization | Optimizer choice alone does not remove an exponentially suppressed information signal. See the [review](https://arxiv.org/abs/2405.00781), Section VII. |
| Generic error mitigation | Sampling overhead can remain exponential; not a universal noise-induced plateau cure. |
| Dynamic parameterized circuits | [arXiv:2411.05760](https://arxiv.org/abs/2411.05760) is a discovery lead; peer-reviewed version and exact reset construction need verification before admission. |
| General small Lie algebras | The design principle is covered through symmetry and matchgates; a separate general algebra-representation implementation is not bundled. |
| Local adversarial discriminators | Related local-loss argument, but no complete adversarial training implementation or dataset-level bound audit is bundled. |
| Quantum Boltzmann models | Gibbs-preparation and convex-training results need a separate algorithm/data-access audit; outside this executable collection. |
| Projected/fidelity kernels, frozen features, re-uploading, Nyström, classical surrogates | Now implemented in the [learning layer](learning/index.md), with purpose-specific guarantees and separate error/access accounting. |
| 2026 initialization preprints | Discovery leads only under the peer-reviewed-only policy; no universal conclusion is inferred from their titles or abstracts. |

## Requirements for promotion to a theorem reproduction

See the [fundamental-method audit](fundamentals_audit.md) for encoding versus
initialization, the previous omissions, and the QNG evidence distinction.

For every strategy marked a restricted demonstration: implement the exact
distribution and complete geometry; identify the theorem's derivative metric;
evaluate or bound its input/observable factors; establish the valid region with
constants; justify the full learning objective; and reproduce the relevant
published finite-size calculation. Add independent checks and update the status.

Community additions must preserve negative findings and identify classical
simulation shortcuts. The project is designed to grow without claiming that
every known method has already been captured.
