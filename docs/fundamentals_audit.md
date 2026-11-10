# Fundamental-method coverage audit

The original collection had a coverage gap: foundational encoding/initialization
choices and useful optimizers were not all given dedicated implementations.
This audit distinguishes omission from a deliberate limit on scientific claims.

| Requested method | Previous coverage | Current coverage and explanation |
|---|---|---|
| Local cost functions | Already included | [Guide and all three SDKs](strategies/local_loss.md), with exact local/global variance formulas and tests. No implementation was missing. |
| Near-identity data encoding | Only indirectly discussed through Gaussian initialization | [Dedicated encoder guide and three SDKs](strategies/near_identity_encoding.md). Data encoding and trainable-parameter initialization are different operations; the original taxonomy did not make that distinction visible enough. |
| Identity-block initialization | Named as an outstanding item in the coverage ledger | [Dedicated construction and three SDKs](strategies/identity_block_init.md). It was deferred during the initial incomplete proof-to-code audit, not excluded because the method lacks value. The independent-copy circuit and a rigorously derived initial-gradient special case are now implemented and tested. |
| Natural gradient descent | Mentioned only among methods that do not generally cure barren plateaus | [QNG guide and three SDKs](strategies/natural_gradient.md). The proven-remedy filter excluded it as a standalone cure. That filter should not have prevented useful optimizer coverage; it is now explicitly an optimization companion, without a general plateau-avoidance claim. |

The encoder guide cites data-encoding trainability research while explicitly
deriving its own product-state example. It does not attribute a universal
near-identity-encoding theorem to that source. Identity blocks cite the
[published initialization strategy](https://doi.org/10.22331/q-2019-12-09-214).
QNG cites the [published geometric method](https://doi.org/10.22331/q-2020-05-25-269).

## How to combine these methods

Choose and validate the feature encoding; initialize independent shallow blocks
near identity where the task permits; use a faithful local objective; then select
an optimizer appropriate to the remaining geometry and measurement budget.
Recheck gradients for the complete learning objective. A good encoder, a large
initial derivative, or an improved optimization step does not by itself prove
generalization or a computational advantage.
