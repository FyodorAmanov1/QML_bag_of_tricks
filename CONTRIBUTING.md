# Contributing

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion use the project's Apache-2.0 terms, as described in its section 5.
Contributors retain their copyright. Preserve existing third-party notices,
identify imported material and its license, and credit original method authors.
See [licensing and citation](QML_utility/LICENSING_AND_CITATION.md).

Application reproductions and project benchmarks are welcome. Start from the
[Utility project workflow](QML_utility/PRACTICAL_WORKFLOW.md) and the
[QML motivation guide](docs/why_qml.md), select a concrete use case, and provide
the source paper, runnable code, data provenance and full comparison. Use the
application issue template to share a reproduction or a project result.

Start with the coverage ledger and select one concrete missing construction or
proof-to-code obligation. Report publication identity, theorem locator, exact
ensemble, metric, scaling variables, input/observable factors, and limitations.

Each strategy needs a guide, three readable native SDK implementations, common
numeric fixtures, resource accounting, a control with its differences explained,
and independent verification. Architecture names do not certify theorem hypotheses.
Mark simplified demonstrations as such. Retain negative findings.

Run `python -m pytest` and `python -m mkdocs build --strict`. Include exact
dependency versions and commands in reports. Avoid accuracy or advantage claims
from a few simulator sizes. A new kernel, nonlinear loss, noise model, parameter
sharing rule, or dataset may require a new mathematical argument.

`scripts/assemble.py` is the authoring source for generated guides, catalog rows,
and SDK modules. Learning guides, native backends and recipe entrypoints use
`scripts/assemble_learning.py`. Update the corresponding generator, regenerate, and inspect
the outputs. Other source files are maintained directly.
