# Published evidence and implementation access

**A peer-reviewed result with available author code is a valid basis for using
or adapting a method. This repository does not require an independent rerun of
every paper before including its method.** Run a fair evaluation on your own
project before claiming that the method improves that project.

## Read the labels separately

| Label | Meaning |
|---|---|
| Published result | The cited peer-reviewed paper reports the stated result under its data, access, comparator and metric conditions. |
| Author source available | A linked author implementation exists; the inspected version and files are pinned where available. Availability alone does not grant reuse rights or certify agreement with every paper setting. |
| Runnable here | The specific local recipe or component can execute with the documented dependencies. Its scope may differ from the author's full pipeline. |
| Verified here | The named local tests or saved experiments were executed and checked. A circuit test verifies the circuit contract, not full training performance. |
| Independently reproduced here | A separately executed paper-scale comparison supports the stated local reproduction claim. This is an optional additional evidence level. |

The levels are not interchangeable. Lack of an independent rerun here does not
invalidate the paper's evidence or make the entire library unfinished. Conversely,
a paper citation alone does not make every task-specific implementation part of
this package, guarantee new-project performance or resolve missing source access.

## Where to obtain implementations

Five author repositories have pinned inspected files in
[the source manifest](upstream_manifest.json). Use the linked evidence guide for
the primary publication, exact pin, implementation scope and known differences.

| Method | Author implementation or access route | What this repository supplies |
|---|---|---|
| [Continual plasticity](evidence/continual_plasticity.md) | [Author CIFAR/MNIST training code](https://github.com/sxzgroup/quantum-plasticity/tree/c3c6cb4a91971d4fad6e468df6ac071851bb7460) | Native SU(4) references and original-data settings. The public CIFAR script defaults to 30 tasks; the paper's main trajectory is 3,000. |
| [Quantum graphons](evidence/quantum_graphons.md) | [Author QGNR source](https://github.com/QuaRobot/QGNR/tree/97ee491cecf8d8ce9847a3503735117b89977a71) | Source inspection and reconstruction/transfer guidance; repository-wide reuse terms and the downstream classification pipeline need resolution. |
| [Quantum self-attention](evidence/quantum_self_attention.md) | [Paddle Quantum application](https://github.com/PaddlePaddle/Quantum/blob/760c787f29e18db77ec6fba5e91da6166ae6599e/applications/text_classification/qsann_classification.py) | Tested Gaussian attention aggregation and a comparison protocol; paper/code convention differences are recorded. |
| [Photonic kernels](evidence/photonic_interference_kernel.md) | [Author code and archived tasks](https://github.com/dapingQ/PhoQuKs/tree/9a04579985d6c653b89b3de6b6699ef857067e0c) | Kernel/engineered-label recipe and explicit classical-simulation controls. Author artifacts are not an experiment conducted by this repository. |
| [Volatility reservoirs](evidence/volatility_reservoir.md) | [Author Julia source and archived predictions](https://github.com/LeeQY1996/Quantum-Reservoir-computing-for-Realized-Volatility-Forecasting/tree/d2e9b0aa599afd33c6cfb10352e2c36fe92fa76f) | Forecast/data-access guidance; referenced local packages and reuse terms require review before execution. |
| [Photonic process learning](evidence/photonic_process.md) | [Institutional code/data archive](https://doi.org/10.11583/dtu.29517107.v1) | Access model and archive link; no file-level source pin or optical acquisition implementation asserted here. |
| [Late-query Bell learning](evidence/late_query_bell.md) | Local native references and restricted benchmark | Three-SDK Bell acquisition and an executed, separately scoped local audit. No complete author experiment pipeline is pinned. |
| [Molecular reservoirs](evidence/molecular_reservoir.md) | Published paper and supporting information | Data/descriptor guidance; exact author regression artifacts have not been recovered here. This is an evidence lead, not an author-code-backed ready-to-run route. |
| [Pauli-channel learning](evidence/pauli_channel_learning.md) | Analysis code described as available on request in the inspected manuscript | Three-SDK ideal acquisition components; full device analysis is a separate access requirement. |
| [Discrete-log kernels](evidence/discrete_log_kernel.md) | Conditional theory and resource analysis | A theory boundary, not a practical ready-to-run advantage demonstration. |

Public source with missing or uncertain reuse terms should be inspected for
method understanding; resolve permission before copying or redistributing it.
Dependency stacks and specialized hardware remain case-specific requirements.

## Agent workflow: adapt by default

1. Match your task/data/access conditions using [case selection](CASE_SELECTION.md).
2. Read the primary publication and inspect the linked author code or the scoped
   local recipe. Reconcile relevant conventions and review reuse terms.
3. Adapt to your project with a declared data/split/budget contract. Preserve
   tuned classical controls, validation-only selection and held-out evaluation.
4. Report a **new-project result** with its actual outcome. Do not call it an
   independent paper reproduction unless you chose and completed that track.

`select_case` returns the primary publication and `implementation_access` for
each match. Set `"require_author_code": true` to restrict matching to the five
file-inspected public author-source routes. This does not certify their license,
runtime environment or exact agreement with the paper.

The planner defaults to `"intent": "adapt"`. Use `"intent": "replicate"` only
when you want to independently repeat the original experiment. Both routes keep
the case-specific caveats and require validation before deployment; an original
paper-scale rerun is not a prerequisite to an adaptation experiment.
