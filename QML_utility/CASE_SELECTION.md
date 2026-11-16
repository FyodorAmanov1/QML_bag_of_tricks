# Which result should an agent try to transfer?

"Confirmed" here means a result reported in a peer-reviewed publication within
its stated comparison. It does not mean independently reproduced in this folder
or guaranteed on a new project. All ten records retain their evidence type and
local implementation scope in [the catalog](catalog.json).

| Your actual problem | Replication candidate | Condition worth testing | First experiment that could support transfer |
|---|---|---|---|
| A long stream of changing image tasks; preserving ability to learn the next one matters | [Continual QNN](evidence/continual_plasticity.md) | Persistent compact-unitary model and complete amplitude inputs under continued updates | Original long CIFAR trajectory, then matched L2 and plasticity-aware classical controls; report forgetting separately |
| Several unaligned graphs plausibly sampled from one latent graphon | [QGNR](evidence/quantum_graphons.md) | Re-uploaded coordinates and permutation-invariant transport loss | Recover original reconstruction errors at the same resolution; compare classical oscillatory representations with the same solver |
| Small molecular descriptor samples with continuous activity labels | [Fixed molecular reservoir](evidence/molecular_reservoir.md) | Nonlinear fixed features may retain useful geometry when subsamples shrink | Recover 18-feature/MMACD protocol; compare full and selected classical features, train-only selection, and a fixed-test learning curve |
| Short sentiment sentences with affordable token-level circuit evaluation | [QSANN](evidence/quantum_self_attention.md) | Quantum projected features with classical Gaussian attention | Replay short-text comparison; swap in classical features with the same attention, and include TF-IDF/frozen-encoder controls |
| Historical monthly time series with a one-step volatility target | [Ising reservoir](evidence/volatility_reservoir.md) | Input/memory dynamics under causal rolling refits | Audit calendar and selection; walk-forward evaluation with equally informed classical reservoirs and regularized temporal models |
| You want a controlled demonstration of interference-dependent kernel geometry | [Photonic kernel](evidence/photonic_interference_kernel.md) | Designed labels align with a geometry difference between kernels | Recover all generated tasks; retain distinguishable-photon and exact classical optical simulation controls |
| You possess quantum state copies, and an observable-magnitude query arrives later | [Two-copy Bell acquisition](evidence/late_query_bell.md) | Coherent joint acquisition preserves query-relevant statistics | Matched-copy late-query protocol, optimal inference on specified local records, and query-known-early ablation |
| You can probe a multimode optical process and retain EPR references | [Photonic process learning](evidence/photonic_process.md) | Joint CV acquisition changes the sampling requirement | Replay the exact published hypothesis test with losses, squeezing and mode/sample counts |
| You can query a Pauli channel and retain calibrated reference qubits | [Entanglement-enhanced channel learning](evidence/pauli_channel_learning.md) | Commuting Choi observables plus auxiliary-noise mitigation | Start with signed-coefficient identities, then recover SPAM-robust depth fits and full calibration costs |
| Labels actually follow the special cryptographic construction and fault tolerance is available | [Discrete-log kernel](evidence/discrete_log_kernel.md) | Efficient quantum arithmetic under a stated classical-hardness assumption | Verify the input distribution, oracle/arithmetic access and resource analysis; a tiny modulus cannot demonstrate speedup |

## Priorities and limits

For an ordinary classical-data project, continual learning and graphon
reconstruction remain the first full-replication targets here. Their task
structure is explicit. Molecular reservoirs, short-text attention and volatility
add application-specific leads; their source gaps, comparator limits and modest
or conditional benefits make strong-control replication especially important.
The optical kernel case is a designed experiment, not a shortcut to success on
external labels. Quantum acquisition cases require physical input access that a
classical dataset does not supply.

None of these papers establishes that your data conditions are sufficient for
superiority. Before committing substantial training resources, define a measurable
transfer hypothesis, inspect artifacts, and freeze the comparison using
[the agent workflow](AGENT_WORKFLOW.md). For a non-winning local method use
[the contextual follow-up](CONTEXTUAL_RESULTS.md), which leaves other training
and data regimes open without assuming they will succeed.
