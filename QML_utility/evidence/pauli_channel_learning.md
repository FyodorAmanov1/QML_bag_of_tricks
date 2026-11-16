# Learning signed Pauli-channel coefficients with noisy quantum memory

[Seif et al., Nature Communications (2026)](https://doi.org/10.1038/s41467-026-75553-0)
reports error-mitigated entanglement-enhanced learning: hypothesis tests up to
64 total qubits and intrinsic-noise characterization up to 16 qubits. The stated
experimental per-qubit overhead is **1.33 ± 0.05**, below the entanglement-free
base **2**, under its learning-access assumptions. This is a sampling result,
not classification accuracy on classical arrays.

The inspected source is the peer-reviewed accepted publisher manuscript, marked
article in press. Final data/code recovery remains an explicit replication blocker.

The required inputs are channel queries and system-reference Bell pairs.
Commuting Choi observables `P tensor transpose(P)` encode signed Pauli eigenvalues.
Auxiliary-memory characterization and SPAM-robust depth fits are part of the
published method and cost.

## Agent implementation

Use [the protocol](../protocols/pauli_channel_learning.json). The three native
`pauli_channel_probabilities` functions test small ideal Pauli-channel identities.
`models/pauli_channel.py` decodes signed coefficients and checks calibration
division. These are building blocks, not a recreation of the full error-mitigated
device experiment or a verified fit to its data.

The transpose changes the Y convention compared with the two-copy state-square
method. Test X/Y/Z channels explicitly. Reject near-zero calibration factors,
include calibration covariance and experiment cost, and measure drift. Ask for
selected coefficients; printing all `4**n` values remains exponential. The
entanglement-free comparator can use controls within the system but lacks the
entangled external reference. Do not silently weaken that access model.
The ideal channel reference estimates signed coefficients directly. The real
gate-layer experiment has additional symmetrization and identifiability limits;
recover those definitions before comparing its estimates with a simulator.
