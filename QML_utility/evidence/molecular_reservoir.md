# Molecular activity: fixed reservoirs and small-subsample robustness

[Beaulieu et al., JCIM 65, 8475–8485 (2025)](https://doi.org/10.1021/acs.jcim.5c00958)
reports molecular-activity regression from descriptors using fixed quantum
reservoir features. The published qualitative benefit is robustness as sample
size decreases, rather than a universally better molecular predictor.

The [publisher supporting information](https://doi.org/10.1021/acs.jcim.5c00958.s001)
specifies five MMACD problems, 100/200/800-record subsamples, SHAP selection of
18 descriptors and one-/two-body observable embeddings. Recover these conditions
before transferring the method to a new endpoint. UMAP separation is an analysis
of embeddings, not itself held-out regression superiority.

## Agent implementation

Use [the protocol](../protocols/molecular_reservoir.json). Reconstruct the fixed
dynamics, pulse encoding, observable order and classical readout. Fit supervised
feature selection inside training data in a separately labeled clean extension;
retain the exact original replay alongside it. Compare full- and selected-feature
classical models on identical records, including GPR/RBF, SVR and tree ensembles.

Run a learning curve with a fixed held-out test set and scaffold/group splits
where identifiers allow. Separate feature-acquisition cost from fitting the
classical head. Fixed dynamics avoid quantum-gradient training; they do not
automatically give cheap preparation, low sampling cost or superiority.

## Feature-acquisition recipe and unresolved scope

The SI's Section S2.5 prescribes the following feature pipeline: scale the selected
descriptor vector to detunings in [-6,+6] MHz; assign one descriptor per atom;
initialize the register to zero; construct the Rydberg Hamiltonian; evolve and
collect the specified observable expectations at 0.4-microsecond intervals;
concatenate the values into a classical feature vector and fit the selected
regressor. This trains the readout rather than variational quantum parameters.
The main-text geometry, Rabi pulse, total duration and complete observable list
must be recovered before writing an exact SDK/analog implementation.

The SI describes standardization before splitting and supervised feature selection
before making the small subsamples. Audit which records each operation sees.
An exact historical replay and a clean nested-selection extension are different
experiments; do not describe a leaked pipeline as deployable generalization.
Classical spin-reservoir features are also a relevant domain-specific control.

**Source consistency:** final SI Section S2.4 identifies MMACD14 for detailed
analyses, while other SI prose and its S1 caption say MMACD4; the precursor also
uses MMACD4. Resolve this with original artifacts or authors rather than silently
choosing an identifier. The protocol records the contradiction.
No figure-derived regression numbers or locally reproduced molecular wins are claimed.
