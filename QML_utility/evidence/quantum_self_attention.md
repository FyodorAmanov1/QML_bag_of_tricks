# Short text: projected quantum query/key/value features

[Li, Zhao & Wang, Science China Information Sciences 67, 142501 (2024)](https://doi.org/10.1007/s11432-023-3879-7)
reports QSANN improvements over simple classical self-attention and mean-embedding
baselines. The detailed [author manuscript](https://arxiv.org/pdf/2205.05625v2),
Tables I/III, uses balanced 1,000-sentence Yelp/IMDb/Amazon subsets, 80/20 splits
and nine parameter initializations. These are short sentiment sentences, not
the full IMDb corpus or a language-model benchmark.

Four-qubit circuits supply scalar query/key expectations and value features.
Gaussian weights `exp(-(q_i-k_j)^2)` are normalized classically. The reported
accuracy differences are modest and parameter counts omit embedding parameters;
do not call them end-to-end compression or wins over pretrained transformers.

## Agent implementation

Follow [the protocol](../protocols/quantum_self_attention.json) and the pinned
Paddle Quantum source, reconciling their differences first. Recover tokenization, vocabulary/OOV rules, padding,
embedding training, residuals and optimizer schedule before training. Keep data
split repetitions separate from nine initialization repetitions.

`models/projected_attention.py` implements only the measured-feature aggregation.
It performs no quantum query, training or sentence classification. Use it to
compare quantum and classical query/key/value maps with identical postprocessing.
Include TF-IDF regularized linear models and a frozen pretrained encoder head.
Count circuits per token and measurement basis, shots, classical attention memory
and embedding parameters. A positive new-project result needs untouched-test
evidence; the source's short-text win is a replication motivation.

## Model flow and source differences

For each token, prepare an encoded state and apply separate trainable query,
key and value circuits. Estimate first-qubit Z for scalar query/key features
and the declared Pauli features for values. Form all pairwise Gaussian weights,
normalize over valid keys, then aggregate values. Recover the paper's residual,
rescaling and pooling equations before applying its sigmoid classification head
and training objective. All-pairs aggregation is classical and retains quadratic
sequence-length storage in this reference implementation. Measured feature
acquisition and quantum-gradient estimates still require a shot budget.

The pinned author code is **not an exact implementation of the cited settings**:
at four qubits and embedding depth one, its token parameter array contains 24
scalars rather than the manuscript's dimension 12; its head initializes with
standard deviation 0.001 rather than 0.01; its generated value features also have
a different order/repetition. The code additionally multiplies aggregated values
by pi. Reconcile the full circuit/readout/residual contract and locate the correct
source version before reporting reproduction. A successful run of the public
application alone cannot resolve those differences.
