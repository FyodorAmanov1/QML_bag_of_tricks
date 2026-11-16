# Permutation-aware graphon reconstruction

**Primary result:** Qiao et al., npj Quantum Information 12, 8 (2026;
online 26 November 2025), [paper](https://www.nature.com/articles/s41534-025-01141-7),
Tables 1–2. QGNR combines re-uploading quantum coordinates with
Gromov-Wasserstein reconstruction. The benchmark uses 13 graphons, ten graphs
of 50–300 nodes per graphon, and ten trials. Single-graphon parameter counts
are 169 versus 921 for IGNR.

## Conditions and implementation

The object being learned is a shared continuous edge-probability function from
unaligned graph collections. Rapid variation and ambiguous degree ordering are
motivating difficulties. This is not ordinary node-feature classification.

Follow [the reconstruction protocol](../protocols/graphon_reconstruction.json).
Use the same transport loss/solver, graph trials, coordinate symmetry and
evaluation resolution. Preserve ties and uncertainty across all graphons.
Frequency support alone does not establish independently adjustable coefficients
or a training guarantee. Require permutation-aware Fourier, IGNR/SIREN and FNO
controls with matched tuning.

Table 4's separate graph-classification result is modest: QGNR-C means of
72.60/49.46/85.58/63.89% on IMDB-BINARY/IMDB-MULTI/MUTAG/PTC-MR, versus best listed
classical means 72.00/49.01/84.58/63.62%. These are not large generic GNN gains.

The inspected [author repository](https://github.com/QuaRobot/QGNR) exposes
single-graphon training, an older dependency stack, no recorded repository license,
and no verified conditional-classification pipeline. Sources are pinned and
inspected, not copied or executed here. Recover the missing pipeline before
claiming a Table 4 replication. Parameter savings are distinct from speedup.
