# Entanglement-assisted physical-process learning

**Primary result:** Liu et al., Science 389, 1332–1335 (2025),
[paper](https://preskill.caltech.edu/pubs/preskill-2025-quantum-learning.pdf).

The inputs are uses of an unknown multimode bosonic displacement process, not
stored tabular features. EPR probe/reference entanglement and joint continuous-
variable measurements reveal fine characteristic-function structure.

Keep two published comparisons separate:

- **100-mode reconstruction:** 11.8 orders of sample reduction against an
  extrapolated vacuum-probe reconstruction scheme. The paper explicitly explains
  why that reconstruction comparison alone is not its provable separation.
- **120-mode hypothesis test:** success `0.563 +/- 0.025` with `10^5` samples,
  compared with an entanglement-free bound `0.5 + 3.8e-11`; the paper reports
  a distinct 9.2-order equivalent-sample advantage.

## Replication route

[DTU data and analysis](https://doi.org/10.11583/dtu.29517107.v1) provides
separate code/data archives and licenses. Start by replaying those analyses;
do not simulate a tiny unrelated classifier and call it reproduction. The
publisher PDF's shortened dataset DOI is corrected here using the institutional
dataset record.

Record the process family, query range, number of modes, uses of the process,
squeezing, loss and statistical success criterion. Match the entanglement-free
access model; distinguish measured observations from extrapolated sample counts.

**Use when:** the project can prepare and retain optical probe/reference modes
and perform joint CV measurement. Ordinary Qiskit/PennyLane/Cirq gate demos do
not supply this acquisition setup. No optical experiment or raw-data replay has
been performed here; no general image/CSV prediction benefit follows.
