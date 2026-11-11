# Upstream alloy artifacts

From Zongrui Pei and collaborators' repository:
https://github.com/peizong/quantum-machine-learning-metallurgy

Pinned commit: `406f405980456c69213eae16d5df257f3fce0323`.
Paper: https://doi.org/10.1016/j.matt.2024.05.035.

- `data.csv` is the original `simulator_data_process/alloy-property-avg.csv`.
- `published_8.npy` and `published_10.npy` are the original sequential NumPy
  archives `simulator_data_process/res/zz_1.2_8.npy` and `zz_1.2_10.npy`.
  Load each of their seven arrays with `allow_pickle=False`.
- `LICENSE.upstream` preserves the repository's MIT license and attribution.

Original paths, immutable download URLs, sizes and SHA-256 hashes are recorded
in `../sources.json`. The files are unchanged apart from possible Git text
line-ending conversion; the audit explicitly accepts LF/CRLF equivalents.
These are author-supplied inputs/reference results. They are not newly generated
data or evidence that the current implementation executed 300 new splits.
