# Source and implementation audit — 16 November 2026

Pinned commits and inspected-file SHA-256 values are in `upstream_manifest.json`.
Only public author sources were inspected. They were not executed or vendored.

## Continual plasticity

- Author repository: `sxzgroup/quantum-plasticity`; Apache-2.0 recorded.
- `src/cifar.py` has shared grayscale/L2 preprocessing, persistent models,
  matched Adam settings and the original-data task construction.
- Its default depth is 10. A default script is not the depth sweep or the L2
  ablation suite reported by the paper. Dependencies are not version-pinned.
- Source returns two logits `[z, -z]`. Their softmax corresponds to a `2z`
  logistic scale; the paper describes a Z logit. Record/reconcile this before
  asserting exact reproduction. The references here implement the paper's
  exponential-map SU(4) definition, not an assertion about TensorCircuit's
  internal `su4` decomposition.
- A matrix exponential of a sum of noncommuting generators is not generally
  separately periodic in each raw coefficient. Do not silently wrap each
  coefficient modulo 2π or treat compact unitary outputs as a proof that every
  gradient stays large. The local tests include this distinction.
- The MNIST schedules differ between the two model families in Appendix A.1;
  use the matched CIFAR experiment as the first comparison.
- Classical unitary/orthogonal and plasticity-aware algorithms are required
  follow-up controls, not results claimed to be in the authors' original table.

## Graphons

- Author repository: `QuaRobot/QGNR`; no license recorded in the inspected tree.
- Public paths include single-graphon training and a roughly 14 MB graph archive.
  No conditional graph-classification entrypoint was found in that tree.
- README stack includes TorchQuantum 0.1.7 and Qiskit 0.38.0. Do not install it
  into this repository's Qiskit 2.3.1 environment.
- Empty `ppath` plus `'/Result/'` constructs a filesystem-root output directory.
  Review and redirect it inside a dedicated checkout before any execution.
- Data loading uses pickle. Establish artifact trust before deserialization.

## Interpretation

These gaps reduce replication readiness; they do not by themselves refute the
paper. They are why this folder reports *published successes with replication
obligations* rather than declaring locally validated universal winners.

## Molecular reservoirs

- The peer-reviewed JCIM abstract supports small-subsample robustness. The final
  [publisher SI](https://api.figshare.com/v2/articles/29832777) was inspected and
  its PDF hash retained in the protocol. Main-text pulse settings and exact
  regression artifacts remain recovery requirements.
- Final SI Section S2.4 names MMACD14 for detailed analyses, while other SI prose,
  the S1 caption and the 2024 precursor use MMACD4. This is an unresolved internal
  inconsistency, not a recovered dataset identity. Do not merge their tables or treat UMAP/classification diagnostics as
  regression MSE. No visually estimated regression scores are recorded.
- The related QuEra QRC tutorials are not the exact published molecular pipeline.
  Their existence does not establish a complete reproduction path. The SI's
  named preprocessing/regression scripts have not been recovered here.
- The described preprocessing orders standardization before a split and SHAP
  selection before small subsampling. Audit outer-test exposure. A clean nested
  extension is necessary before transfer recommendations; disjoint train/test
  records alone do not rule out selection leakage.

## Projected text attention

- Author Paddle Quantum implementation is pinned as `attention`. The inspected
  files include the application entrypoint and `paddle_quantum/qml/qsann.py`.
  GitHub's license metadata returns `NOASSERTION`; that is not a license grant.
  Both inspected Python files carry Apache-2.0 notices; repository-wide license
  scope was not separately established or used to vendor code here.
- Tables I/III in the corresponding author manuscript provide detailed settings
  and short-sentence scores. Their source version is explicit in the protocol;
  reconcile against the journal version before claiming an exact replay.
- The small CSANN comparison does not establish a win over pretrained encoders.
  Embedding parameters are excluded from the cited parameter counts. Data-split
  uncertainty is different from the nine initialization repeats.
- Public code at four qubits/depth one has a token parameter array of 24 scalars
  rather than paper dimension 12, and head initialization std 0.001 rather than
  0.01. Observable order/repetition, pi rescaling and residual conventions also
  require reconciliation. Do not equate the public application with exact paper
  reproduction merely because it bears the same model name.

## Interference-aware photonic kernels

- Author `dapingQ/PhoQuKs` is pinned as `photonic_kernel`; MIT recorded.
  README pins a particular DisCoPy optics commit. The inspected simulation file
  imports `geo_diff`, while the inspected helper is `data_utils.py`; that helper
  also loads local pickle/NumPy archives at import time. No author code was run.
- Targets are engineered from both full-task kernel matrices. The return-Fock
  probability of the composed interferometers is the kernel; substituting a dot
  product of output probabilities would change it.
- Selected-task panels and five-task averages have different scopes. Recover
  hyperparameter selection and all archived tasks before a replay; do not infer
  missing accuracy values from chart positions. Archived objects were not loaded.

## Volatility reservoirs

- The author's Julia source and archived predictions are pinned as `volatility`.
  README says "Coming soon!"; source references local packages not recovered in
  the inspected files. No license is recorded and no notebook was executed.
- Forecast calendar, scaling fit scope, feature-selection chronology, revisions
  and full tuning histories need auditing. Archived predictions alone do not
  establish clean out-of-sample execution.
- Paper reports the best of 100 reservoir instances and states training-window
  time-series-aware validation. Recover the exact candidate/selection history;
  report all trials and ensure no forecast-test selection in a clean extension.
- Preserve the paper's reversal between one- and five-step forecasts and the
  classical methods in the model confidence set. Do not turn MCS membership
  into unique superiority or a forecast error into trading returns.

## Pauli-channel learning

- The Nature Communications accepted publisher manuscript is peer-reviewed but
  marked unedited/article in press. Its data-availability text still contains
  an unresolved Figshare hyperlink; analysis code is available on request.
  No final archive location is invented here.
- Native small-system references implement Bell/Choi identities, not the
  SPAM-robust depth-decay experiment or its auxiliary calibration acquisition.
  Hardware gate-noise targets include symmetrized eigenvalues and identifiability
  restrictions; a free-standing ideal Pauli channel is a simpler contract.

## Why older successes are not the default shortlist

The original alloy/channel replays remain checks of their stated original
comparators, while stronger local classical controls beat those tested
configurations on those data and budgets. Entity matching's recorded gain belongs
to a combined routing pipeline; its tested standalone quantum head loses.
The multiclass product kernel has a cheap exact classical counterpart. These
do not justify an agent's default quantum recommendation. These observations
do not establish method-wide failure on other data or with modified training;
see [the bounded outcome ledger](CONTEXTUAL_RESULTS.md). Exact classical kernel
equivalence is a structural fact about the unchanged kernel, rather than a
limited-training observation. Original files and non-winning results are preserved.

SAT-6, ADMET and the newer clinical-stacking paper remain endpoint-specific
replication candidates. Small reported gains or wins against one baseline do
not establish a transferable rule such as "QML works on medical data".
