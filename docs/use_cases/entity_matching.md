# Entity matching with a hybrid quantum head and TF-IDF triage

**Published comparison: reported, not reproduced. Local model: runnable reduced adaptation.**

[Bischof et al., Scientific Reports 15, 4318 (2025)](https://doi.org/10.1038/s41598-025-88177-z)
report **96.5% accuracy for HQNN plus TF-IDF versus 94.0% for TF-IDF** on a
603-row extended sentence-pair test. F1 is 0.940 versus 0.906. The winning
pipeline uses six qubits and noiseless simulation. Standalone HQNN reaches
88.7% on this test; its Osaka hardware evaluation is lower. The original test
favored TF-IDF, and the extension deliberately adds ambiguous cases. These
conditions belong beside the positive result. See [Table 4 and the combination
section](https://www.nature.com/articles/s41598-025-88177-z.pdf).

The paper's data are available from its corresponding author on reasonable
request. We have not acquired them, contacted the authors or recovered an
author checkpoint. No verified author-code URL is supplied.

## When to try it

Use this architecture when a cheap similarity rule solves most record pairs
but leaves a well-defined ambiguous region. Product catalogs, duplicate
records and short descriptions are suitable project hypotheses. Measure the
ambiguous region on training/validation data; do not construct a favorable
test distribution after seeing errors.

The useful design idea is a routing policy with a stronger head for difficult
pairs. The experiment must distinguish the benefit of routing, representation
and classifier from any contribution of quantum computation.

## Runnable model and exact implementation boundary

`TfidfQuantumMatcher` fits a TF-IDF vocabulary and computes three pair
descriptors: cosine similarity, token-count difference and token Jaccard score.
A trainable linear map with a bounded `tanh` output supplies rotation angles.
The head alternates RX encoding, ring ZZ interactions and RY fields, then uses
an RZ–RY–RZ/CNOT layer and the first-qubit probability as the match score.
The encoder and quantum parameters are optimized jointly with SPSA.

The TF-IDF threshold and uncertainty interval are learned from training pairs.
Only pairs inside the interval call the quantum head during inference. These
choices define our adaptation. The descriptor encoder replaces the paper's
token embedding; the circuit depth, objective, optimizer and data also differ.
The authored template fixture is a software integration example, not the
paper's hand-crafted dataset or evidence of its gain.

```bash
python -m qml_tricks.learning.case_models --case entity_matching --framework qiskit --output results/local/entity.json
```

Replace `qiskit` with `pennylane` or `cirq`. See [model APIs and saved runs](models.md).

## Agent benchmark contract

1. Group by underlying entity or product family before making train, validation
   and test splits. Keep duplicate descriptions and all pairs sharing a held-out
   entity together. Fit vocabulary, preprocessing and embeddings on training
   entities only.
2. Preserve the source test distribution. Report an ambiguous-case stress test
   as a separate endpoint, with its construction frozen before evaluation.
3. Fit the routing interval using training or validation errors. Benchmark
   TF-IDF alone and the identical route with a classical neural head on the same
   representation. Include tuned string similarity, regularized logistic
   regression and a pretrained semantic matcher when relevant.
4. Give both heads the same representation, labels and hyperparameter budget.
   Include encoder parameters in model size; a small quantum layer does not
   imply a small total model or a space advantage.
5. Save pair IDs, scores, route decisions and predictions. Report precision,
   recall, F1, PR-AUC, calibration and accuracy, including performance separately
   inside and outside the routing interval. Count false merges separately from
   missed matches if their operational costs differ.

The local runner retains TF-IDF and regularized logistic controls. It does not
yet implement a parameter-matched classical neural routing head or a transformer
comparator, so its small fixture cannot isolate a quantum benefit.

## Trainability and costs

Start with a small head, inspect training loss and gradient/shot diagnostics,
and compare a [classical-pretraining workflow](../learning/classical_pretraining.md)
when a useful encoder exists. [SPSA](../learning/spsa.md) reduces objective calls
per update but does not guarantee informative gradients or solve a plateau.
Jointly trained encoding angles occur repeatedly; a naive parameter-shift
formula applied to the whole shared coordinate is generally invalid.

Record vocabulary and descriptor storage, encoder fitting, all circuit calls,
shots, routing fraction, latency, total parameters and peak host memory. The
reference uses exact statevectors and no QPU calls. A physical port needs data
loading and repeated measurements; a low routing fraction reduces inference
queries but does not remove quantum training cost.

> Agent task: implement and freeze an entity-group benchmark. Compare TF-IDF,
> classical and quantum heads under the same route and representation. Preserve
> the ordinary and ambiguous-case tests separately. Save all seeds, pair-level
> predictions and complete costs. Identify every departure from Bischof et al.
