# Alloy-phase prediction: Pei et al., Matter (2024)

Primary paper: [Designing complex concentrated alloys with quantum machine
learning and language modeling](https://doi.org/10.1016/j.matt.2024.05.035).
The authors report a best QSVM accuracy of 89.4% for alloy-phase classification.
Their [public repository](https://github.com/peizong/quantum-machine-learning-metallurgy)
provides 1,252 rows, concentration-averaged descriptors, code and result archives.
This is a relevant materials-screening application using classical inputs.

## What is reproduced

The source for this task is `simulator_data_process/scripts/run_1.2_split.py`,
pinned at the commit in `reproductions/sources.json`. Its inputs have 15
descriptors and a binary phase label. Do not substitute the repository root's
different `training-data.csv`, which also contains a text-mining score, or its
exploratory notebook, which contains different configurations.

The local experiment has three distinct parts:

1. Recompute all 300-split means from the authors' archived eight- and ten-qubit
   scores. This verifies the archived statistic, not the circuit computation.
2. Independently run the prescribed circular ZZ feature map with one repetition
   and default SVC on ten fresh evaluations of the original seeds 22–31, with
   four classical SVC kernels on the same representation. Compare each fresh
   score with its corresponding archived score; report any numerical mismatch.
3. Run ten new stratified 60/20/20 splits with train-only preprocessing. Select
   qubit count and SVC regularization using validation only, then refit and test.
   Compare with tuned full-descriptor L2-logistic regression, RBF-SVC and
   histogram boosting. Preserve the earlier replay scores alongside this test.

## Mathematical and implementation contract

Each encoding layer uses Hadamards, phases `P(2*x_j)`, and a circular sequence
of `CX-P(2*(pi-x_j)*(pi-x_k))-CX` interactions. The downstream fidelity kernel
is `abs(<psi(x)|psi(z)>)**2`, followed by a regularized precomputed-kernel SVC.
Independent [Qiskit](../source/replications/qiskit.py),
[PennyLane](../source/replications/pennylane.py) and
[Cirq](../source/replications/cirq.py) constructions are checked against the
Qiskit library convention. There are no trainable quantum angles in this replay.

Reconstruct preprocessing carefully. The pinned source divides by global
feature maxima, fits PCA and StandardScaler on training rows, and fits final
MinMax bounds on training **and held-out** inputs. Only the author-workflow
replay retains that last access. The controlled extension fits every transform
on its fit partition. Never present the replay as leakage-free evaluation.

PCA orientation also matters for this feature map. The port fixes full SVD
and the legacy U-based sign convention rather than relying on a modern
scikit-learn default. Kernel geometry can change when component signs change.
The exact package versions and score differences are recorded; numerical
agreement is assessed rather than assumed.

## Use cases and limits

Try the model for small materials-screening tables when a fidelity kernel may
fit the descriptor geometry. Include regularization and an external validation
set before making costly fabrication decisions. Check descriptor duplicates
and consider grouped composition/family splits for new applications; random
row splits do not establish transfer to an unseen alloy family.

An `n`-qubit complex128 statevector needs `16*2**n` bytes, before host/SDK
overhead. The stored Gram matrix scales quadratically with sample count; exact
statevector overlap is a simulator implementation. No finite-shot hardware
kernel, fabrication campaign or runtime advantage is reproduced here. The
[results](results.md) distinguish success on the original comparison from the
stronger-baseline extension.
