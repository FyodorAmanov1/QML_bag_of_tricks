# Molecular carcinogenicity screening with QBoost

**Evidence: peer-reviewed reported result; not locally reproduced.**
This is a quantum-annealing application lead and an agent implementation plan.
No molecular dataset, annealing run or QBoost reproduction is supplied here.

## What motivates trying it?

[Salloum et al., IEEE Access (2025)](https://doi.org/10.1109/ACCESS.2025.3531391)
report **0.8614 ROC-AUC for QBoost versus 0.8258 for LightAutoML** on
Carcinogens_Lagunin. **This positive endpoint is one of ten: LightAutoML leads
on the other nine.** Table 2 lists 196 training examples, with 42 positives.
The experiments use a D-Wave Advantage 4.1 annealer. See Tables 1–2 and Section
VII in the [primary article PDF](https://inspirehep.net/files/f7e8c2722a6e150153c21830a0fae9fe).

The paper's availability statement names code and data links, but their targets
and exact split artifacts have not been resolved here. The point estimates
alone do not establish significance or performance on new molecular scaffolds.

## When to use this lead

Consider an experiment when you have a small binary molecular endpoint, a
reproducible descriptor pipeline and a useful pool of inexpensive weak learners.
The hypothesis is that regularized binary selection of those learners might
improve the ensemble. Testing the solver is a second question: an ensemble
benefit can exist even when a classical solver finds equally good weights.

Use this as a research screening benchmark. A project adaptation needs new
evidence for its own chemical families and endpoint. Set a separate decision
threshold and calibration requirement if downstream users consume probabilities.

## Agent implementation contract

1. **Recover the scientific inputs.** Identify the original molecule records,
   labels, descriptor settings and split IDs before claiming a replication.
   Check licenses, duplicates, salts, conflicting labels and near-duplicate
   structures. Freeze a manifest of canonical IDs and processing decisions.
2. **Set an honest generalization task.** Preserve the original split for an
   author-protocol replay. Add a separately labeled scaffold or temporal holdout
   for a project adaptation. Fit feature selection and imputation using training
   folds only. Do not move positive examples into test to improve a headline.
3. **Build a fixed weak-learner pool.** Specify learners, descriptor access,
   fitting folds and prediction scaling before solving for weights. Use
   out-of-fold training predictions to avoid selecting weights on artificially
   perfect in-sample scores. Refit the chosen base learners on the full training
   partition before evaluating the untouched test partition.
4. **Compare the same optimization problem.** Solve the identical QUBO using a
   classical exact solver on small fixtures and a classical heuristic on the
   intended size. Then evaluate the annealer with fixed reads and time budget.
   Report the objective value and downstream predictions separately.
5. **Include strong endpoint controls.** Benchmark regularized logistic
   regression, RBF-SVM, random forests, gradient boosting and LightAutoML when
   available. Match data access and publish actual tuning budgets. Choose
   thresholds and calibration from validation data only.

## A precise QUBO preflight for an adaptation

This is a generic teaching formulation for validating an implementation. Match
the paper's equations and normalizations before treating it as a replication.
Let \(H_{it}\in[-1,1]\) be out-of-fold weak predictions for \(m\) training
examples and \(T\) learners, and \(y_i\in\{-1,+1\}\). With binary selections
\(w\in\{0,1\}^T\), one possible regularized loss is

\[
E(w)=\frac{1}{m}\left\|\frac{H}{T}w-y\right\|_2^2
     +\lambda\sum_t w_t.
\]

Put \(A=H/T\). Up to the constant \(y^Ty/m\), the symmetric-matrix convention
is \(E(w)=w^TQw+\mathrm{constant}\), with

\[
Q=\frac{A^TA}{m}
 +\operatorname{diag}\left(-\frac{2A^Ty}{m}+\lambda\mathbf{1}\right).
\]

The diagonal incorporates linear terms because \(w_t^2=w_t\). If a sampler
uses an upper-triangular coefficient dictionary, use \(Q_{tt}\) on its diagonal
and \(2Q_{st}\) for \(s<t\). Exhaustively enumerate a tiny fixture to check
that every bit string has the expected energy before submitting a job.
Select regularization and bit precision with validation data; an annealer's
lowest energy does not automatically imply best held-out ROC-AUC.

## Framework and resource requirements

Quantum annealing uses an annealer interface such as D-Wave Ocean. Running a
classical simulated-annealing solver is a useful control and must be labeled
classical. A Qiskit, PennyLane or Cirq QAOA port would be a different solver
experiment requiring its own evidence. The repository's
[fidelity-kernel guide](../learning/fidelity_kernel.md) is another gate-model
molecular hypothesis, not an implementation of this QBoost study.

Record fingerprint generation, weak-learner fitting and \(O(mT+T^2)\) dense
prediction/Gram storage. Keep logical-variable count distinct from physical
qubits after embedding. Save solver ID, embedding, chain strength, chain-break
rate, reads, anneal schedule, rejected runs, programming/readout time, queue
time and monetary cost. Include all classical and hardware tuning costs.

## Completion criteria for an agent

Deliver per-molecule predictions for every planned split, ROC-AUC and PR-AUC,
calibration, sensitivity at a prespecified specificity, uncertainty and total
resources. Compare identical weak-learner pools under different solvers. Save
all ten endpoints if reproducing the study; report a new endpoint collection
in full if adapting it. A single favorable endpoint motivates exploration
without establishing an ADMET-wide advantage.

Suggested task to give an agent:

> Recover the source artifacts for the QBoost carcinogenicity result. Stop a
> replication claim if exact artifacts cannot be recovered; propose a labeled
> adaptation instead. Implement and exhaustively validate a small QUBO, compare
> classical and quantum solvers on identical inputs, and benchmark scaffold-held
> predictions against tuned regularized and ensemble models. Preserve all seeds
> and endpoint outcomes.
