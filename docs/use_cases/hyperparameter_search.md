# Quantum-assisted hyperparameter search

**Evidence: peer-reviewed reported result; not locally reproduced.**
This repository provides a workflow plan and links to author code. No annealer,
distributed training job or independent HPO reproduction was run here.

## What motivates trying it?

[Wulff et al., Quantum Machine Intelligence (2024)](https://link.springer.com/article/10.1007/s42484-024-00198-5)
use quantum-trained support vector regression (QT-SVR) to predict learning
curves inside Swift-Hyperband. For their CIFAR-10 CNN search, Section 4 and
Figure 4(a) report averages over three seeds:

| Search method | Target-model training epochs |
|---|---:|
| Hyperband | 3,237 |
| Swift-Hyperband with classical SVR | 3,014 |
| Swift-Hyperband with QT-SVR | **2,960** |

This is **1.79% fewer epochs than classical SVR prediction**, or 9.4% fewer
than plain Hyperband, at similar target-model performance. It is not a measured
end-to-end speedup. Quantum predictions aggregate multiple annealed solutions;
classical prediction uses one solution. Other tasks do not uniformly favor
QT-SVR. [Author code](https://github.com/JP-Amboage/qtml-hybrid-workflow) is public;
we inspected availability without executing its pipeline.

## When to use this lead

Consider a pilot when neural-network training dominates your budget, you can
retain learning curves from completed trials, and useful early predictions
would avoid many expensive epochs. The target network stays classical. The
quantum component solves a small performance-predictor training problem.

First test classical early stopping and curve prediction on historical trials.
If their savings are small, annealer overhead can erase any improvement. A
cheaper validation error estimate is not enough: the selected final model must
also satisfy your prespecified quality target.

## Agent implementation contract

1. **Log a replayable search.** Fix dataset partitions, hyperparameter search
   space, seed schedule, target-model initialization, maximum epochs and
   promotion rules. Save partial learning curves and hardware-time logs.
   Keep the final test set outside all scheduler and predictor decisions.
2. **Make a predictor dataset.** Use only information available at each stopping
   point: current hyperparameters and the observed curve prefix. Predict a
   specified later validation metric. Split by whole trials, not by rows from
   the same curve. Fit preprocessing using training trials only.
3. **Prevent time leakage.** During online evaluation, train the predictor only
   on labels already observed from earlier completed trials. A historical replay
   must enforce the same information boundary. Count the cost of collecting
   those completed trials, including any pilot used only by one method.
4. **Implement classical controls first.** Include plain Hyperband, its classical
   SVR-prediction variant, and an equally sized classical predictor ensemble.
   A random forest or gradient-boosted curve predictor is a useful additional
   project control. Hold the target-model search and scheduler constant.
5. **Add the annealer predictor.** Read the author's QSVR formulation, bit
   precision, penalty terms, embedding and solution-aggregation rule before
   porting it. Compare tiny instances against a classical exact solution.
   Preserve all sampled solutions and the deterministic aggregation rule.
6. **Measure decisions and final outcomes.** Save stopped/promoted trial IDs,
   rankings, predictor errors, every selected configuration and final model
   scores. Analyze premature rejection of strong late-improving trials.
   Repeat full searches with the planned seeds, rather than repeatedly scoring
   one favored search history.

## Framework and cost accounting

The upstream workflow uses quantum annealing, with D-Wave-specific solver
access. Existing Qiskit, PennyLane and Cirq gate-model recipes do not reproduce
QT-SVR merely by replacing its optimizer. Run a classical replay first and
make any hardware submission an explicit project configuration.

Count target-model epochs and their durations, predictor-data collection,
QUBO construction, bit precision, embedding, logical and physical variables,
anneal reads, programming/readout/queue time, aggregation, network calls,
distributed-worker idle time and monetary cost. Log peak host/GPU memory and
the predictor matrix dimensions. Annealer training and classical inference
have separate budgets; the comparison must include both.

Report these endpoints separately:

- **Search quality:** best validation metric and final untouched-test metric.
- **Target-model work:** total training epochs and GPU-hours, including pilots.
- **Operational cost:** elapsed wall time and total classical/QPU spending.

For an epoch saving, use
\(100(B-Q)/B\), naming baseline \(B\) explicitly. Reducing 3,014 to 2,960
epochs is a different comparison from reducing 3,237 to 2,960. A lower epoch
count with worse final quality is a tradeoff, not automatic superiority.

## Completion criteria for an agent

Deliver replayable trial histories, scheduler decisions, a fixed protocol,
predictions on held-out trials, full-search outcomes and resource logs. Compare
the quantum predictor with the classical ensemble using the same number of
aggregated predictions. Confirm that any resource saving survives the complete
cost boundary and the frozen final-quality threshold.

Suggested task to give an agent:

> Implement a historical learning-curve replay for Hyperband and classical
> Swift-Hyperband. Enforce time-ordered predictor training and count the pilot
> trials. Add a quantum-annealer QT-SVR predictor only after exact small-instance
> checks. Compare an equally sized classical predictor ensemble, then report
> final quality, total epochs, GPU-hours, wall time and monetary cost for every
> planned search seed. Label this as an adaptation until the paper protocol and
> source artifacts have been recovered.
