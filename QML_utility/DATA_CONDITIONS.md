# Match the problem before implementing the model

| Question | Consequence |
|---|---|
| Are inputs ordinary arrays, quantum state copies, or queries to a physical channel? | An acquisition advantage cannot be transferred to a CSV after acquisition. |
| Is the goal fixed-task prediction, continued adaptation, retention, or system identification? | A plasticity result does not show better IID accuracy or less forgetting. |
| Is useful information lost in preprocessing? | Preserve full task-relevant input before adding a quantum bottleneck. |
| Does a graph dataset plausibly share an exchangeable graphon structure? | QGNR is motivated by this reconstruction problem, not arbitrary tabular similarity. |
| Are these small-subsample molecular descriptors and a continuous activity endpoint? | Fixed-reservoir robustness motivates a learning curve with same-input controls, not a generic small-data recommendation. |
| Are text sequences short and the comparator a simple attention model? | QSANN evidence does not transfer directly to long-context or pretrained-language-model comparisons. |
| Were targets engineered from the candidate kernels? | Preserve that condition; external-label transfer is a separate experiment. |
| Is the forecast one-step, chronologically split and refit using past information? | The volatility result reverses at five steps; audit selection and data vintages. |
| Does a query arrive before or after acquisition? | Known-early observables may make joint two-copy acquisition unnecessary. |
| Are sign, magnitude, or the entire distribution required? | Bell-square estimation discards the sign. |
| Are channel queries and retained reference qubits available? | Signed Choi estimation differs from two-copy state-square estimation; include reference calibration and costs. |
| Can the specified loading/readout be implemented at the budget? | A log-sized register does not imply cheap amplitude loading or full readout. |
| Does the comparator have the same useful information? | Match temporal context, preprocessing, training data and tuning access. |

These are falsifiable project questions, not a classifier that diagnoses quantum
advantage from dataset metadata. Validate motivation on training/validation data;
freeze choices before inspecting test outcomes. For small datasets report
learning curves, uncertainty and split sensitivity instead of selecting one win.

Practical data requirements and *proven sufficient conditions* are different.
The graphon and continual-learning papers motivate data regimes empirically.
Their presence in a new project does not establish that quantum processing is
necessary. Classical Fourier, orthogonal/unitary, replay and reset methods remain
important adversaries.

A no-gain finding on one configuration does not close the method for other data
or training. Diagnose representation loss, optimization and information mismatch
before choosing a prespecified follow-up. Retain the original outcome alongside
the new experiment, including if the follow-up also finds no gain.
