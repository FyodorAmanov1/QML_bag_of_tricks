# One-step volatility: a horizon-specific reservoir result

[Li et al., Physical Review Research 8, 023028 (2026)](https://journals.aps.org/prresearch/pdf/10.1103/rbj7-4wnq)
Table II reports one-step QR2 MSE **0.103**, versus **0.1089** for the best listed
classical RCX. The five-step comparison reverses: classical RC **0.1528**, versus
QR1 **0.1556** and QR2 **0.1663**. Several classical methods remain in the model
confidence set; membership does not establish unique superiority.

The task is historical monthly realized-volatility forecasting with rolling
re-estimation, an approximately 571-month initial window and causal predictors.
It is not a random-split tabular task or a claim about trading profits.

## Agent implementation

Use [the protocol](../protocols/volatility_reservoir.json). Recover the target
transform, forecast calendar, covariate vintages, Ising couplings, input/memory
partition, readout regularization and feature-selection scope. Keep one-step
open-loop forecasting separate from recursive horizons.

Use nested chronological validation and match temporal information for HAR/ARMAX,
regularized lag models, LSTM and classical reservoirs. Treat any new market period
as a new experiment. Save date-indexed predictions, all origins, failed fits and
paired temporal uncertainty. Seeds do not create independent financial histories.
Count reset, evolution, measurement and refitting costs. Original source has
partial notebooks and archived predictions, not a verified complete replay here.

## Temporal reservoir recipe

The paper uses ten qubits split into input and memory, with three lag-encoding
steps. At a step, prepare input qubits from the current lag features, combine
them with the retained memory density matrix, evolve under a fixed fully
connected Ising Hamiltonian, and trace out the input subsystem for the next step.
After the final step, Pauli-Z expectations form a classical ridge-readout vector.
QR2 concatenates two reservoir outputs with different final evolution durations;
it therefore uses additional acquisition resources. Recover the actual partition,
couplings, durations, target scaling and lag order from the original artifacts.

The paper evaluates 100 reservoir instances and reports its best-performing
instance. It states time-series-aware cross-validation, but the exact selection
scope must be independently checked, including feature selection and LSTM/RC
tuning. Preserve all 100 results in a replay and use an inner chronological
validation procedure in a new-project extension. Exact density-matrix expectation
values in simulation do not establish a finite-shot hardware cost or scaling win.
