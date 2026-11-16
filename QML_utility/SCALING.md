# Scale acquisition and fitting separately

The original reference caps total CSV rows at 512 because some candidates use
dense kernels or full-batch quantum gradients. The `batched_features` adapter
accepts an explicit limit up to 20,000 records while keeping the existing native
circuit unchanged. This is a linear-memory engineering route, not an asymptotic
quantum-advantage result or a claim that 20,000 observations make QML preferable.

```json
{
  "adapter":"batched_features",
  "methods":["quantum_features"],
  "quantum":{"qubits":2,"layers":1,"batch_size":128,"head_epochs":10},
  "budget":{"max_rows":20000,"max_csv_bytes":33554432,"max_feature_bytes":67108864,"state_preparations_per_trial":100000,"seconds_per_operation":300}
}
```

Add these fields to the complete numeric project contract. The adapter compares
fixed quantum features with dummy, regularized linear, random-Fourier and
histogram-boosting controls. Three prespecified candidates per non-dummy family
give 13 trials in total. The classical random-feature control is a useful check
against inexpensive nonlinear embeddings; its map has 128 random coordinates.
Broaden the classical search and add domain-specific controls before making a
performance recommendation.

## What changes

1. Fit medians/scaling on the training partition only, then upload every input
   coordinate through the [unchanged native circuit](recipes/CIRCUIT_CONTRACT.md).
2. Prepare at most `batch_size` statevectors at a time. Extract local XYZ
   expectations and release the state batch.
3. Cache only the N-by-3n measured-feature table. Fit a train-only feature scaler,
   then make multiple SGD passes over that cache without repeating quantum
   acquisition. Classification uses a logistic SGD head; regression standardizes
   training targets and uses an SGD regressor before inverting target scaling.
4. At inference, acquire features for new records and apply the frozen scaler/head.

The implementation uses [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
and [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
with `partial_fit`. Its head differs from the reference's exact logistic/ridge
head; do not label the two adapters mathematically identical.

## Cost and limits

For N records, n qubits and batch size B, the local-feature cache is `8*N*3*n`
bytes and a complex128 state batch is approximately `16*B*2**n` bytes before
SDK workspaces. Raw inputs, standardized arrays, the scaled feature copy and
classical solver workspaces remain additional linear storage. These component
estimates are not peak-process memory measurements. No N-by-N Gram is constructed
by the quantum adapter. Classical controls have their own reported algorithms
and memory costs.

Feature acquisition is one pass per fit; head epochs do not multiply native
preparations. Validation and test still consume preparations. Physical local
measurements would need three product-basis settings and a precision-dependent
shot budget per input; exact statevector expectations here use zero physical
shots. The adapter is a simulator implementation, not an out-of-core loader or
an online quantum reservoir. CSV bytes are bounded before loading and row
parsing stops after the declared cap is exceeded. It retains 1–64 numeric input
features, 1–6 qubits and 1–4 layers; larger representations require explicit design.

For a useful teacher whose production acquisition is too expensive, consider
[offline feature distillation](OFFLINE_DISTILLATION.md). For a genuine kernel
task consider the existing [Nyström companion](../docs/learning/nystrom.md),
which still needs quantum queries on new observations. Fidelity-kernel and
trainable-circuit candidates remain on the reference path rather than being
silently approximated by this adapter.
