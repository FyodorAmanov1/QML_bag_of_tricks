# Multiclass classification with selected quantum kernels

**Published comparison: reported, not reproduced. Local model: runnable reduced adaptation.**

[Ding et al., Physical Review A 111, 062410 (2025)](https://doi.org/10.1103/PhysRevA.111.062410)
report higher mean five-fold classification accuracy for selected quantum
kernels than the tested linear, polynomial, sigmoid and Gaussian kernels on
Iris, Tae, Penguin, Glass, Ecoli and Vowel. See Figure 7 and Section V.B in the
[published primary PDF](https://dr.ntu.edu.sg/server/api/core/bitstreams/8098261f-9a5a-4190-929e-cfa4930efaaf/content).
Paired graph values are not presented here as exact numbers. Table IV's later
70/30 results are a separate evaluation stage and must not be paired with
cross-validation bars to manufacture a numerical gain.

The paper cites public datasets. Exact split manifests and fitted artifacts
have not been recovered; no author-code URL has been verified here.

## When to try it

Use a selected fidelity-kernel model for modest sample counts and a compact
classical representation. It is a useful way to test task-dependent geometry
while the supervised classifier stays convex. Fix the feature budget before
comparison, and tune regularization for conventional kernels as carefully as
the quantum encoding.

A quantum implementation is not always necessary to obtain the representation.
The product Pauli kernels in particular have an exact efficient classical
formula. That control is part of the runnable example.

## Runnable model

`MulticlassQuantumKernel` constructs independent native states with six feature
maps: full, linear or circular IQP-style interactions, and Pauli X, Y or Z
product maps. It fits a classical one-vs-rest SVM to

\[
K_{ij}=|\langle\psi(x_i)|\psi(x_j)\rangle|^2.
\]

For the IQP-style adaptation, start with Hadamards, apply RZ angles \(a_i\),
then \(R_{ZZ}(a_i a_j)\) on the chosen edges. The local implementation uses
\(a=\pi s\tanh(\operatorname{StandardScaler}(x))\), with the scaler fitted
only on training data. There is one encoding block. The product maps apply
RX or RY to zero states, or RZ to plus states, with the same angles.

For these product maps the SDK convention is
\(R_P(a)=e^{-iaP/2}\), hence

\[
K(a,b)=\prod_j\cos^2((a_j-b_j)/2).
\]

Appendix D uses doubled coordinates in its equivalent expression. Keep actual
rotation angles distinct from those coordinates. Tests verify the formula,
probability ordering, PSD and circuit equivalence across all three SDKs.

```bash
python -m qml_tricks.learning.case_models --case multiclass_kernel --framework pennylane --output results/local/multiclass.json
```

The local experiment selects six quantum configurations and six classical SVC
configurations using a new validation split of Iris. It also runs the exact
classical Pauli control. Bounded preprocessing, regularization, solver and
splits are adaptation choices. These runs do not reproduce QEM-SVM's scores.

## Agent benchmark contract

1. Split first. Fit missing-value treatment, scaling and PCA inside training
   folds. For kernel-family selection, use nested cross-validation or a separate
   validation partition before the untouched test evaluation.
2. Include tuned linear, RBF and polynomial SVMs, a regularized linear model,
   tree ensembles and a domain-specific baseline. Publish all candidate grids
   and selection scores. Keep every failed configuration and seed.
3. Compare the exact classical product formula using identical preprocessing,
   regularization and decision rules. Identical predictions reveal the same
   model; they are not an additional quantum advantage.
4. Check diagonals, symmetry, PSD and finite-shot instability. A finite-shot
   Gram matrix may be indefinite. Any repair must be fit on training data and
   must define a consistent out-of-sample extension.
5. Report accuracy, macro-F1, class recall, calibration where applicable and
   paired uncertainty. A generic generalization upper bound or a selected
   favorable split does not establish out-of-distribution superiority.

## Trainability, loading and resources

This adaptation trains the classical SVM and selects the feature map; it does
not backpropagate through circuit parameters. It avoids variational optimization
without avoiding kernel concentration, sampling noise or representation failure.
See the [fidelity-kernel guide](../learning/fidelity_kernel.md).

For \(N\) training points and \(M\) test points, budget up to
\(N(N-1)/2\) training and \(MN\) test overlap estimates per configuration,
each with its own loading circuits and shots. Dense Gram storage is
\(O(N^2)\). The reference additionally caches \(O(N2^q)\) complex amplitudes
on the host and computes overlaps directly; this is simulator access, not a
hardware shortcut. Record full tuning and classical SVM costs.

> Agent task: benchmark a selected multiclass fidelity kernel with nested
> preprocessing and validation. Include the identical classical Pauli formula,
> tuned conventional kernels and a strong tabular model. Save held-out
> predictions and the complete query, memory and tuning budget.
