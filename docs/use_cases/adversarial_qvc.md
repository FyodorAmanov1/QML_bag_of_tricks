# Amplitude-encoded QVCs for resistance to transferred attacks

**Published comparison: reported, not reproduced. Local model: runnable reduced adaptation.**

[West et al., Physical Review Research 5, 023186 (2023)](https://doi.org/10.1103/PhysRevResearch.5.023186)
report that QVCs resist transferred ConvNet-generated attacks better than the
tested classical networks across their image tasks. Figure 2 compares ConvNet,
ResNet18 and QVCs on 250 attacked images per dataset. The study uses MNIST,
Fashion-MNIST, CIFAR-2 and Celeb-A with deep statevector-simulated classifiers.
**Attacks optimized directly against QVCs remain effective.** This is a
threat-model-dependent transfer benefit, not universal adversarial robustness
or higher clean accuracy. See [Figure 2 and Appendices A–C](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.5.023186).

No exact graph-digitized percentages are asserted. Author-trained parameters,
attack examples and split manifests have not been recovered here.

## When to try it

Consider the model when your project studies transfer between independently
trained classifiers. The hypothesis is that a different representation may
make adversarial inputs from one surrogate less effective on another target.
Measure the intended threat model before using the model in a defense pipeline.

Keep the attack source distinct from both transfer targets. A source classifier
is white-box to its own attack; an independent quantum or classical target has
a different information boundary. Be explicit about that asymmetry.

## Runnable model and boundary

`AmplitudeQVCClassifier` pads a real feature vector to a power of two and loads
its normalized amplitudes. It applies RZ–RY–RZ rotations and a nearest-neighbor
CZ chain per layer. The first \(c\) local Z expectations are class logits,
with softmax and cross-entropy. This readout requires at least \(c\) qubits.
A blank vector maps to the all-zero basis state by a documented convention.

The small reference uses binary scikit-learn digit images pooled to 4×4,
four qubits, one layer and a short SPSA fit. The paper uses full-sized images,
much deeper circuits and Adam training. This adaptation is not a replication
of the paper's accuracy, attack budgets or defense result.

```bash
python -m qml_tricks.learning.case_models --case adversarial_qvc --framework cirq --output results/local/robustness.json
```

The saved experiment includes a tuned regularized logistic attack source, an
independent classical MLP target, and the QVC target. It evaluates clean inputs,
classical-source PGD perturbations and QVC white-box PGD perturbations on all
three models. It stores the perturbed inputs as well as predictions.
Three PGD steps and one restart are integration diagnostics; a serious robustness
assessment requires stronger attacks and convergence studies.

## Agent benchmark contract

1. Freeze training, validation and test images and attack endpoints. Declare
   preprocessing, pixel range, norm, epsilon, step size, steps, restarts and
   allowed access to the target. Report clean accuracy separately.
2. Create a full source-to-target attack matrix, including independent classical
   transfer targets and quantum white-box attacks. Reuse the same perturbed
   images for every target in each source row.
3. Differentiate the complete preprocessing and amplitude normalization map.
   The executable estimates gradients with central differences on the classical
   simulator before normalization. This is costly access and not a free
   hardware gradient oracle.
4. Test adaptive attacks, multiple seeds, restarts and larger iteration counts.
   Compare standard strong attacks and adversarially trained classical controls.
   If a sampled predictor is randomized, include expectation-over-randomness
   attacks. Check for gradient masking and attack implementation errors.
5. Save per-image predictions and perturbations. Report worst-case robust
   accuracy across the declared attack suite and uncertainty on held-out images.
   Restrict any robustness statement to the evaluated threat model and budget.

## Trainability and resource accounting

A compact register does not make a deep image QVC trainable. Inspect gradient
norms and noise before scaling; [local observables](../strategies/local_loss.md)
alone do not transfer a plateau-avoidance theorem to an arbitrary image loss.
The reference's short SPSA training is an engineering choice, not evidence of
convergence or avoidance of barren plateaus.

An input dimension \(d\) needs \(\lceil\log_2 d\rceil\) amplitude qubits, but
arbitrary state preparation can need \(O(d)\) gates. Record loading, circuit
depth, shots, training evaluations and attack queries. One exact complex128
statevector uses \(16\cdot2^q\) bytes; gradient workspaces and batching cost more.
Cirq's initial-state simulator argument is explicitly counted as state loading
access, not a physical state-preparation circuit. Hardware noise, finite-shot
robustness and practical latency remain unevaluated by the local reference.

> Agent task: implement a cross-model attack matrix with a QVC and independent
> classical targets. Include QVC white-box attacks and strong convergence sweeps.
> Differentiate preprocessing correctly, retain perturbed test images, and
> report clean accuracy, worst-case attacked accuracy and all query costs.
