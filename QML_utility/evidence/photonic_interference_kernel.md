# Photonic kernels: successful geometry on engineered targets

[Yin et al., Nature Photonics 19, 1020–1027 (2025)](https://www.nature.com/articles/s41566-025-01682-5)
demonstrates classification with two photons in six modes. Figure 4 compares
interfering/distinguishable-photon kernels and standard kernels on 40–100-point
tasks, with two-thirds training and five generated tasks for averaged curves.

**Essential condition:** labels are constructed using the geometric difference
between the two Gram matrices. They are not independent real-world labels.
This is useful evidence that kernel inductive bias and interference can matter
for a particular target geometry. It is not a practical speedup demonstration:
this small optical system is classically simulable.

## Agent implementation

Follow [the protocol](../protocols/photonic_interference_kernel.json) and the
pinned [author artifacts](https://github.com/dapingQ/PhoQuKs). Keep geometric-label
generation separate from model fitting and disclose its full-task access.
Recover label discretization, C selection, count normalization, all five tasks
and selected-instance versus averaged plotting conventions.

Include distinguishable photons, exact classical low-photon simulation, RBF,
polynomial, linear and neural tangent controls. A transfer experiment must freeze
external labels independently and tune inside training folds. Retain both tasks
where the geometry helps and tasks where it does not. Record optical shots,
loss/postselection and quadratic Gram storage. No plot-digitized scores or
local hardware replay is claimed here.

## Kernel and task recipe

For each phase vector x construct the paper's interferometer U(x). For a fixed
input Fock pattern s, estimate the probability of returning to s under
`U(x_j).dagger() @ U(x_i)`; use the interfering-photon transition rule for KQ and
the distinguishable-photon rule for KC. The two probability rules must share
the phase vectors, layouts and input pattern. Build both Gram matrices, then
reconstruct the paper's geometric-difference label generator. It uses the whole
designed task, including candidate test inputs; disclose that access explicitly.

Only after fixing that task should the SVM train on its declared two-thirds
partition. For an ordinary application, supply labels independently and use
validation-only model selection as a separately labeled transfer experiment.
An SVM win over KC isolates an interference-dependent representation in this
construction; it does not beat a classical simulator of KQ at this photon count.
