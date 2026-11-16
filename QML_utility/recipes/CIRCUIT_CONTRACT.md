# Portable numeric reference: exact circuit and resource contract

This is an engineering architecture, not a replica of a successful application
paper. Relevant peer-reviewed foundations are [re-uploading](https://doi.org/10.22331/q-2020-02-06-226),
[quantum kernels](https://doi.org/10.1038/s41586-019-0980-2),
[projected kernels](https://doi.org/10.1038/s41467-021-22539-9) and
[Pauli-rotation derivatives](https://doi.org/10.1103/PhysRevA.99.032331).
Their results do not certify this composite model's trainability or superiority.

Fit feature medians and standardization on training rows only. An entirely
missing training feature is retained as a constant. Bound each standardized
coordinate z with `a = pi * tanh(z / 3)`. This preserves coordinate coverage but
is a lossy nonlinear map in finite precision: extreme values saturate. Classical
controls retain full standardized values, and the Fourier control receives the
same bounded angles. Do not describe the transform as information preserving.

For each layer and every block of up to n consecutive input coordinates:

1. On logical qubit q, apply `RY(scale*a_j)` followed by `RZ(scale*a_j)` for
   each coordinate in that block. Every input coordinate is uploaded once per
   layer, even when feature count exceeds qubit count.
2. If enabled, apply the directed CNOT chain `0->1, 1->2, ..., n-2->n-1`.
3. After all feature blocks in that layer, apply one trainable RY and one
   trainable RZ per qubit. Each theta coordinate appears in exactly one gate.

All rotations use `exp(-i*theta*P/2)`. Parameters have shape `(layers,n,2)` and
logical wire 0 is most significant. Qiskit maps it to physical wire n-1;
PennyLane/Cirq use explicit logical order. Fixed feature maps use a seeded
normal angle initialization with std 0.3. Trainable maps start at std 0.01;
these are engineering choices, not identity-block or general plateau guarantees.

`portable_contract.local_features` returns Z on every qubit, then X, then Y.
Three global product-basis settings suffice to acquire these local expectations
per input in a sampled implementation. All current values are exact expectations
computed from native simulated statevectors, with zero physical shots. No
measurement-noise or QPU-performance result follows from them.

State preparation has `2*d*layers` data rotations, `2*n*layers` trainable rotations
and, when enabled, `(n-1)*ceil(d/n)*layers` entanglers. One complex128 statevector
requires `16*2**n` bytes before SDK workspaces. A cached training state table costs
`16*N*2**n` bytes; a float64 Gram costs `8*N*N` bytes before solver workspaces.
Reported byte counts are component estimates rather than peak process memory.

For trainable features, with P=2*n*layers, one full-batch gradient evaluation
uses `N*(1+2*P)` state preparations. Final diagnostics and validation/test calls
are additional. Circuit-gradient counters do not hide the classical head cost.
Fidelity kernels here are inner products of cached exact statevectors, not
physical kernel-estimation circuits; physical overlap acquisition must count
pair queries, two preparations per overlap protocol and precision costs separately.
