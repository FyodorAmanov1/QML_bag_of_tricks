# Late-query quantum-property learning

**Primary result:** Huang et al., Science 376, 1182–1186 (2022),
[paper](https://preskill.caltech.edu/pubs/preskill-2022-quantum-advantage.pdf),
[author code](https://github.com/quantumlib/ReCirq/tree/master/recirq/qml_lfe).
The publication establishes learning-access separations and a superconducting
experiment. This folder does not reproduce that experiment.

## Precisely what the local audit tests

For independently supplied copies of

\[
\rho_{P,a}=2^{-n}(I+aP),
\]

compare `a=0` with `a=+m` or `-m`. A full-weight Pauli query `P` arrives after
acquisition. Bell acquisition on two copies estimates
`Tr(P rho)^2=a^2`. A random local Pauli record matches the query with probability
`3**(-n)`. Better regularization cannot reconstruct information missing from
that declared measurement record.

`benchmarks/late_query.py` samples exact sufficient statistics for this family;
it avoids dense density matrices. Both inference rules know the same magnitude
and priors. Bell rounds consume two copies. An early-query direct measurement
is included and removes the gap. This is not a bound against every possible
single-copy acquisition strategy, and not a learning benchmark on classical arrays.

The three SDK references verify Bell probability ordering and the YY sign on
small explicit density matrices. Run the audit across sizes, seeds and readout
noise, retaining per-state counts, labels and predictions.

**Use when:** independent quantum copies and two-copy coherence exist, magnitude
is the target, and the query arrives late. **Reject when:** input is already a
classical CSV, signed expectation is required, or the query is known early.
