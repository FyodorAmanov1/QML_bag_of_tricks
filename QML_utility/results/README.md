# Executed local protocol audits

These are restricted analytic simulations, not published hardware results or
paper-scale classical-data reproductions. Each setting uses five seeds,
2,000 balanced test states per seed and 512 copies per state. Both inference
rules know the same signal magnitude and priors. The local protocol uses
uniform random pre-query single-qubit Pauli measurements. The final control
learns the query before measurement. Scores are balanced-accuracy means and
across-seed sample SDs, not significance estimates or proof of a general gap.

## Ideal acquisition

| Qubits per copy | Two-copy Bell | Optimal specified local records | Query known early |
|---|---:|---:|---:|
| 2 | 100.00% ± 0.00 | 99.95% ± 0.05 | 100.00% ± 0.00 |
| 4 | 100.00% ± 0.00 | 82.39% ± 1.18 | 100.00% ± 0.00 |
| 8 | 100.00% ± 0.00 | 50.06% ± 0.10 | 100.00% ± 0.00 |
| 12 | 100.00% ± 0.00 | 50.00% ± 0.00 | 100.00% ± 0.00 |

[Full per-state sufficient statistics and predictions](late_query_reference.json.gz).

## Independent 3% measured-bit errors

| Qubits per copy | Two-copy Bell | Optimal specified local records | Query known early |
|---|---:|---:|---:|
| 2 | 100.00% ± 0.00 | 99.84% ± 0.08 | 100.00% ± 0.00 |
| 4 | 99.98% ± 0.04 | 71.75% ± 1.05 | 100.00% ± 0.00 |
| 8 | 99.63% ± 0.12 | 50.04% ± 0.04 | 100.00% ± 0.00 |
| 12 | 96.94% ± 0.64 | 50.00% ± 0.00 | 100.00% ± 0.00 |

[Full per-state sufficient statistics and predictions](late_query_readout_noise.json.gz).

## Reproduce and inspect

```bash
python -m QML_utility.benchmarks.late_query --output QML_utility/results/late_query_reference.json.gz
python -m QML_utility.benchmarks.late_query --readout-error 0.03 --output QML_utility/results/late_query_readout_noise.json.gz
python -m QML_utility.report_results
```

The report audit recomputes decisions using binomial log-PMFs independently
of the sampling script’s closed-form likelihood ratios. It verifies counts,
source hash, copy accounting and all saved predictions. At two qubits the
random local protocol already works. At larger weight it rarely matches the
late query; knowing that query early closes the gap. Bell uses 2n coherent
qubits, and this sampler measures no physical loading, gate noise or wall-time
advantage. Readout errors are a declared model, not calibrated hardware noise.
