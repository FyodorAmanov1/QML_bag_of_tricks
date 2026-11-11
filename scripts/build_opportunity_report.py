"""Render every application comparison; retain controls that remove a gain."""
import json
import gzip
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
raw_path = ROOT/'results/opportunities.json'
archive = ROOT/'results/opportunities.json.gz'
raw = raw_path.read_bytes() if raw_path.exists() else gzip.decompress(archive.read_bytes())
data = json.loads(raw)
archive.write_bytes(gzip.compress(raw, mtime=0))
compact = {key:data[key] for key in ('protocol', 'selected_applications', 'native_verification',
    'summary', 'paired_comparisons', 'elapsed_seconds', 'platform', 'threadpools', 'versions',
    'source_hashes', 'protocol_sha256', 'limits')}
compact.update(full_report='opportunities.json.gz', full_json_sha256=hashlib.sha256(raw).hexdigest(),
    experiment_configurations=len(data['cases']),
    base_evaluations=sum(m['model']!='classical_selected' for c in data['cases'] for m in c['models']))
(ROOT/'results/opportunities_summary.json').write_text(json.dumps(compact, indent=2)+'\n', encoding='utf-8')
summary = {(tuple(r['configuration']), r['model']):r for r in data['summary']}
paired = {(tuple(r['configuration']), r['control']):r for r in data['paired_comparisons']}


def get(config, model):
    return summary[(tuple(config), model)]


def fmt(config, model):
    r = get(config, model)
    return 'failed' if r['mean'] is None else f"{r['mean']:.4f}" + (f" ({r['failures']} failed)" if r['failures'] else '')


def gain(config, control):
    r = paired.get((tuple(config), control))
    if not r:
        return 'unavailable'
    lo, hi = r['descriptive_paired_bootstrap_95']
    return f"{r['mean_gain']:.4f} [{lo:.4f}, {hi:.4f}]; {r['positive_seeds']}/{r['seed_count']} positive"


base_rows = [m for c in data['cases'] for m in c['models'] if m['model'] != 'classical_selected']
failures = sum('error' in m for m in base_rows)
trial_failures = sum('error' in t for m in base_rows for t in m['trials'])
warning_count = sum(len(t.get('warnings', [])) for m in base_rows for t in m['trials'])
lines = ['# Application benchmark results', '',
    '![Application-specific comparisons](../assets/application_comparisons.png)', '',
    f"The fixed protocol produced **{len(data['cases'])} configurations** over {len(data['protocol']['seeds'])} independent data-generation seeds, **{len(base_rows)} base model/control evaluations**, {failures} failed final evaluations and {trial_failures} failed tuning candidates. Pooled classical-champion rows reuse the validation-selected family result and are not additional independent evaluations. The recorded run took {data['elapsed_seconds']:.1f} seconds on its recorded machine.", '',
    f"All {warning_count} candidate warnings are retained in the JSON report. A convergence warning is not silently counted as a failed or successful convergence proof. Figures and tables include all pre-specified grids, not just winning settings. Curves are seed means; shaded bands in the state-learning panels show plus/minus one seed standard deviation.", '',
    '## Periodic calibration: representation benefit and its classical explanation', '',
    'Lower RMSE is better. The quantum feature bank and cheap classical Fourier bank implement the same functions in the exact limit. The conventional champion is selected using validation scores from Ridge, Elastic Net, RBF-SVR, L2-MLP and boosting. It gets a 45-candidate pooled search; each individual family and the quantum readout get nine.', '',
    '| Training labels | Shots/setting | Quantum Ridge | Selected classical | Fourier Ridge | Ridge | Elastic Net | RBF-SVR | L2-MLP | Boosting |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
r = data['protocol']['ramsey']
for size in r['train_sizes']:
    for shots in r['shots_per_setting']:
        key = ('ramsey_calibration', size, shots)
        if (key, 'quantum_ramsey_ridge') not in summary:
            continue
        models = ['quantum_ramsey_ridge', 'classical_selected', 'fourier_ridge', 'ridge', 'elastic_net', 'rbf_svr', 'mlp_l2', 'boosting']
        lines.append(f"| {size} | {shots or 'exact'} | " + ' | '.join(fmt(key, name) for name in models) + ' |')
lines += ['', 'Positive paired gain below means the quantum pipeline had **lower RMSE**. These are descriptive seed-bootstrap intervals, not corrected significance tests.', '',
    '| Training labels | Mode | Gain over validation-selected classical | Gain over Fourier-Ridge |', '|---|---|---|---|']
for size in r['train_sizes']:
    for shots in r['shots_per_setting']:
        key = ('ramsey_calibration', size, shots)
        if (key, 'quantum_ramsey_ridge') in summary:
            lines.append(f"| {size} | {shots or 'exact'} | {gain(key, 'classical_selected')} | {gain(key, 'fourier_ridge')} |")
lines += ['', 'The signal is a synthetic periodic physical response with known harmonic structure and held-out future periods. Gains over generic models do not establish a quantum-specific advantage: the Fourier control is decisive. Finite-shot quantum deployment additionally consumes state copies at every inference. See the [implementation and use cases](ramsey_calibration.md).', '',
    '## Quantum-state characterization: benefit of two-copy acquisition', '',
    'Higher balanced accuracy is better. Copy counts below are per state and per acquisition protocol. The Bell pipeline has twice as many simultaneous qubits but the **same total state-copy budget**. All learned models have L2/other explicit regularization tuning. The local Bayes control is optimal for the selected random local record and known signal family. The query-known control gets advance information and changes the task.', '',
    '| Input qubits | Copies | Bit error | Bell + L2 logistic | Selected local ML | Local Bayes | Local L2 logistic | Local RBF-SVC | Local L2-MLP | Query-known Bayes |',
    '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
p = data['protocol']['pauli']
for n in p['qubits']:
    for copies in p['state_copies']:
        for noise in p['readout_bit_flip']:
            key = ('late_pauli_query', n, copies, noise)
            if (key, 'bell_logistic_l2') not in summary:
                continue
            models = ['bell_logistic_l2', 'classical_selected', 'local_bayes', 'local_logistic_l2', 'local_rbf_svc', 'local_mlp_l2', 'query_known_bayes']
            lines.append(f'| {n} | {copies} | {noise} | ' + ' | '.join(fmt(key, name) for name in models) + ' |')
lines += ['', 'Positive paired gain here means **higher balanced accuracy** for the Bell pipeline.', '',
    '| Input qubits | Copies | Bit error | Gain over optimal local-record control | Gain over query-known control |', '|---|---|---|---|---|']
for n in p['qubits']:
    for copies in p['state_copies']:
        for noise in p['readout_bit_flip']:
            key = ('late_pauli_query', n, copies, noise)
            if (key, 'bell_logistic_l2') in summary:
                lines.append(f'| {n} | {copies} | {noise} | {gain(key, "local_bayes")} | {gain(key, "query_known_bayes")} |')
lines += ['', 'The useful regime is late queries, limited copies and sufficiently reliable two-copy operations. A gain concerns what the acquisition records retain; it is not a claim that quantum logistic regression optimizes better than classical logistic regression. Changing acquisition to know the query early can remove that incentive. All acquisition here is simulated; large cases use a special-family analytic Born sampler validated against independent native circuits. See the [full access and implementation contract](experimental_learning.md).', '',
    '## Reproduction and limits', '', '```bash',
    'python -m qml_tricks.learning.opportunities --framework qiskit --output results/opportunities.json',
    'python scripts/build_opportunity_report.py', '```', '',
    'Run `python scripts/audit_opportunity_results.py` to independently recompute saved metrics and verify split isolation, validation choices, copy accounting and source content. The audit accepts LF/CRLF equivalents for cross-platform Git checkouts while preserving the original source hashes.', '',
    'Download the [complete gzip-compressed JSON](../source/results/opportunities.json.gz) or the [readable summary JSON](../source/results/opportunities_summary.json). The full report includes every trial, warning, split, test prediction, resource count, record digest, version and source hash. The CLI regenerates the expanded `results/opportunities.json` locally; the report script creates its compressed archive and compact summary for distribution. The installed protocol JSON fixes the grid and seeds. The CLI restricts BLAS/OpenMP pools to one thread, with actual pool information recorded.', '',
    'The ten seeds generate independent synthetic datasets under a fixed simulator model. They do not constitute independent hardware experiments, measured sensor/device datasets, or robustness to unknown physics. Multiple grid comparisons are exploratory; bootstrap intervals are descriptive and uncorrected for multiplicity. The reporting rule requires ten seeds, at least eight positive gains and a positive lower interval endpoint; it is not a theorem or a controlled hypothesis test.', '',
    'No runtime, memory or universal prediction advantage over all classical algorithms is established. The conventional local measurement family is restricted; the publication’s broader lower bounds must not be inferred from these finite tables. Generic regularized models, matched domain controls, optimal record inference and task-changing controls are deliberately shown together.', '',
    'The original [tabular real-data results](../study.md) are preserved. Real semiconductor claims and their missing local dataset are documented in the [application guide](index.md).']
(ROOT/'docs/applications/results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')
colors = ['#4f5cce', '#168f84', '#b36b12', '#696f7b']
sizes = r['train_sizes']
for model, shots, label, color in [('quantum_ramsey_ridge', None, 'Quantum exact', colors[0]),
        ('quantum_ramsey_ridge', 1024, 'Quantum 1024 shots', '#8191ed'),
        ('classical_selected', None, 'Selected classical', colors[1]),
        ('fourier_ridge', None, 'Classical Fourier', colors[2])]:
    if all((('ramsey_calibration', size, shots), model) in summary for size in sizes):
        style = 's--' if model == 'fourier_ridge' else 'o-'
        axes[0].plot(sizes, [get(('ramsey_calibration', size, shots), model)['mean'] for size in sizes], style, label=label, color=color,
                     markerfacecolor='none' if model == 'fourier_ridge' else color)
axes[0].set(xlabel='Training labels', ylabel='Test RMSE (log scale)', title='Periodic calibration: new periods')
axes[0].set_yscale('log'); axes[0].legend(fontsize=8)
for model, label, color in [('bell_logistic_l2', 'Bell + L2 logistic', colors[0]),
        ('classical_selected', 'Selected local ML', colors[1]),
        ('local_bayes', 'Optimal local-record rule', colors[2]),
        ('query_known_bayes', 'Query known early', colors[3])]:
    for ax, configs, x, title in [(axes[1], [('late_pauli_query', n, 512, 0.) for n in p['qubits']], p['qubits'], '512 state copies; ideal readout'),
                                  (axes[2], [('late_pauli_query', 8, b, .01) for b in p['state_copies']], p['state_copies'], '8 input qubits; 1% readout flips')]:
        if all((tuple(k), model) in summary for k in configs):
            values = np.array([get(k, model)['mean'] for k in configs])
            spread = np.array([get(k, model)['seed_sd'] for k in configs])
            style = 'x:' if model == 'query_known_bayes' else 'o-'
            ax.plot(x, values, style, label=label, color=color)
            ax.fill_between(x, np.maximum(0, values-spread), np.minimum(1, values+spread), color=color, alpha=.09)
        ax.set(title=title, ylabel='Test balanced accuracy', ylim=(.4, 1.03))
        ax.axhline(.5, color='#a9adb5', linestyle='--', linewidth=.7)
axes[1].set_xlabel('Input qubits (Bell uses twice as many)'); axes[1].legend(fontsize=8)
axes[2].set_xlabel('State copies per example per protocol'); axes[2].set_xscale('log', base=2)
fig.suptitle('Application prototypes: quantum benefit depends on representation and data access')
dest = ROOT/'docs/assets'; dest.mkdir(exist_ok=True)
fig.savefig(dest/'application_comparisons.png', dpi=160)
fig.savefig(dest/'application_comparisons.svg')
plt.close(fig)
print(f'Rendered {len(data["cases"])} configurations and {len(base_rows)} base evaluations; {failures} final failures.')
