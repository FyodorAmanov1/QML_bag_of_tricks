"""Render observed study results and release artwork without invented outcomes."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/study.json').read_text())
assets=ROOT/'docs/assets';assets.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
for ax,name in zip(axes,('wine_binary','breast_cancer','diabetes')):
    rows=[r for r in data['summary'] if r['dataset']==name]
    positions=np.arange(len(rows));means=np.array([r['mean'] for r in rows])
    lower=np.array([r['seed_mean_bootstrap_interval_95'][0] for r in rows]);upper=np.array([r['seed_mean_bootstrap_interval_95'][1] for r in rows])
    colors=['#4f5cce' if r['model'] in ('frozen_features','projected_kernel','fidelity_kernel') else '#168f84' for r in rows]
    ax.barh(positions,means,color=colors,xerr=np.vstack([means-lower,upper-means]),capsize=3)
    ax.set_yticks(positions,[r['model'].replace('_',' ') for r in rows]);ax.invert_yaxis()
    ax.set_title(name.replace('_',' '));ax.set_xlabel('RMSE (lower is better)' if name=='diabetes' else 'Balanced accuracy (higher is better)')
    if name!='diabetes':ax.set_xlim(0,1.05)
fig.suptitle('Small real-data study · 10 repeated splits · 3 validation candidates per model')
for ext in ('png','svg'):fig.savefig(assets/f'real_data_study.{ext}',dpi=150)
plt.close(fig)

lines=['# Small real-data benchmark study','','![Observed model comparisons](assets/real_data_study.png)','','## What was measured','',
       f"The recorded run contains {len(data['runs'])} selected model/split evaluations across three real-observation datasets, with {sum('error' in r for r in data['runs'])} failed final evaluations. It completed in {data['elapsed_seconds']:.1f} seconds on the recorded machine. Runtime is machine-specific.",'',
       'Each dataset uses a fixed 128-row subset, ten split seeds and three candidate configurations per model. Candidates see only training/validation data; the selected configuration is refit on train+validation and evaluated once on its untouched test partition. All preprocessing and target scaling is fitted within the relevant fit partition.','',
       'Full-feature classical models and a two-PCA-component classical linear control are included. The three quantum models use two qubits, one layer and ideal native Qiskit simulation. Equal trial counts do not imply equal compute budgets; every candidate records time, warnings and available circuit/measurement costs.','',
       '## Observed results','','| Dataset | Model | Metric | Mean | Seed SD | Descriptive 95% interval | Mean selection/refit seconds |','|---|---|---|---:|---:|---|---:|']
for r in data['summary']:
    lo,hi=r['seed_mean_bootstrap_interval_95'];lines.append(f"| {r['dataset']} | {r['model']} | {r['metric']} | {r['mean']:.4f} | {r['std']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['mean_selection_refit_seconds']:.3f} |")
lines+=['','## Interpretation and limits','',
        'The tested classical configurations have better mean predictive scores than these small quantum-feature configurations on all three datasets. This is a result about the recorded pipelines, subset sizes and tuning grids. It is not a conclusion about every QML method or every dataset. Representation compression, angle encoding, circuit family and readout all change the model. The two-feature classical control helps expose, but does not isolate, those effects.','',
        'Intervals bootstrap the means across overlapping repeated splits. They are descriptive, not independent-replication confidence guarantees or significance tests. A real-observation dataset can still be too small for practical generalization claims. The medical datasets are numerical research benchmarks, not validated clinical tools.','',
        'Only selected hyperparameters have been explored. No noisy hardware runs, broad AutoML search, learning curves over large sample sizes or quantum-advantage evidence are established. The original dataset descriptions and content hashes are stored in results/study.json. No test score was used to select a candidate.','',
        '## Reproduce and extend','','```bash','python -m qml_tricks.learning.study --seeds 0 1 2 3 4 5 6 7 8 9 --limit 128 --trials 3 --output results/study.json','python scripts/build_study_report.py','```','',
        'The study uses the bundled wine (classes 0 and 1), Wisconsin breast-cancer and diabetes observations, avoiding a runtime dataset download. See [scikit-learn dataset provenance](https://scikit-learn.org/stable/datasets/toy_dataset.html) and the embedded original descriptions. These small datasets are explicitly described by scikit-learn as illustrative.','',
        'For a larger follow-up, freeze a new protocol before running additional datasets or encoders. Preserve these negative/default-configuration results rather than replacing them with a favorable subset.']
(ROOT/'docs/study.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Simple code-native release artwork, editable through this script.
fig=plt.figure(figsize=(12.8,6.4),dpi=100,facecolor='#101a32');ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
ax.text(.07,.80,'QML BAG OF TRICKS',fontsize=37,color='#ffffff',weight='bold',transform=ax.transAxes)
ax.text(.07,.63,'Choose. Implement. Benchmark.',fontsize=27,color='#83e0ce',transform=ax.transAxes)
ax.text(.07,.45,'Agent-friendly recipes for quantum machine learning',fontsize=19,color='#e6eafa',transform=ax.transAxes)
ax.text(.07,.30,'Qiskit  /  PennyLane  /  Cirq',fontsize=19,color='#b6c0f3',transform=ax.transAxes)
ax.text(.07,.15,'Explicit assumptions · Classical controls · Reproducible artifacts',fontsize=15,color='#e6eafa',transform=ax.transAxes)
fig.savefig(assets/'social-preview.png',facecolor=fig.get_facecolor());fig.savefig(assets/'social-preview.svg',facecolor=fig.get_facecolor());plt.close(fig)
print('Rendered study report, figures and release artwork.')
