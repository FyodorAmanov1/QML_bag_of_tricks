"""Build a paper-reproduction report without promoting archive replay to rerun."""
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
data={name:json.loads(gzip.decompress((ROOT/f'results/paper_{name}.json.gz').read_bytes()))
      for name in ('alloys','communications')}
a=data['alloys']['alloys'];c=data['communications']['communications']


def aggregate(cases):
    names=list(dict.fromkeys(m['model'] for case in cases for m in case['models']))
    return {name:dict(mean=float(np.mean([m['test']['accuracy'] for case in cases for m in case['models'] if m['model']==name])),
                     seed_sd=float(np.std([m['test']['accuracy'] for case in cases for m in case['models'] if m['model']==name],ddof=1))) for name in names}


replays={n:aggregate([r for r in a['fresh_replays'] if r['qubits']==n]) for n in (8,10)}
control=aggregate(a['controlled']);comm=aggregate(c['runs'])
classical_differences=[]
for case in a['fresh_replays']:
    archived=next(r for r in a['author_archive'] if r['qubits']==case['qubits'])
    for j,name in enumerate(('linear','poly','rbf','sigmoid')):
        score=next(m['test']['accuracy'] for m in case['models'] if m['model']==name)
        classical_differences.append(abs(score-archived['classical_scores'][case['seed']-22][j]))
compact=dict(alloys=dict(archive=a['author_archive'],fresh_replays=replays,controlled=control,
    maximum_archived_score_disagreement=max(abs(r['fresh_minus_archived']) for r in a['fresh_replays']),
    maximum_classical_archived_score_disagreement=max(classical_differences),dataset=a['dataset']),
    communications=comm,
    limits='Bounded numerical replications. Alloy archives are author-supplied, fresh replay uses 10/300 splits per architecture, and communications uses three new datasets at one SNR. No hardware, new alloys or language-model reproduction.')
(ROOT/'results/paper_summary.json').write_text(json.dumps(compact,indent=2)+'\n',encoding='utf-8')
lines=['# Classical-application paper reproduction results','',
       '![Paper reproductions and stronger controls](../assets/paper_reproductions.png)','',
       'These are two classical-input applications. The archive calculation, fresh simulations and stronger-baseline tests are reported separately. No hardware result, new physical material or general quantum advantage is claimed.','',
       '## Alloy phases: archived 300-split statistics','',
       'This table recomputes the authors’ stored scores; it is **not** 300 new circuit runs. Higher accuracy is better.','',
       '| Qubits | Quantum SVC | Linear SVC | Polynomial SVC | RBF-SVC | Sigmoid SVC |','|---|---:|---:|---:|---:|---:|']
for row in a['author_archive']:
    lines.append(f"| {row['qubits']} | {row['quantum_mean']:.6f} | "+' | '.join(f"{row['classical_means'][k]:.6f}" for k in ('linear','poly','rbf','sigmoid'))+' |')
lines+=['','The ten-qubit archive mean recovers the reported 89.4% rounded figure. Its best classical archived mean is a different quantity from a modern tuned full-feature baseline.','',
        '## Alloy phases: fresh native circuit replay','',
        'Twenty fresh quantum evaluations cover seeds 22–31 for each of eight and ten qubits, with four classical comparators each. They retain the pinned author script’s preprocessing, including held-out feature access for scaling. Full SVD uses an explicit legacy PCA sign convention. Each score is compared with the corresponding archived seed.','',
        '| Qubits | Quantum SVC | Linear SVC | Polynomial SVC | RBF-SVC | Sigmoid SVC |','|---|---:|---:|---:|---:|---:|']
for n in (8,10):
    lines.append(f'| {n} | '+' | '.join(f"{replays[n][k]['mean']:.6f}" for k in ('quantum','linear','poly','rbf','sigmoid'))+' |')
delta=compact['alloys']['maximum_archived_score_disagreement']
lines += ['',f'Maximum absolute fresh-versus-archived quantum score difference on these seeds: **{delta:.12f}**. The corresponding maximum difference for all 80 classical comparator scores is **{max(classical_differences):.12f}**. This is a numerical check, not an assumption that all legacy dependencies are interchangeable.','',
    '## Alloy phases: train-only preprocessing and tuned baselines','',
    'Ten new stratified 60/20/20 splits use validation-only selection, then train+validation refitting and untouched-test prediction. Quantum selection covers eight/ten qubits and three C values. Classical controls use all 15 raw descriptors and training-only scaling. They have 5 (L2-logistic), 12 (RBF-SVC) and 9 (histogram boosting) candidate configurations.','',
    '| Model | Mean test accuracy | Seed SD |','|---|---:|---:|']
for name,row in control.items():lines.append(f"| {name} | {row['mean']:.6f} | {row['seed_sd']:.6f} |")
lines+=['',
    'These rows test a stronger claim than archive recovery. Report them even if the original quantum ordering does not survive. The dataset includes '+str(a['dataset']['duplicate_feature_rows'])+' repeated descriptor rows; random row splits do not establish transfer to new composition families. The paper’s full 300-split parameter sweep, language model and alloy-candidate design are outside this reproduction.','',
    '## Classical wireless channel equalization','',
    'A connected six-qubit reservoir processes classical four-PAM messages distorted by the paper’s nonlinear channel at 20 dB. Three independent seeded datasets each contain 100 training, 20 validation and 100 test messages of 100 symbols. The original quantum readout and memoryless comparator use training messages only; stronger temporal controls refit on train+validation after selecting nine candidates each. Lower symbol error is better.','',
    '| Model | Mean test symbol error | Across-seed SD |','|---|---:|---:|']
for name,row in comm.items():lines.append(f"| {name} | {1-row['mean']:.6f} | {row['seed_sd']:.6f} |")
lines += ['',
    'Rows ending in `train_only` are a separately recorded post-run fairness audit: the same previously selected classical configurations are fit on exactly the original 100 training messages, matching the quantum model. They introduce no new candidate grid or seed, and the earlier train+validation fits are retained.', '',
    'Published numerical references at this SNR are 0.0021 error for the ideal connected reservoir and 0.0446 for direct logistic regression. Those are reference values, not regenerated raw datasets. The notebook uses unseeded message generation, so exact printed-number reproduction is not claimed. The fresh results test the stated architecture and ordering on independently generated messages.', '',
    'The stronger controls use causal eight-sample windows: polynomial per-lag features plus L2 logistic regression, and an L2-regularized MLP. Beating a memoryless classifier does not establish superiority to these or to all classical equalizers. Exact reservoir features here use classical SciPy eigendecomposition and mean-channel contraction, not QPU sampling. The hardware experiment and finite-shot trajectories are excluded.', '',
    '## Reproducibility and limits','',
    'The [reproduction guide](index.md) contains commands and links to the peer-reviewed papers. Pinned author commit IDs, artifact hashes, the original alloy data and its upstream license are under `reproductions/`. Full local predictions and configurations are stored as `results/paper_alloys.json.gz` and `results/paper_communications.json.gz`; `results/paper_summary.json` is readable without decompression.', '',
    'Run `python scripts/audit_paper_results.py` to independently recompute held-out scores, validate split isolation, validate clean preprocessing fit indices and check artifact hashes. These bounded runs cannot establish hardware speedup or population-wide superiority. Repeated alloy row splits overlap; their SD is descriptive, not an independent-replication confidence interval. Communications has three seeds, not a powered superiority study.', '',
    'Timing fields are implementation diagnostics, not a matched cost comparison: quantum alloy timing includes feature acquisition and selection, while classical per-model fields cover final refit/prediction. Hardware data loading, shot estimation and compilation would add costs.']
(ROOT/'docs/reproductions/results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
for ax,values,title,error in [(axes[0],replays[10],'Alloys: fresh author-workflow replay',False),
                              (axes[1],control,'Alloys: train-only tuned comparison',False),
                              (axes[2],comm,'Communications: fresh 20 dB simulation',True)]:
    names=list(values);means=[1-values[k]['mean'] if error else values[k]['mean'] for k in names]
    spreads=[values[k]['seed_sd'] for k in names]
    if error:
        labels={'quantum_reservoir':'Quantum reservoir','memoryless_logistic':'Memoryless logistic',
                'polynomial_delay_logistic':'Polynomial delays (+val)', 'delay_mlp_l2':'MLP delays (+val)',
                'polynomial_delay_logistic_train_only':'Polynomial delays (matched)',
                'delay_mlp_l2_train_only':'MLP delays (matched)'}
        for i,(name,mean,spread) in enumerate(zip(names,means,spreads)):
            ax.errorbar(mean,i,xerr=spread,fmt='o',capsize=3,color='#4f5cce' if name=='quantum_reservoir' else '#168f84')
        ax.set_yticks(range(len(names)),[labels[k] for k in names]);ax.set_xscale('log')
    else:
        ax.barh(names,means,xerr=spreads,color=['#4f5cce' if k=='quantum' else '#168f84' for k in names],capsize=3)
    ax.invert_yaxis();ax.set_title(title,fontsize=10)
    ax.set_xlabel('Symbol error (lower is better)' if error else 'Accuracy (higher is better)')
    ax.set_xlim((.0002,.1) if error else (0,1.))
fig.suptitle('Two classical-data papers: fresh simulations and stronger controls; bars show seed SD',fontsize=12)
fig.savefig(ROOT/'docs/assets/paper_reproductions.png',dpi=160)
fig.savefig(ROOT/'docs/assets/paper_reproductions.svg')
print('Rendered both paper studies and their complete controlled comparisons.')
