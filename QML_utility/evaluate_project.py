"""Prespecified repeated holdouts or expanding-window evaluation, with paired results."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from .project_data import load_data,split_indices
from .run_project import run
from .audit_project import audit


def evaluation_plan(config,base=Path('.')):
    x,y,rows,ids,payload = load_data(config,base)
    specification = config.get('evaluation',{})
    kind = specification.get('kind','repeated_holdout')
    baseline = specification.get('baseline','linear')
    allowed = ('dummy','linear','random_features','hist_boosting') if config.get('adapter')=='batched_features' else ('dummy','linear','rbf','boosting','fourier_linear')
    if baseline not in allowed:
        raise ValueError('Baseline must be a classical family in this adapter')
    margin = specification.get('practical_margin',0.)
    if type(margin) not in (int,float) or not np.isfinite(margin) or margin < 0:
        raise ValueError('practical_margin must be finite and nonnegative in primary metric units')
    plans = []
    original = config['data']['split']
    if kind == 'repeated_holdout':
        if original['kind'] not in ('iid','group'):
            raise ValueError('Repeated holdouts require IID/group splits; use walk_forward for chronology')
        seeds = specification.get('seeds',[7,17,27])
        if not isinstance(seeds,list) or not 2 <= len(seeds) <= 20 or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError('Provide 2..20 distinct nonnegative integer seeds, chosen before results')
        for number,seed in enumerate(seeds):
            parts = split_indices(y,rows,original,seed,config['task'],ids)
            plans.append(dict(run_id=f'run_{number:02}',seed=seed,split={key:[ids[i] for i in values] for key,values in parts.items()}))
    elif kind == 'walk_forward':
        if original['kind'] != 'chronological':
            raise ValueError('walk_forward requires the chronological time-column contract')
        # Validate unique ISO chronology and the classification target contract.
        split_indices(y,rows,original,config.get('seed',7),config['task'],ids)
        order = sorted(range(len(rows)),key=lambda i:datetime.fromisoformat(rows[i][original['time_column']]))
        folds = specification.get('n_splits',3)
        size = specification.get('test_rows',max(4,len(rows)//10))
        validation = specification.get('validation_rows',size)
        minimum = specification.get('min_train_rows',4)
        gap = original.get('gap_rows',0)
        if any(type(value) is not int or value < 4 for value in (size,validation,minimum)) or type(folds) is not int or not 2 <= folds <= 20:
            raise ValueError('Require 2..20 folds and >=4 train/validation/test rows')
        for number in range(folds):
            start = len(order)-(folds-number)*size
            val_end,val_start = start-gap,start-gap-validation
            train_end = val_start-gap
            if train_end < minimum:
                raise ValueError('Not enough chronological history for this window/gap plan')
            split = dict(train=[ids[i] for i in order[:train_end]],validation=[ids[i] for i in order[val_start:val_end]],
                         test=[ids[i] for i in order[start:start+size]])
            plans.append(dict(run_id=f'run_{number:02}',seed=config.get('seed',7),split=split))
    else:
        raise ValueError('evaluation.kind must be repeated_holdout or walk_forward')
    return dict(schema_version=1,kind=kind,baseline=baseline,practical_margin=margin,
                data_sha256=hashlib.sha256(payload).hexdigest(),runs=plans,
                frozen_before_execution=True,selection='Separate training and validation inside every outer run; test does not select candidates')


def summarize(records,plan,task):
    families = sorted({model['method'] for record in records if record.get('report') for model in record['report']['models']})
    summaries,paired = {},{}
    for method in families:
        values,deltas,missing = [],[],[]
        for record in records:
            lookup = {model['method']:model for model in record.get('report',{}).get('models',[]) if model['status']=='evaluated'}
            if method in lookup:
                values.append(lookup[method]['test_score'])
            if method not in lookup or plan['baseline'] not in lookup:
                missing.append(record['run_id']);continue
            difference = lookup[method]['test_score']-lookup[plan['baseline']]['test_score']
            deltas.append(dict(run_id=record['run_id'],improvement=float(difference if task=='classification' else -difference)))
        summaries[method] = dict(completed_runs=len(values),expected_runs=len(records),
                                 mean=float(np.mean(values)) if values else None,
                                 sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
        improvements = [row['improvement'] for row in deltas]
        paired[method] = dict(baseline=plan['baseline'],per_run=deltas,missing_runs=missing,
                              mean_improvement=float(np.mean(improvements)) if improvements else None,
                              runs_exceeding_practical_margin=sum(value>plan['practical_margin'] for value in improvements))
    tests = [set(row['split']['test']) for row in plan['runs']]
    overlap = [dict(run_a=plan['runs'][i]['run_id'],run_b=plan['runs'][j]['run_id'],shared_test_records=len(tests[i]&tests[j]))
               for i in range(len(tests)) for j in range(i)]
    return dict(families=summaries,paired_comparisons=paired,test_overlap=overlap,
                uncertainty='Descriptive across-run variation only. Repeated partitions overlap and time windows remain temporally dependent; no independent-sample confidence interval or p-value is asserted.',
                no_test_selected_deployment_winner=True,advantage_established=False)


def evaluate(config,output,base=Path('.')):
    plan = evaluation_plan(config,base)
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new evaluation output directory')
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    records = []
    for entry in plan['runs']:
        child = copy.deepcopy(config)
        child.pop('evaluation',None)
        child['seed'] = entry['seed']
        source = Path(child['data']['path'])
        child['data']['path'] = str((Path(base)/source).resolve()) if not source.is_absolute() else str(source)
        child['data']['split'].update(kind='predefined',**{key+'_ids':value for key,value in entry['split'].items()})
        record = dict(run_id=entry['run_id'],status='failed')
        try:
            report = run(child,output/entry['run_id'])
            verified = audit(output/entry['run_id'],child)
            record.update(status=report['status'],report=report,audit=verified)
        except Exception as error:
            record.update(error=str(error),traceback=traceback.format_exc())
        records.append(record)
        # Save partial progress so interruption does not hide completed runs.
        (output/'progress.json').write_text(json.dumps([dict(run_id=row['run_id'],status=row['status']) for row in records],indent=2)+'\n')
    summary = summarize(records,plan,config['task'])
    summary.update(kind=plan['kind'],metric='balanced_accuracy' if config['task']=='classification' else 'rmse',
                   practical_margin=plan['practical_margin'],baseline=plan['baseline'],
                   status='completed' if all(row['status']=='completed' for row in records) else 'partial_with_failures',
                   runs=[{key:value for key,value in record.items() if key!='report'} for record in records])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines = ['# Prespecified multi-run evaluation','',f"Metric: {summary['metric']}. Baseline: {plan['baseline']}.",'',
             '| Family | Completed runs | Mean test metric | Mean paired improvement |','|---|---:|---:|---:|']
    lines += [f"| {method} | {value['completed_runs']}/{value['expected_runs']} | {value['mean']} | {summary['paired_comparisons'][method]['mean_improvement']} |" for method,value in summary['families'].items()]
    lines += ['',summary['uncertainty'],'','Positive improvement favors the row method. No deployment winner was chosen from test outcomes.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    result = evaluate(json.loads(args.config.read_text()),args.output,args.config.resolve().parent)
    print(json.dumps(dict(status=result['status'],runs=len(result['runs']),report=str(args.output/'REPORT.md')),indent=2))


if __name__ == '__main__':
    main()
