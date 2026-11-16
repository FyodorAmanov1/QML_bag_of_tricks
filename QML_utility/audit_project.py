"""Recompute saved metrics and validation selection from original CSV records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .project_data import load_data


def independent_score(task,truth,prediction):
    truth,prediction = np.asarray(truth),np.asarray(prediction)
    if truth.shape != prediction.shape or not len(truth):
        raise ValueError('Prediction/target shape mismatch')
    if task == 'regression':
        return float(np.sqrt(np.mean((truth.astype(float)-prediction.astype(float))**2)))
    return float(np.mean([np.mean(prediction[truth==label]==label) for label in np.unique(truth)]))


def audit(directory,config,base=Path('.')):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text())
    if json.loads((directory/'config.json').read_text()) != config:
        raise ValueError('Original config differs from frozen snapshot')
    x,y,rows,ids,payload = load_data(config,base)
    if hashlib.sha256(payload).hexdigest() != report['data_sha256']:
        raise ValueError('CSV hash mismatch')
    for relative,digest in report['source_sha256'].items():
        snapshot = directory/'source'/'QML_utility'/relative
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError('Source snapshot hash mismatch')
    positions = {identity:index for index,identity in enumerate(ids)}
    for relative,digest in report.get('library_source_sha256',{}).items():
        if hashlib.sha256((directory/'source/qml_tricks'/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Canonical library source snapshot hash mismatch')
    subsets = {key:[positions[identity] for identity in values] for key,values in report['splits'].items()}
    if any(set(subsets[a])&set(subsets[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('Overlapping saved partitions')
    selection = json.loads((directory/'selection.json').read_text())
    completed = {}
    for trial in report['trials']:
        if trial['status'] == 'completed':
            actual = independent_score(config['task'],y[subsets['validation']],trial['validation_predictions'])
            if not np.isclose(actual,trial['validation_score'],atol=1e-12,rtol=0):
                raise ValueError('Validation metric mismatch')
            completed.setdefault(trial['method'],[]).append(trial)
    choose = max if config['task']=='classification' else min
    expected = {method:choose(trials,key=lambda trial:trial['validation_score']) for method,trials in completed.items()}
    if set(expected) != set(selection['selected']):
        raise ValueError('Missing/extra selected family')
    for method,trial in expected.items():
        if selection['selected'][method] != dict(validation_score=trial['validation_score'],candidate=trial['candidate']):
            raise ValueError('Candidate selection differs from validation rule')
    champion = choose(expected,key=lambda method:expected[method]['validation_score']) if expected else None
    if champion != selection['validation_champion'] or champion != report['validation_champion']:
        raise ValueError('Champion selection differs from validation rule')
    tested = 0
    for result in report['models']:
        if result['selected_candidate'] != selection['selected'][result['method']]['candidate']:
            raise ValueError('Tested candidate differs from frozen choice')
        if result['status'] != 'evaluated':
            continue
        if result['test_ids'] != report['splits']['test'] or result['test_targets'] != y[subsets['test']].tolist():
            raise ValueError('Test target or ID mismatch')
        actual = independent_score(config['task'],y[subsets['test']],result['test_predictions'])
        if not np.isclose(actual,result['test_score'],atol=1e-12,rtol=0):
            raise ValueError('Test metric mismatch')
        artifact = directory/result['artifact']
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != result['artifact_sha256']:
            raise ValueError('Model artifact hash mismatch')
        tested += 1
    return dict(verified=True,completed_validation_trials=sum(map(len,completed.values())),
                evaluated_families=tested,partial=report['status']!='completed',
                scope='Saved metric, data, selection and artifact consistency; not performance significance or advantage')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.run,json.loads(args.config.read_text()),args.config.resolve().parent),indent=2))


if __name__ == '__main__':
    main()
