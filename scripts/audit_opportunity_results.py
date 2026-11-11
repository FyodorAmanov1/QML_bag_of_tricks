"""Independently audit saved application predictions, selection and resources."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
raw = gzip.decompress((ROOT/'results/opportunities.json.gz').read_bytes())
data = json.loads(raw)
compact = json.loads((ROOT/'results/opportunities_summary.json').read_text(encoding='utf-8'))
assert hashlib.sha256(raw).hexdigest() == compact['full_json_sha256']
assert hashlib.sha256(json.dumps(data['protocol'], sort_keys=True).encode()).hexdigest() == data['protocol_sha256']
source = ROOT/'src/qml_tricks'
for name, digest in data['source_hashes'].items():
    contents = (source/name).read_bytes()
    # Windows Git checkouts may use CRLF; Linux uses LF. Preserve the original
    # snapshot hash and accept only that byte content under either EOL spelling.
    lf = contents.replace(b'\r\n', b'\n')
    accepted = {hashlib.sha256(b).hexdigest() for b in (contents, lf, lf.replace(b'\n', b'\r\n'))}
    assert digest in accepted, f'Source differs beyond line endings: {name}'

count = 0; base = 0
for case in data['cases']:
    regression = case['application'] == 'ramsey_calibration'
    primary = 'rmse' if regression else 'balanced_accuracy'
    groups = [case['split_indices'][name] for name in ('train', 'validation', 'test')]
    train, val, test = groups
    assert all(not set(a)&set(b) for i,a in enumerate(groups) for b in groups[i+1:])
    target = np.array(case['targets'] if regression else case['labels'])
    assert sorted(train+val+test) == list(range(len(target)))
    if regression:
        phase = np.array(case['phase'])
        assert np.all((phase[train+val]>=0)&(phase[train+val]<2*np.pi))
        assert np.all((phase[test]>=2*np.pi)&(phase[test]<6*np.pi))
        expected = 0 if case['shots'] is None else len(target)*6*case['shots']
        assert case['acquisition']['hypothetical_hardware_state_copies'] == expected
    else:
        for indices in groups:
            assert np.sum(target[indices]) == len(indices)//2
        assert case['acquisition']['bell_pairs_per_state']*2 == case['state_copies']
        assert case['acquisition']['total_copies_per_protocol'] == len(target)*case['state_copies']
        assert case['acquisition']['max_simultaneous_qubits_bell'] == 2*case['qubits']
    names = {r['model']:r for r in case['models']}
    for row in case['models']:
        if row['model'] != 'classical_selected':
            base += 1
        if 'test' not in row:
            assert 'error' in row
            continue
        y, pred = np.array(row['test_targets']), np.array(row['test_predictions'])
        np.testing.assert_array_equal(y, target[test])
        score = float(np.sqrt(mean_squared_error(y,pred))) if regression else float(balanced_accuracy_score(y,pred))
        np.testing.assert_allclose(score, row['test'][primary], atol=1e-12)
        if regression:
            np.testing.assert_allclose(mean_absolute_error(y,pred), row['test']['mae'], atol=1e-12)
        if row['trials']:
            assert len(row['trials']) == 9
            candidates = [t for t in row['trials'] if 'validation' in t]
            best = (min if regression else max)(candidates, key=lambda t:t['validation'][primary])
            assert best['trial'] == row['selected_trial']
        if row['model'] == 'classical_selected':
            winner = names[row['selected_family']]
            np.testing.assert_array_equal(pred, winner['test_predictions'])
            assert row['test'] == winner['test']
            eligible = [r for r in names.values() if r['model'] not in
                        ('fourier_ridge','quantum_ramsey_ridge','bell_logistic_l2',
                         'local_bayes','query_known_bayes','classical_selected') and 'test' in r]
            values = {r['model']:next(t['validation'][primary] for t in r['trials'] if t['trial']==r['selected_trial']) for r in eligible}
            chosen = (min if regression else max)(values,key=values.get)
            assert row['selected_family'] == chosen
        count += 1
assert base == compact['base_evaluations']
assert len(data['cases']) == compact['experiment_configurations']
print(f'Audited {len(data["cases"])} configurations, {base} base evaluations and {count} saved metric rows; source hashes, predictions, selection and copy accounting match.')
