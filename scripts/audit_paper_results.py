"""Independently verify paper reproduction metrics, splits and provenance."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1]
studies={name:json.loads(gzip.decompress((ROOT/f'results/paper_{name}.json.gz').read_bytes())) for name in ('alloys','communications')}
source=ROOT/'src/qml_tricks';count=0
for study in studies.values():
    for name,digest in study['source_hashes'].items():
        # CLI routing evolved; verify its pinned historical bytes, while all
        # numerical source files must still match their original checksums.
        path=ROOT/'reproductions/source_snapshots/__main__.py.upstream' if name=='__main__.py' else source/name
        b=path.read_bytes();lf=b.replace(b'\r\n',b'\n')
        assert digest in {hashlib.sha256(a).hexdigest() for a in (b,lf,lf.replace(b'\n',b'\r\n'))},name
    for item in study['provenance']:
        if item['local_path'].startswith('reproductions/'):
            b=(ROOT/item['local_path']).read_bytes();lf=b.replace(b'\r\n',b'\n')
            assert item['sha256'] in {hashlib.sha256(a).hexdigest() for a in (b,lf,lf.replace(b'\n',b'\r\n'))}
data=np.loadtxt(ROOT/'reproductions/alloys/data.csv',delimiter=',',skiprows=1)
alloys=studies['alloys']['alloys']
for case in alloys['fresh_replays']+alloys['controlled']:
    train,test=case['train_indices'],case['test_indices'];val=case.get('validation_indices',[])
    assert not set(train)&set(test) and not set(val)&set(test) and not set(train)&set(val)
    for row in case['models']:
        np.testing.assert_array_equal(row['test_targets'],data[test,-1])
        y,pred=row['test_targets'],row['test_predictions']
        assert row['test']['accuracy']==accuracy_score(y,pred)
        assert row['test']['balanced_accuracy']==balanced_accuracy_score(y,pred)
        count+=1
        if row['model']=='quantum' and val:
            expected=set(train)|set(val)
            for key in ('maximum_fit_indices','pca_fit_indices','bounds_fit_indices'):
                assert set(row['preprocessing'][key])==expected
            assert row['selected']==max(row['trials'],key=lambda r:r['validation_accuracy'])
        elif val:
            best=max(row['trials'],key=lambda r:r['validation_accuracy'])
            assert row['parameters']==best['parameters']
for case in studies['communications']['communications']['runs']:
    for row in case['models']:
        y=np.array(case['test_labels']).reshape(-1)
        np.testing.assert_array_equal(row['test_targets'],y)
        assert row['test']['accuracy']==accuracy_score(y,row['test_predictions'])
        assert row['test']['balanced_accuracy']==balanced_accuracy_score(y,row['test_predictions'])
        if 'trials' in row:
            assert row['parameters']['choice']==max(row['trials'],key=lambda r:r['validation_accuracy'])['choice']
        if row['model'].endswith('_train_only'):
            assert row['training_messages']==case['train_messages']
            original=next(m for m in case['models'] if m['model']==row['selection_reused_from'])
            assert row['parameters']==original['parameters']
        count+=1
extra=studies['communications'].get('matched_training_audit')
if extra:
    contents=(ROOT/extra['script']).read_bytes();lf=contents.replace(b'\r\n',b'\n')
    assert extra['script_sha256'] in {hashlib.sha256(b).hexdigest() for b in (contents,lf,lf.replace(b'\n',b'\r\n'))}
print(f'Audited {count} fresh model evaluations, held-out targets/metrics, clean fit indices, validation choices and pinned source content.')
