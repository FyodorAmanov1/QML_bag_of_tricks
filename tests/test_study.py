"""Selection isolation and provenance checks for the real-data protocol."""
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from qml_tricks.learning.study import dataset,run_study


def test_dataset_provenance_and_binary_filter():
    x,y,task,meta=dataset('wine_binary',40)
    assert task=='classification' and len(x)==40 and set(y)=={0,1}
    assert len(meta['source_rows'])==40 and len(meta['content_sha256'])==64
    assert meta['original_shape']==[178,13]


def test_real_study_selection_and_untouched_test(tmp_path):
    path=tmp_path/'study.json'
    result=run_study(path,seeds=(7,),limit=40,trials=2,datasets=('wine_binary',),models=('linear','linear_pca2','frozen_features'))
    assert len(result['runs'])==3
    for row in result['runs']:
        tr,va,te=map(set,(row['train_indices'],row['validation_indices'],row['test_indices']))
        assert not tr&va and not tr&te and not va&te
        assert set(row['refit_indices'])==tr|va
        assert len(row['trials'])==2
        assert row['selected_trial']==max(row['trials'],key=lambda t:t['validation']['balanced_accuracy'])['trial']
        assert row['test']['balanced_accuracy']==balanced_accuracy_score(row['test_targets'],row['test_predictions'])
        assert 'error' not in row
    assert json.loads(path.read_text())['protocol']['trials_per_model']==2


def test_notebooks_have_executed_outputs_and_no_errors():
    from pathlib import Path
    import nbformat
    root=Path(__file__).resolve().parents[1]
    for p in (root/'notebooks').glob('*.ipynb'):
        nb=nbformat.read(p,as_version=4)
        for cell in nb.cells:
            if cell.cell_type=='code':
                assert cell.execution_count is not None
                assert all(o.output_type!='error' for o in cell.outputs)
