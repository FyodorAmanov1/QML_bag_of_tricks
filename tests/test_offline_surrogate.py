"""Frozen teacher, measured-pair distillation, validation and classical export."""
import numpy as np
import pytest
from qml_tricks.learning.offline_surrogate import OfflineFeatureSurrogate,ClassicalFeatureStudent
from qml_tricks.learning.models import QMLModel


def pairs():
    rng=np.random.default_rng(83);x=rng.normal(size=(100,2))
    q=x@np.array([[.3,.4],[-.2,.1]])+.2
    y=(x[:,0]-.4*x[:,1]>0).astype(int)
    return x,q,y


def test_measured_feature_pairs_and_teacher_free_export(tmp_path,monkeypatch):
    x,q,y=pairs()
    model=OfflineFeatureSurrogate(alpha=1e-6).fit_from_pairs(x[:60],q[:60],y[:60],x[60:80],q[60:80],y[60:80])
    assert model.validation_['passed']
    assert model.validation_['relative_feature_rmse']<1e-5
    assert model.validation_['teacher_student_agreement']==1
    path=tmp_path/'student.joblib';model.export_student(path)
    student=ClassicalFeatureStudent.load(path)
    assert not hasattr(student,'teacher_')
    # Inference must not consult ANY native SDK, even after reloading.
    import qml_tricks.learning.models as models
    monkeypatch.setattr(models,'import_module',lambda *a: (_ for _ in ()).throw(AssertionError('quantum call')))
    np.testing.assert_array_equal(student.predict(x[80:]),model.predict(x[80:]))
    assert (tmp_path/'student.joblib.validation.json').exists()


def test_failed_fidelity_blocks_deployment():
    rng=np.random.default_rng(4);x,q,y=pairs()
    model=OfflineFeatureSurrogate(max_relative_feature_rmse=.01).fit_from_pairs(x[:60],rng.normal(size=(60,2)),y[:60],x[60:80],q[60:80],y[60:80])
    assert not model.validation_['passed']
    with pytest.raises(ValueError,match='failed'):model.student()


@pytest.mark.parametrize('sdk',['qiskit','pennylane','cirq'])
def test_quantum_teacher_is_trained_before_and_frozen_during_distillation(sdk):
    x,q,y=pairs()
    teacher=QMLModel('reuploading',framework=sdk,steps=1).fit(x[:20],y[:20])
    initial=teacher.theta_.copy();calls=teacher.resources_['state_preparations']
    distiller=OfflineFeatureSurrogate().fit_teacher(teacher,x[:20],y[:20],x[20:30],y[20:30])
    np.testing.assert_array_equal(initial,teacher.theta_)
    np.testing.assert_array_equal(initial,distiller.teacher_.theta_)
    assert teacher.resources_['state_preparations']==calls
    assert distiller.resources_['teacher_parameters_frozen']


def test_no_validation_leakage_and_no_retraining_during_audit():
    x,q,y=pairs();model=OfflineFeatureSurrogate().fit_from_pairs(x[:60],q[:60],y[:60],x[60:80],q[60:80],y[60:80])
    original=model.surrogate_.predict(x[80:]).copy()
    model.audit_pairs(x[80:],q[80:]+100,y[80:])
    np.testing.assert_array_equal(original,model.surrogate_.predict(x[80:]))
    with pytest.raises(ValueError,match='Overlapping'):
        OfflineFeatureSurrogate().fit_from_pairs(x[:60],q[:60],y[:60],x[:20],q[:20],y[:20])
