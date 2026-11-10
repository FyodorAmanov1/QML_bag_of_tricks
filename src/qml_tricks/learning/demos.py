"""Bounded native examples for every learning/estimation recipe."""
from importlib import import_module
import numpy as np
from .methods import parameter_shift,allocate_shots,pauli_shadows,shadow_expectation,mmd_unbiased,born_mmd_gradient
from .models import QMLModel,MODELS


def run_recipe(method,framework='qiskit',seed=7):
    if method=='kipu_offline_surrogate':
        from .offline_surrogate import OfflineFeatureSurrogate
        rng=np.random.default_rng(seed)
        data=rng.uniform(-1,1,size=(512,2));labels=(np.sum(data**2,axis=1)>.7).astype(int)
        teacher=QMLModel('reuploading',framework=framework,steps=3,seed=seed).fit(data[:100],labels[:100])
        distiller=OfflineFeatureSurrogate(alpha=1e-6,student_kind='rbf_ridge').fit_teacher(teacher,data[:384],labels[:384],data[384:448],labels[384:448])
        qtest=distiller.teacher_._features(distiller.teacher_._transform(data[448:]))
        test=distiller.audit_pairs(data[448:],qtest,labels[448:])
        before=distiller.teacher_.resources_['state_preparations']
        if distiller.validation_['passed']:
            distiller.student().predict(data[448:])
        inference_calls=distiller.teacher_.resources_['state_preparations']-before
        return dict(method=method,framework=framework,teacher='trained native re-uploading reference, not Kipu DQFE',
                    validation=distiller.validation_,untouched_test=test,resources=distiller.resources_,
                    student_inference_quantum_calls=inference_calls,
                    deployment_allowed=distiller.validation_['passed'])
    state=import_module(f'qml_tricks.learning.backends.{framework}').state
    rng=np.random.default_rng(seed);theta=np.array([.3,-.4]);x=np.array([.2,.5])
    if method in MODELS:
        data=rng.normal(size=(24,3));labels=(data[:,0]+data[:,1]>0).astype(int)
        model=QMLModel(method=method,framework=framework,steps=2,seed=seed).fit(data[:16],labels[:16])
        before=dict(model.resources_);pred=model.predict(data[16:])
        result=dict(method=method,framework=framework,accuracy=float(np.mean(pred==labels[16:])),
                    training_resources=before,total_resources=model.resources_,guarantee='See method guide; no generic advantage claim')
        if method=='kernel_surrogate':result['held_out_surrogate_audit']=model.audit_surrogate(data[16:])
        return result
    if method=='parameter_shift':
        z=np.array([1,1,-1,-1])
        def expectation(t):return float(np.abs(state(t,x,2,1))**2@z)
        shift=parameter_shift(expectation,theta)
        finite=[]
        for j in range(2):
            d=np.zeros(2);d[j]=1e-5;finite.append((expectation(theta+d)-expectation(theta-d))/2e-5)
        return dict(parameter_shift=shift.tolist(),central_reference=finite,maximum_error=float(np.max(np.abs(shift-finite))))
    if method=='shot_allocation':
        c=np.array([1.,.2,.5]);v=np.array([.1,.8,.3]);counts=allocate_shots(c,v,300)
        return dict(shots=counts.tolist(),total=int(counts.sum()),known_variance=float(np.sum(c*c*v/counts)),
                    equal_allocation_variance=float(np.sum(c*c*v/100)),pilot_cost_included=False)
    if method=='classical_shadows':
        psi=state(theta,x,2,1);bases,signs=pauli_shadows(psi,2,1000,seed)
        z=np.array([1,1,-1,-1]);return dict(shots=1000,estimate=shadow_expectation(bases,signs,'ZI',groups=5),
            exact=float(np.abs(psi)**2@z),measurement_mode='host simulation of local Pauli measurements')
    if method=='born_mmd':
        target=np.array([[0,0],[1,1]]*16)
        def sampler(t,count):
            psi=state(t,x,2,1);p=np.abs(psi)**2;p/=p.sum()
            indices=rng.choice(4,count,p=p)
            return ((indices[:,None] >> np.array([1,0])) & 1)
        trace=[]
        for _ in range(5):
            trace.append(mmd_unbiased(sampler(theta,64),target))
            theta-=.1*born_mmd_gradient(sampler,theta,target,64)
        return dict(mmd_estimates=trace,parameters=theta.tolist(),kernel='Gaussian on binary vectors',
                    guarantee='unbiased estimators under independent draws, not global convergence')
    raise ValueError(method)
