"""Sample-based Born machine with a characteristic binary-support kernel."""
from importlib import import_module
import numpy as np
from .methods import born_mmd_gradient


class BornMachine:
    def __init__(self,framework='qiskit',n_qubits=2,depth=1,steps=5,seed=7,batch_size=32):
        self.framework,self.n,self.depth,self.steps,self.seed,self.batch_size=framework,n_qubits,depth,steps,seed,batch_size
        if not 1<=n_qubits<=6 or depth<1 or steps<0 or batch_size<2:raise ValueError('Invalid bounded Born-machine configuration')

    def sample(self,count):
        if count<1:raise ValueError('Positive sample count required')
        state=import_module(f'qml_tricks.learning.backends.{self.framework}').state(self.theta_,np.zeros(self.n),self.n,self.depth)
        p=np.abs(state)**2;p/=p.sum()
        self.resources_['native_state_simulations']+=1;self.resources_['simulated_measurement_shots']+=count
        indices=self.rng_.choice(2**self.n,count,p=p)
        return ((indices[:,None]>>np.arange(self.n-1,-1,-1)) & 1)

    def fit(self,x,y=None):
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=self.n or not np.isin(x,[0,1]).all():raise ValueError('Expected binary sample matrix with n_qubits columns')
        self.rng_=np.random.default_rng(self.seed);self.theta_=self.rng_.normal(0,.1,self.n*self.depth)
        self.resources_=dict(native_state_simulations=0,simulated_measurement_shots=0)
        for _ in range(self.steps):
            def sampler(theta,count):
                previous=self.theta_;self.theta_=theta
                try:return self.sample(count)
                finally:self.theta_=previous
            self.theta_-=.1*born_mmd_gradient(sampler,self.theta_,x,self.batch_size)
        return self
