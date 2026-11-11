"""Post-run fairness audit: refit selected classical controls on train only.

No new hyperparameters or seeds are selected. The original paper-style quantum
and memoryless models used 100 training messages; the first extension also
refit classical models on 20 validation messages. Retain both comparisons.
"""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from qml_tricks.learning.paper_reproductions import channel_messages,delay_features,fit_record

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'results/paper_communications.json.gz'
data=json.loads(gzip.decompress(path.read_bytes()));protocol=data['protocol']['communications']
with threadpool_limits(limits=1):
    for case in data['communications']['runs']:
        rng=np.random.RandomState(case['seed']);length=protocol['symbols_per_message']
        parts=[channel_messages(rng,protocol[key],length,protocol['snr_db'])
               for key in ('train_messages','validation_messages','test_messages')]
        inputs=np.concatenate([p[0] for p in parts]);y=np.concatenate([p[1] for p in parts]).reshape(-1)
        np.testing.assert_allclose(parts[-1][0],case['test_inputs'],rtol=0,atol=0)
        tr=np.arange(parts[0][0].size);te=np.arange(parts[0][0].size+parts[1][0].size,len(y))
        case['models']=[r for r in case['models'] if not r['model'].endswith('_train_only')]
        for name in ('polynomial_delay_logistic','delay_mlp_l2'):
            original=next(r for r in case['models'] if r['model']==name)
            params=original['parameters'];choice=params['choice']
            model=(LogisticRegression(C=(.01,1.,100.)[choice%3],max_iter=500)
                   if name=='polynomial_delay_logistic' else
                   MLPClassifier(hidden_layer_sizes=((16,),(32,),(64,))[choice//3],
                                 alpha=(.0001,.01,1.)[choice%3],random_state=case['seed'],max_iter=200))
            x=delay_features(inputs,params['degree'])
            result=fit_record(name+'_train_only',make_pipeline(StandardScaler(),model),x,y,tr,te,params)
            result.update(training_messages=protocol['train_messages'],validation_used_only_for_selection=True,
                          selection_reused_from=name)
            case['models'].append(result)
data['matched_training_audit']=dict(status='Post-run fairness audit, not a pre-specified new superiority test',
    selection='Same families, seeds and validation-selected hyperparameters; final fitting uses exactly the quantum replay training messages',
    script='scripts/match_communication_training_budget.py',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path.write_bytes(gzip.compress(json.dumps(data,indent=2,allow_nan=False).encode(),mtime=0))
print('Retained original results and added six train-only classical fits with matched quantum training budgets.')
