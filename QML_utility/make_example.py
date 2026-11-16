"""Write a real-observation integration example; not a performance study."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_wine,load_diabetes


def create_example(output,task='classification',framework='qiskit'):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new example directory')
    output.mkdir(parents=True)
    source = load_wine() if task == 'classification' else load_diabetes()
    rng = np.random.default_rng(42)
    # Fixed class-balanced smoke subset or fixed random regression subset;
    # selection never depends on any model outcome. Preserve every feature.
    indices = np.concatenate([rng.choice(np.flatnonzero(source.target==c),24,replace=False) for c in np.unique(source.target)]) if task == 'classification' else rng.choice(len(source.target),72,replace=False)
    names = ['x'+str(i) for i in range(source.data.shape[1])]
    with (output/'data.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['row_id']+names+['target'])
        for i in indices:
            writer.writerow([int(i)]+source.data[i].tolist()+[source.target[i]])
    config = dict(data_kind='classical',modality='numeric_table',task=task,objective='iid_'+task,
                  capabilities={},allow_exploratory=True,framework=framework,seed=7,
                  methods=['quantum_features','projected_kernel','fidelity_kernel','trainable_local'],
                  quantum=dict(qubits=2,layers=1,steps=6),
                  budget=dict(max_rows=512,state_preparations_per_trial=30000,seconds_per_operation=120),
                  data=dict(path='data.csv',features=names,target='target',id_column='row_id',split=dict(kind='iid')),
                  provenance=dict(source='sklearn bundled wine' if task=='classification' else 'sklearn bundled diabetes',
                                  scope='72-record integration smoke example; not paper replication, medical validation or advantage evidence',
                                  source_rows=indices.tolist()))
    (output/'project.json').write_text(json.dumps(config,indent=2)+'\n')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=['classification','regression'],default='classification')
    parser.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    args = parser.parse_args()
    create_example(args.output,args.task,args.framework)
    print(args.output/'project.json')


if __name__ == '__main__':
    main()
