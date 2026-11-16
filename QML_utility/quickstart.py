"""Create and execute a complete local example without downloading datasets."""
import argparse
import csv
import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer,load_diabetes
from .make_example import create_example
from .recommend import recommend
from .run_project import run
from .audit_project import audit


def create_batched_example(output,task,framework):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    source = load_breast_cancer() if task=='classification' else load_diabetes()
    names = ['x'+str(i) for i in range(source.data.shape[1])]
    with (output/'data.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle);writer.writerow(['row_id']+names+['target'])
        writer.writerows([i]+row.tolist()+[source.target[i]] for i,row in enumerate(source.data))
    config = dict(data_kind='classical',modality='numeric_table',task=task,objective='iid_'+task,
                  capabilities={},allow_exploratory=True,framework=framework,seed=7,adapter='batched_features',
                  methods=['quantum_features'],constraints=dict(n_samples=len(source.data)),
                  quantum=dict(qubits=2,layers=1,batch_size=64,head_epochs=10),
                  budget=dict(max_rows=20000,state_preparations_per_trial=100000,seconds_per_operation=300,max_feature_bytes=64*1024*1024),
                  data=dict(path='data.csv',features=names,target='target',id_column='row_id',split=dict(kind='iid')),
                  provenance=dict(source='sklearn bundled breast cancer' if task=='classification' else 'sklearn bundled diabetes',
                                  scope='All original records/features; numeric integration example, not medical validation or advantage evidence'))
    return config


def quickstart(output,framework='qiskit',task='classification',preset='quick'):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh quickstart directory')
    output.mkdir(parents=True)
    project = output/'project'
    if preset=='scalable':
        config = create_batched_example(project,task,framework)
    else:
        config = create_example(project,task,framework)
        if preset=='quick':
            config['methods'] = ['quantum_features','fidelity_kernel']
    (project/'project.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'portfolio.json').write_text(json.dumps(recommend(config),indent=2)+'\n')
    report = run(config,output/'run',project.resolve())
    checked = audit(output/'run',config,project.resolve())
    (output/'audit.json').write_text(json.dumps(checked,indent=2)+'\n')
    return dict(status=report['status'],report=str(output/'run/REPORT.md'),config=str(project/'project.json'),audit=checked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    parser.add_argument('--task',choices=['classification','regression'],default='classification')
    parser.add_argument('--preset',choices=['quick','full','scalable'],default='quick')
    args = parser.parse_args(argv)
    print(json.dumps(quickstart(args.output,args.framework,args.task,args.preset),indent=2))


if __name__ == '__main__':
    main()
