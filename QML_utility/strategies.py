"""Direct access to canonical main-library strategies from the utility entry point."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def essentials():
    from qml_tricks.core import catalog
    canonical = {row['id']:row for row in catalog()}
    records = json.loads((ROOT/'training_essentials.json').read_text())['strategies']
    return [dict(record,canonical=canonical[record['id']],
                 command=f"qml-utility strategy {record['id']} --framework qiskit --compare") for record in records]


def resolve(name):
    from qml_tricks.core import catalog
    aliases = {alias:row['id'] for row in essentials() for alias in row['aliases']}
    name = aliases.get(name,name)
    if name not in {row['id'] for row in catalog()}:
        raise ValueError('Unknown canonical strategy: '+name)
    return name


def check(name,framework='qiskit',compare=False,**options):
    from qml_tricks.core import RunConfig,experiment
    name = resolve(name)
    config = dict(n_qubits=2,depth=1,steps=1,initializations=2,repetitions=2,seed=7)
    config.update(options)
    variants = ['strategy','baseline'] if compare else ['strategy']
    return dict(strategy=name,scope='Canonical mechanism/optimizer experiment, not a task-level advantage reproduction',
                implementation=f'qml_tricks.strategies.{name}.{framework}_impl',
                results=[experiment(name,framework,RunConfig(**config,variant=variant)).to_dict() for variant in variants])


def main(argv=None):
    from qml_tricks.core import catalog
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all',action='store_true',help='List the entire canonical research library')
    args = parser.parse_args(argv)
    print(json.dumps(catalog() if args.all else essentials(),indent=2))


def run_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--n-qubits',type=int,default=2)
    parser.add_argument('--depth',type=int,default=1)
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--metric-regularization',type=float,default=.001)
    parser.add_argument('--encoding-scale',type=float,default=.1)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    result = check(args.name,args.framework,args.compare,n_qubits=args.n_qubits,depth=args.depth,
                   steps=args.steps,seed=args.seed,metric_regularization=args.metric_regularization,
                   encoding_scale=args.encoding_scale)
    payload = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(payload)
    else:
        print(payload,end='')
