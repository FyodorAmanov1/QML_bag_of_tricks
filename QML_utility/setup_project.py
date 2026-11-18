"""Start one project with published cases, core training tools and an executable contract."""
import argparse
import copy
import json
from pathlib import Path
from .make_example import create_example
from .recommend import recommend
from .strategies import essentials,check
from .evidence_access import implementation_access


def setup(output,profile=None,framework='qiskit',optimizer='qng',check_essentials=False,profile_base=Path('.'),case_id=None):
    cases = json.loads((Path(__file__).parent/'catalog.json').read_text())['cases']
    cases = [dict(case, implementation_access=implementation_access(case),
                  independent_replication_required_for_use=False) for case in cases]
    selected = next((case for case in cases if case['id']==case_id),None)
    if case_id and selected is None:
        raise ValueError('Unknown published case')
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a fresh setup directory')
    output.mkdir(parents=True)
    if profile is None and selected:
        config = dict(data_kind=selected['data_kind'],objective=selected['objective'],
                      capabilities=dict.fromkeys(selected['required_capabilities'],False),
                      intent='adapt',selected_case=selected['id'])
    elif profile is None:
        config = create_example(output/'project',framework=framework)
        config['data']['path'] = str((output/'project/data.csv').resolve())
        config['quantum']['steps'] = 3
    else:
        config = copy.deepcopy(profile)
        if config.get('data',{}).get('path'):
            source = Path(config['data']['path'])
            config['data']['path'] = str((Path(profile_base)/source).resolve()) if not source.is_absolute() else str(source)
    config['framework'] = framework
    config.setdefault('quantum',{}).update(optimizer=optimizer,metric_regularization=.001)
    portfolio = recommend(config)
    records = essentials()
    manifest = dict(entry_point='QML_utility',canonical_library='qml_tricks',same_distribution=True,
                     project=config,portfolio=portfolio,training_essentials=records,published_cases=cases,selected_case=selected,
                     publication_scope='Published case success is specific to data, access, comparator and metric; no new-project guarantee',
                     checks={})
    if check_essentials:
        for record in records:
            result = check(record['id'],framework,compare=True)
            path = output/(record['id']+'.json')
            path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            manifest['checks'][record['id']] = dict(status='executed',artifact=path.name,scope=result['scope'])
    (output/'project.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'setup.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    if selected and selected.get('protocol'):
        (output/'case_protocol.json').write_bytes((Path(__file__).parent/selected['protocol']).read_bytes())
    lines = ['# Unified QML project setup','',
             'Published-success candidates and training essentials belong to different evidence roles.','',
             '## Published starting points','',
             '| Case | Task | Evidence |','|---|---|---|']
    lines += [f"| {row['id']} | {row['objective']} | {row['evidence_type']} |" for row in cases]
    lines += ['', 'Read the corresponding evidence card and primary source. Selection does not assert that your project has the required conditions.',
              '', '## Published cases matching declared conditions','']
    matches = portfolio['published_case_matches']
    lines += [f"- {row['id']}: {row['guide']}" for row in matches] or ['No exact case match yet. See the full publication catalog; this does not establish method-wide uselessness.']
    lines += ['', '## Essential training methods','']+[f"- {row['title']} (`{row['id']}`): {row['project_scope']}" for row in records]
    lines += ['', '## Next step','']
    if portfolio['executable_route']:
        lines += ['```bash',
              f'python -m QML_utility run --config "{output / "project.json"}" --output "{output / "comparison"}"',
              '```']
    else:
        lines += ['Resolve the case data/access conditions and follow its original protocol. No generic CSV model replaces this task.']
    lines += ['', 'Review the profile/data contract before running. This setup does not claim a predictive gain.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--case',dest='case_id',help='Start from a published-case protocol; capabilities initially remain unconfirmed')
    parser.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit')
    parser.add_argument('--optimizer',choices=['qng','adam','gd'],default='qng')
    parser.add_argument('--check-essentials',action='store_true')
    args = parser.parse_args(argv)
    result = setup(args.output,json.loads(args.profile.read_text()) if args.profile else None,args.framework,
                   args.optimizer,args.check_essentials,args.profile.resolve().parent if args.profile else Path('.'),args.case_id)
    print(json.dumps(dict(setup=str(args.output/'README.md'),essential_methods=len(result['training_essentials']),
                          matching_published_cases=[row['id'] for row in result['portfolio']['published_case_matches']],
                          essential_checks=len(result['checks'])),indent=2))
