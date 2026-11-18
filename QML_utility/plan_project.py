"""Generate an agent experiment contract; never train or claim a model win."""
import argparse
import json
from pathlib import Path
from .select_case import select, ROOT


def build_plan(profile):
    selection = select(profile)
    intent = profile.get('intent', 'adapt')
    if intent not in ('adapt', 'replicate', 'follow_up'):
        raise ValueError('intent must be adapt, replicate or follow_up')
    project = profile.get('project', {})
    if not isinstance(project, dict):
        raise ValueError('project must be a JSON object')
    required = ['data_provenance', 'target', 'primary_metric', 'split_strategy', 'compute_budget']
    missing = [key for key in required if not isinstance(project.get(key), str) or not project[key].strip()]
    cases = []
    for match in selection['matches']:
        protocol = match['protocol']
        cases.append(dict(case_id=match['id'], guide=match['guide'],
                          publication=match['publication'], implementation_access=match['implementation_access'],
                          independent_paper_replication_required=intent=='replicate',
                          replication_protocol=json.loads((ROOT/protocol).read_text()) if protocol else None,
                          blockers=match['blockers'], controls=match['necessary_controls']))
    followup = json.loads((ROOT/'protocols/contextual_followup.json').read_text()) if intent == 'follow_up' else None
    ready = bool(cases or followup) and not missing
    return dict(schema_version=1, status='draft_ready_for_agent_review' if ready else 'draft_requires_project_details_or_matched_evidence',
                intent=intent, project=project, missing_project_details=missing, selection=selection,
                evidence_basis='Peer-reviewed publication; author-source availability and local execution are separate fields.',
                independent_paper_replication_required=intent=='replicate',
                case_plans=cases, prospective_followup=followup,
                stage_order=['verify source/artifacts', 'freeze split and budget policy',
                             'diagnose fit/representation', 'validation-only selection',
                             'untouched-test evaluation', 'independent artifact audit'],
                required_artifacts=['environment and source hashes','split IDs and preprocessing fit scope',
                                    'all candidate settings, curves, seeds and warnings','held-out predictions or channel estimates',
                                    'paired uncertainty at the independent data unit','loading/readout/shots/time/memory costs'],
                no_gain_statement='No advantage was observed for the stated configuration, data and budget; modified training and other data structures remain untested.',
                positive_statement_rule='Name the exact data/access/comparator/metric tuple. A gain is not universal superiority.',
                executed=False, guaranteed_better=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = build_plan(json.loads(args.profile.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status=plan['status'], cases=[x['case_id'] for x in plan['case_plans']],
                          missing_project_details=plan['missing_project_details'], executed=False),indent=2))


if __name__ == '__main__':
    main()
