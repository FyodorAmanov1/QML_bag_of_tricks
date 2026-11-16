"""Evidence matching from declared capabilities; never predict advantage."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def select(profile):
    if not isinstance(profile, dict):
        raise ValueError('Profile must be a JSON object')
    if profile.get('data_kind') not in ('classical', 'quantum'):
        raise ValueError('data_kind must be classical or quantum')
    capabilities = profile.get('capabilities', {})
    if not isinstance(capabilities, dict) or any(type(v) is not bool for v in capabilities.values()):
        raise ValueError('capabilities must map names to explicit booleans')
    matches, rejected = [], []
    for case in json.loads((ROOT / 'catalog.json').read_text())['cases']:
        if case['data_kind'] != profile['data_kind']:
            continue
        missing = [name for name in case['required_capabilities'] if capabilities.get(name) is not True]
        if profile.get('objective') != case['objective']:
            missing.append('matching_objective:' + case['objective'])
        item = dict(id=case['id'], guide=case['guide'], missing_conditions=missing,
                    assessment_scope='protocol_requirements_not_method_wide_usefulness')
        if missing:
            rejected.append(item)
        else:
            matches.append(dict(item, status='evidence_matched_replication_candidate',
                                protocol=case.get('protocol'), evidence_type=case['evidence_type'],
                                blockers=case['blockers'], necessary_controls=case['necessary_controls'],
                                guaranteed_better=False, locally_reproduced_paper_win=False))
    ledger = json.loads((ROOT/'experiment_outcomes.json').read_text())
    outcomes = [episode for episode in ledger['episodes'] if episode['dataset'] == profile.get('dataset')]
    return dict(status='replication_candidates' if matches else 'no_evidence_matched_recommendation',
                matches=matches, rejected=rejected,
                interpretation='A missing match is not evidence that a method cannot help. Changes of training or data structure require a new prespecified evaluation.',
                contextual_outcomes=outcomes,
                followup_protocol='protocols/contextual_followup.json',
                next_step='Read the source and protocol; validate before deployment. No generic VQC fallback.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(select(json.loads(args.profile.read_text())), indent=2))


if __name__ == '__main__':
    main()
