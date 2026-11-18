"""Evidence matching from declared capabilities; never predict advantage."""
import argparse
import json
from pathlib import Path
from .evidence_access import implementation_access

ROOT = Path(__file__).resolve().parent


def select(profile):
    if not isinstance(profile, dict):
        raise ValueError('Profile must be a JSON object')
    if profile.get('data_kind') not in ('classical', 'quantum'):
        raise ValueError('data_kind must be classical or quantum')
    capabilities = profile.get('capabilities', {})
    require_code = profile.get('require_author_code', False)
    if type(require_code) is not bool:
        raise ValueError('require_author_code must be an explicit boolean')
    if not isinstance(capabilities, dict) or any(type(v) is not bool for v in capabilities.values()):
        raise ValueError('capabilities must map names to explicit booleans')
    matches, rejected = [], []
    for case in json.loads((ROOT / 'catalog.json').read_text())['cases']:
        if case['data_kind'] != profile['data_kind']:
            continue
        missing = [name for name in case['required_capabilities'] if capabilities.get(name) is not True]
        access = implementation_access(case)
        if require_code and not access['public_author_source_verified']:
            missing.append('verified_public_author_source')
        if profile.get('objective') != case['objective']:
            missing.append('matching_objective:' + case['objective'])
        item = dict(id=case['id'], guide=case['guide'], missing_conditions=missing,
                    assessment_scope='protocol_requirements_not_method_wide_usefulness')
        if missing:
            rejected.append(item)
        else:
            matches.append(dict(item, status='evidence_matched_published_candidate',
                                protocol=case.get('protocol'), evidence_type=case['evidence_type'],
                                publication=case['publication'], implementation_access=access,
                                independent_replication_required_for_use=False,
                                blockers=case['blockers'], necessary_controls=case['necessary_controls'],
                                guaranteed_better=False, locally_reproduced_paper_win=False))
    ledger = json.loads((ROOT/'experiment_outcomes.json').read_text())
    outcomes = [episode for episode in ledger['episodes'] if episode['dataset'] == profile.get('dataset')]
    return dict(status='published_candidates' if matches else 'no_evidence_matched_recommendation',
                matches=matches, rejected=rejected,
                interpretation='A missing match is not evidence that a method cannot help. Changes of training or data structure require a new prespecified evaluation.',
                contextual_outcomes=outcomes,
                followup_protocol='protocols/contextual_followup.json',
                next_step='Use or adapt the published implementation under its reuse terms; evaluate the target project before deployment. Independent paper replication is optional. No generic VQC fallback.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(select(json.loads(args.profile.read_text())), indent=2))


if __name__ == '__main__':
    main()
