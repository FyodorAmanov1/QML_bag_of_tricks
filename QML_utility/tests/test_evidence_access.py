"""Published evidence can support use without inventing a local paper rerun."""
import json
from pathlib import Path
import pytest
from QML_utility.evidence_access import implementation_access
from QML_utility.select_case import select
from QML_utility.plan_project import build_plan
from QML_utility.setup_project import setup

ROOT = Path(__file__).resolve().parents[1]


def cases():
    return json.loads((ROOT/'catalog.json').read_text())['cases']


def profile(case_id):
    case = next(row for row in cases() if row['id']==case_id)
    return dict(data_kind=case['data_kind'], objective=case['objective'],
                capabilities=dict.fromkeys(case['required_capabilities'], True))


def test_access_inventory_keeps_author_code_distinct_from_local_execution():
    inspected=[]
    for case in cases():
        access=implementation_access(case)
        if access['public_author_source_verified']:
            inspected.append(case['id'])
            assert len(access['commit'])==40
            assert access['entrypoint'].startswith(access['repository']+'/blob/'+access['commit'])
            assert 'Not executed here' in access['scope']
        assert not case['locally_reproduced_paper_win']
    assert set(inspected)=={'continual_plasticity','quantum_graphons','quantum_self_attention','photonic_interference_kernel','volatility_reservoir'}


def test_author_source_filter_does_not_invent_missing_code_or_reuse_permission():
    source=profile('continual_plasticity');source['require_author_code']=True
    result=select(source)
    assert result['status']=='published_candidates'
    match=result['matches'][0]
    assert match['publication']['peer_reviewed'] and not match['independent_replication_required_for_use']
    assert not match['locally_reproduced_paper_win'] and not match['guaranteed_better']
    missing=profile('molecular_reservoir');missing['require_author_code']=True
    assert not select(missing)['matches']
    graph=select(dict(profile('quantum_graphons'),require_author_code=True))['matches'][0]
    assert graph['implementation_access']['license_recorded'] is None
    source['require_author_code']='yes'
    with pytest.raises(ValueError,match='boolean'):
        select(source)


def test_adaptation_is_default_and_replication_is_an_explicit_separate_track():
    source=profile('continual_plasticity')
    plan=build_plan(source)
    assert plan['intent']=='adapt' and not plan['independent_paper_replication_required']
    assert plan['case_plans'][0]['implementation_access']['public_author_source_verified']
    assert plan['case_plans'][0]['blockers'] and not plan['executed']
    source['intent']='replicate'
    replay=build_plan(source)
    assert replay['independent_paper_replication_required']
    assert replay['case_plans'][0]['independent_paper_replication_required']


def test_case_setup_exports_access_without_inventing_task_capabilities(tmp_path):
    record=setup(tmp_path/'setup',case_id='continual_plasticity')
    assert record['project']['intent']=='adapt'
    assert not any(record['project']['capabilities'].values())
    assert len(record['published_cases'])==10
    assert all('implementation_access' in row for row in record['published_cases'])
    assert all(not row['independent_replication_required_for_use'] for row in record['published_cases'])
