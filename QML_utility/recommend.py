"""Propose an evidence-aware portfolio and implementation route, never a guaranteed winner."""
import argparse
import json
import math
from pathlib import Path
from .select_case import select, ROOT


def recommend(profile):
    if not isinstance(profile,dict):
        raise ValueError('Project profile must be a JSON object')
    evidence = select(profile)
    numeric = profile.get('data_kind') == 'classical' and profile.get('modality') == 'numeric_table' and profile.get('task') in ('classification','regression')
    adapter = profile.get('adapter','reference')
    if adapter not in ('reference','batched_features'):
        raise ValueError('adapter must be reference or batched_features')
    row_limit = 20000 if adapter=='batched_features' else 512
    preferences,budget,constraints = (profile.get(key,{}) for key in ('preferences','budget','constraints'))
    if any(not isinstance(value,dict) for value in (preferences,budget,constraints)):
        raise ValueError('preferences, budget and constraints must be objects')
    for key in ('train_circuit','classical_only_inference'):
        if key in preferences and type(preferences[key]) is not bool:
            raise ValueError('preferences.'+key+' must be an explicit boolean')
    n = constraints.get('n_samples')
    if n is not None and (type(n) is not int or n < 1):
        raise ValueError('constraints.n_samples must be a positive integer')
    memory = budget.get('max_gram_bytes')
    if memory is not None and (type(memory) is not int or memory < 1):
        raise ValueError('budget.max_gram_bytes must be a positive integer')
    quantum = profile.get('quantum',{})
    if not isinstance(quantum,dict):
        raise ValueError('quantum must be an object')
    q,l,steps = (quantum.get(key,default) for key,default in (('qubits',2),('layers',1),('steps',12)))
    if any(type(value) is not int or value < 1 for value in (q,l,steps)):
        raise ValueError('Quantum size and step counts must be positive integers')
    data = profile.get('data',{})
    if not isinstance(data,dict) or not isinstance(data.get('features',[]),list):
        raise ValueError('data must be an object and features a list')
    d = len(data.get('features',[]))
    train_rows = math.ceil(.6*n) if n else None
    gram_bytes = 8*train_rows**2 if n else None
    candidates,withheld = [],[]
    if numeric:
        library = json.loads((ROOT/'method_library.json').read_text())
        for method in library['methods']:
            reasons = []
            if adapter=='batched_features' and method['id']!='quantum_features':
                reasons.append('Batched adapter supports fixed features; use reference for this family')
            if method['id'] == 'trainable_local' and preferences.get('train_circuit') is False:
                reasons.append('Project excludes circuit-gradient training')
            if method['id'] in ('fidelity_kernel','projected_kernel') and gram_bytes and memory and gram_bytes > memory:
                reasons.append('Conservative dense training-Gram plan exceeds max_gram_bytes; sparse/lazy alternatives need a separate adapter')
            estimated_states = (train_rows*(steps*(1+4*q*l)+1)+(n-train_rows)) if n and method['id']=='trainable_local' else n
            if n and method['id']=='trainable_local' and quantum.get('optimizer')=='qng':
                estimated_states += train_rows*steps*(1+4*q*l)
            cap = budget.get('state_preparations_per_trial')
            if cap is not None and (type(cap) is not int or cap < 1):
                raise ValueError('state_preparations_per_trial must be a positive integer')
            if estimated_states and cap and estimated_states > cap:
                reasons.append('Planned preparation count exceeds per-trial cap')
            if (n and n > row_limit) or d > 64 or q > 6 or l > 4 or steps > 200:
                reasons.append('Native reference size limit exceeded')
            if reasons:
                withheld.append(dict(id=method['id'],reasons=reasons,scope='current_reference_plan_not_method_wide_impossibility'))
                continue
            candidates.append(dict(method,role='exploratory_transfer_candidate',
                                   implementation='models/scalable.py:BatchedQuantumFeatures' if adapter=='batched_features' else 'models/estimators.py:QuantumEstimator',
                                   frameworks=['qiskit','pennylane','cirq'],
                                   estimated_state_preparations_per_trial=estimated_states,
                                   estimated_dense_gram_bytes=gram_bytes if method['id'] in ('fidelity_kernel','projected_kernel') else 0,
                                   deployment_status='requires_offline_distillation_adapter' if preferences.get('classical_only_inference') else 'native_simulator_inference',
                                   verified_advantage_for_this_project=False))
    gaps = []
    if numeric:
        data = profile.get('data',{})
        for key in ('path','features','target','split'):
            if not data.get(key):
                gaps.append('data.'+key)
        if profile.get('allow_exploratory') is not True:
            gaps.append('allow_exploratory=true for the runnable engineering comparison')
    else:
        gaps.append('Project-specific adapter required; numeric-table estimator is not a substitute for native graph/text/continual/quantum acquisition protocols')
    checks = ['Check target and loading/access assumptions',
              'Use IID, entity/group or chronological splitting to match deployment',
              'Prespecify tuning policy and include regularized classical controls',
              'Keep representation, fitting and measurement costs separate']
    if numeric and ((isinstance(n,int) and n > row_limit) or d > 64):
        gaps.append(f'Adapter limits exceeded: {row_limit} records and 64 numeric features; design an explicit adapter, not automatic compression')
    from .strategies import essentials
    return dict(schema_version=1,status='portfolio_requires_validation',training_essentials=essentials(),
                published_case_matches=evidence['matches'],unmatched_case_conditions=evidence['rejected'],
                method_candidates=candidates,missing_project_details=gaps,
                withheld_candidates=withheld,
                planning_assumptions='Resource estimates use a 60% training fraction; group sizes and custom adapters can change them. No timing or useful-shot precision is inferred.',
                required_companions=['kipu_offline_surrogate'] if preferences.get('classical_only_inference') and candidates else [],
                offline_deployment_route='offline_surrogate.py:OfflineFeatureStudent' if preferences.get('classical_only_inference') and candidates else None,
                classical_controls=['dummy','regularized_linear','random_features','hist_boosting'] if adapter=='batched_features' else ['dummy','regularized_linear','rbf','boosting','same_angle_fourier'],
                executable_route='python -m QML_utility.run_project --config PROJECT.json --output NEW_RUN_DIRECTORY' if numeric and candidates else None,
                companion_map='companion_map.json',workflow='PRACTICAL_WORKFLOW.md',
                decision_checks=checks,
                fallback='A valid outcome is a classical model. No matched published case is not a proof of quantum uselessness.',
                interpretation='The portfolio proposes falsifiable transfer hypotheses. Its methods are not extra published-success records.',
                guaranteed_better=False,executed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    result = recommend(json.loads(args.profile.read_text()))
    payload = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload,end='')


if __name__ == '__main__':
    main()
