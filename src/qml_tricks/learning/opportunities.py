"""Application-specific QML comparisons with controls that can close the gap.

Run: python -m qml_tricks.learning.opportunities
See docs/applications/ for the data-access and claim boundaries.
"""
import argparse
from copy import deepcopy
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_info, threadpool_limits

from .experimental_learning import (acquire_bell, acquire_local, bell_distribution,
    bell_signs, local_counts, local_features, local_log_likelihood_ratio, pauli_state)

PROTOCOL = Path(__file__).with_name('opportunity_protocol.json')
REGULARIZATION = (1e-4, 1e-3, .01, .1, 1., 10., 100., 1000., 10000.)


def fourier_features(x, frequencies=(1, 2, 3)):
    phase = np.asarray(x).reshape(-1)
    return np.column_stack([f for k in frequencies for f in (np.cos(k*phase), np.sin(k*phase))])


def ramsey_signal(phase):
    """Fixed harmonic calibration response, specified independently of model fits."""
    p = np.asarray(phase)
    return .7*np.cos(p) + .35*np.sin(2*p) - .2*np.cos(3*p) + .15


def candidate(name, trial, seed, task):
    """Nine candidates per learned family, including actual regularization tuning."""
    grid3 = (.001, .1, 10.)
    if name.endswith('ridge') or name == 'ridge':
        params = dict(alpha=REGULARIZATION[trial]); model = Ridge(**params)
    elif name == 'elastic_net':
        params = dict(alpha=(.0001, .001, .01)[trial//3], l1_ratio=(.1, .5, .9)[trial%3], max_iter=5000)
        model = ElasticNet(**params)
    elif name == 'rbf_svr':
        params = dict(C=(.1, 1., 10.)[trial//3], gamma=(.1, 1., 10.)[trial%3], epsilon=.03)
        model = SVR(**params)
    elif name == 'mlp_l2' or name == 'local_mlp_l2':
        params = dict(hidden_layer_sizes=((8,), (32,), (64,))[trial//3],
                      alpha=(.0001, .01, 1.)[trial%3], solver='lbfgs', max_iter=400, random_state=seed)
        model = (MLPRegressor if task == 'regression' else MLPClassifier)(**params)
    elif name == 'boosting':
        params = dict(max_depth=(1, 2, 3)[trial//3], learning_rate=(.03, .1, .3)[trial%3],
                      n_estimators=100, random_state=seed)
        model = GradientBoostingRegressor(**params)
    elif name.endswith('logistic_l2'):
        params = dict(C=REGULARIZATION[trial], max_iter=1000)
        model = LogisticRegression(**params)
    elif name == 'local_rbf_svc':
        params = dict(C=grid3[trial//3], gamma=grid3[trial%3])
        model = SVC(**params)
    else:
        raise ValueError(name)
    pipeline = make_pipeline(StandardScaler(), model)
    if task == 'regression':
        pipeline = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
    return pipeline, params


def metrics(task, y, pred):
    if task == 'classification':
        return dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)))
    return dict(rmse=float(np.sqrt(mean_squared_error(y, pred))), mae=float(mean_absolute_error(y, pred)))


def learn(name, features, y, train, validation, test, seed, task):
    history = []; started = time.perf_counter()
    primary = 'rmse' if task == 'regression' else 'balanced_accuracy'
    for trial in range(9):
        model, params = candidate(name, trial, seed, task)
        clock = time.perf_counter()
        row = dict(trial=trial, parameters=params)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(features[train], y[train])
                pred = model.predict(features[validation])
            row.update(validation=metrics(task, y[validation], pred),
                       warnings=[str(w.message) for w in caught])
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        row['seconds'] = time.perf_counter()-clock
        history.append(row)
    valid = [r for r in history if 'validation' in r]
    result = dict(model=name, trials=history)
    if not valid:
        return dict(result, error='All candidates failed')
    selected = (min if task == 'regression' else max)(valid, key=lambda r:r['validation'][primary])
    refit = np.r_[train, validation]
    model, _ = candidate(name, selected['trial'], seed, task)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(features[refit], y[refit])
            clock = time.perf_counter(); pred = model.predict(features[test]); inference = time.perf_counter()-clock
        pipeline = model.regressor_ if task == 'regression' else model
        result.update(selected_trial=selected['trial'], test=metrics(task, y[test], pred),
                      test_predictions=np.asarray(pred).tolist(), test_targets=y[test].tolist(),
                      refit_feature_mean=pipeline[0].mean_.tolist(),
                      warnings=[str(w.message) for w in caught], inference_seconds=inference)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['selection_refit_seconds'] = time.perf_counter()-started
    return result


def indices(train_size, validation_size, test_size):
    train = np.arange(train_size)
    validation = np.arange(train_size, train_size+validation_size)
    test = np.arange(train_size+validation_size, train_size+validation_size+test_size)
    return train, validation, test


def classical_champion(rows, task):
    """Pool conventional learned families using validation scores ONLY."""
    excluded = {'fourier_ridge', 'quantum_ramsey_ridge', 'bell_logistic_l2',
                'local_bayes', 'query_known_bayes'}
    choices = [r for r in rows if r['model'] not in excluded and 'test' in r]
    if not choices:
        return None
    primary = 'rmse' if task == 'regression' else 'balanced_accuracy'
    def value(row):
        return next(t['validation'][primary] for t in row['trials'] if t['trial'] == row['selected_trial'])
    winner = (min if task == 'regression' else max)(choices, key=value)
    return dict(model='classical_selected', selected_family=winner['model'],
                selected_validation_score=float(value(winner)),
                pooled_candidate_count=sum(len(r['trials']) for r in choices),
                test=deepcopy(winner['test']), test_predictions=winner['test_predictions'],
                test_targets=winner['test_targets'], trials=[])


def ramsey_case(config, seed, train_size, shots, framework, models=None):
    rng = np.random.default_rng(seed)
    train, val, test = indices(train_size, config['validation_size'], config['test_size'])
    phase = np.r_[rng.uniform(*config['train_validation_phase_range'], len(train)+len(val)),
                  rng.uniform(*config['test_phase_range'], len(test))]
    y = ramsey_signal(phase)+rng.normal(0, config['target_noise_sd'], len(phase))
    fourier = fourier_features(phase, config['frequencies'])
    native = import_module(f'qml_tricks.learning.application_backends.{framework}')
    clock = time.perf_counter()
    quantum = native.ramsey_features(phase, config['frequencies'])
    mismatch = float(np.max(np.abs(fourier-quantum)))
    if shots is not None:
        acquisition_rng = np.random.default_rng(np.random.SeedSequence([seed, train_size, shots, 700]))
        quantum = 2*acquisition_rng.binomial(shots, np.clip((1+quantum)/2, 0, 1))/shots-1
    acquired = time.perf_counter()-clock
    rows = []
    for name in models or config['models']:
        features = quantum if name == 'quantum_ramsey_ridge' else fourier if name == 'fourier_ridge' else phase[:, None]
        row = learn(name, features, y, train, val, test, seed, 'regression')
        row['feature_dimension'] = features.shape[1]; rows.append(row)
    champion = classical_champion(rows, 'regression')
    if champion is not None:
        rows.append(champion)
    settings = 2*len(config['frequencies'])
    return dict(application='ramsey_calibration', seed=seed, train_size=train_size, shots=shots,
        split_indices=dict(train=train.tolist(), validation=val.tolist(), test=test.tolist()),
        phase=phase.tolist(), targets=y.tolist(), models=rows,
        native_fourier_max_error=mismatch,
        acquisition=dict(total_rows=len(phase), measurement_settings_per_row=settings,
            simulated_statevector_executions=len(phase)*len(config['frequencies']),
            hypothetical_hardware_state_copies=0 if shots is None else len(phase)*settings*shots,
            training_validation_state_copies=0 if shots is None else (len(train)+len(val))*settings*shots,
            test_state_copies=0 if shots is None else len(test)*settings*shots,
            seconds=acquired, qubits=1, statevector_bytes=32,
            meaning='Native statevector features; separate noncommuting X/Y settings; host binomial sampling. Feature acquisition cached across readout candidates, including held-out inputs without labels.'),
        interpretation='Synthetic periodic sensor response, held-out periods. Exact quantum features have a cheap identical classical Fourier implementation.')


def pauli_case(config, seed, n, copies, noise, models=None):
    train, val, test = indices(config['train_size'], config['validation_size'], config['test_size'])
    state_rng = np.random.default_rng(np.random.SeedSequence([seed, n, 400]))
    labels = []
    for group in (train, val, test):
        if len(group) % 2:
            raise ValueError('Balanced task partitions must have even sizes')
        part = np.tile([0, 1], len(group)//2); state_rng.shuffle(part); labels.extend(part.tolist())
    y = np.array(labels)
    words = [''.join(row) for row in state_rng.choice(list('XYZ'), (len(y), n))]
    amplitude = y*config['nonzero_magnitude']*state_rng.choice([-1, 1], len(y))
    rng_bell, rng_local, rng_known = [np.random.default_rng(np.random.SeedSequence(
        [seed, n, copies, int(noise*1e6), stream])) for stream in (1, 2, 3)]
    local = []; bell = []; known = []; hashes = []; started = time.perf_counter()
    effective = config['nonzero_magnitude']*(1-2*noise)**n
    for word, a in zip(words, amplitude):
        # The record is acquired without a query argument. Query-dependent
        # statistics are computed only afterwards; hidden a never enters fit.
        bits = acquire_bell(word, a, copies, rng_bell, noise)
        bases, outcomes = acquire_local(word, a, copies, rng_local, noise)
        q = float(bell_signs(bits, word).mean()); bell.append([q])
        plus, minus = local_counts(bases, outcomes, word)
        local.append(local_features(plus, minus, n, copies))
        p = rng_known.binomial(copies, (1+a*(1-2*noise)**n)/2)
        known.append(int(local_log_likelihood_ratio(p, copies-p, effective) > 0))
        digest = hashlib.sha256(bits.tobytes()+bases.tobytes()+outcomes.tobytes()).hexdigest()
        hashes.append(digest)
    acquired = time.perf_counter()-started
    local, bell = np.asarray(local), np.asarray(bell)
    optimal = (local_log_likelihood_ratio(local[:, 0], local[:, 1], effective) > 0).astype(int)
    rows = []
    for name in models or config['models']:
        if name in ('local_bayes', 'query_known_bayes'):
            pred = optimal[test] if name == 'local_bayes' else np.array(known)[test]
            row = dict(model=name, test=metrics('classification', y[test], pred),
                       test_predictions=pred.tolist(), test_targets=y[test].tolist(), trials=[],
                       meaning='Known-family optimal equal-prior likelihood ratio; no fitted model. Query-known control changes the acquisition information order.' if name == 'query_known_bayes' else 'Known-family optimal likelihood ratio for the chosen random local measurement record; includes a stronger control than regularization alone.')
        else:
            row = learn(name, bell if name == 'bell_logistic_l2' else local,
                        y, train, val, test, seed, 'classification')
        rows.append(row)
    champion = classical_champion(rows, 'classification')
    if champion is not None:
        rows.append(champion)
    return dict(application='late_pauli_query', seed=seed, qubits=n, state_copies=copies,
        readout_bit_flip=noise, split_indices=dict(train=train.tolist(), validation=val.tolist(), test=test.tolist()),
        models=rows, labels=y.tolist(), queries=words, local_features=local.tolist(), bell_features=bell.tolist(),
        measurement_record_sha256=hashes,
        acquisition=dict(rows=len(y), copies_per_state_per_protocol=copies,
            total_copies_per_protocol=len(y)*copies, bell_pairs_per_state=copies//2,
            max_simultaneous_qubits_bell=2*n, max_simultaneous_qubits_local=n,
            bell_cnot_gates_per_state=n*(copies//2), bell_hadamard_gates_per_state=n*(copies//2),
            local_expected_matching_records=copies/3.**n,
            local_record_bytes_per_state=2*copies*n, bell_record_bytes_per_state=copies*n,
            seconds=acquired,
            meaning='Exact special-family Born-law host sampling, NOT native simulation or hardware shots. Each Bell round consumes two independent rho copies. Native density-matrix circuits validate the law separately. uint8/int8 record storage; state-preparation/SDK overhead excluded.'),
        interpretation='Synthetic quantum-state property classification with late queries and matched physical copy budgets. Local baselines use random single-copy Pauli acquisition; not a benchmark of every single-copy POVM. Query-known control has more advance information.')


def summarize(cases):
    groups = {}
    for case in cases:
        if case['application'] == 'ramsey_calibration':
            key = ('ramsey_calibration', case['train_size'], case['shots']); metric = 'rmse'
        else:
            key = ('late_pauli_query', case['qubits'], case['state_copies'], case['readout_bit_flip']); metric = 'balanced_accuracy'
        for row in case['models']:
            groups.setdefault((key, row['model'], metric), []).append(row)
    rows = []
    for (key, name, metric), runs in groups.items():
        scores = np.array([r['test'][metric] for r in runs if 'test' in r])
        rows.append(dict(configuration=list(key), model=name, metric=metric,
            mean=float(scores.mean()) if len(scores) else None,
            seed_sd=float(scores.std(ddof=1)) if len(scores)>1 else 0.,
            successes=len(scores), failures=len(runs)-len(scores)))
    paired = []
    for case in cases:
        successful = {r['model']:r for r in case['models'] if 'test' in r}
        qname = 'quantum_ramsey_ridge' if case['application'] == 'ramsey_calibration' else 'bell_logistic_l2'
        if qname not in successful:
            continue
        if case['application'] == 'ramsey_calibration':
            key = [case['application'], case['train_size'], case['shots']]; metric = 'rmse'
        else:
            key = [case['application'], case['qubits'], case['state_copies'], case['readout_bit_flip']]; metric = 'balanced_accuracy'
        for name, row in successful.items():
            if name == qname:
                continue
            qscore = successful[qname]['test'][metric]; score = row['test'][metric]
            gain = score-qscore if metric == 'rmse' else qscore-score
            paired.append(dict(configuration=key, control=name, seed=case['seed'], gain=float(gain)))
    comparisons = []
    for key in dict.fromkeys((tuple(p['configuration']), p['control']) for p in paired):
        values = np.array([p['gain'] for p in paired if (tuple(p['configuration']), p['control']) == key])
        rng = np.random.default_rng(77)
        interval = np.quantile(rng.choice(values, (2000, len(values))).mean(axis=1), [.025, .975])
        comparisons.append(dict(configuration=list(key[0]), control=key[1], mean_gain=float(values.mean()),
            positive_seeds=int(np.sum(values > 1e-10)), seed_count=len(values),
            descriptive_paired_bootstrap_95=interval.tolist(),
            observed_gain=bool(len(values)>=10 and interval[0]>0 and np.sum(values>1e-10)>=8)))
    return rows, comparisons


def run_opportunities(output='results/opportunities.json', framework='qiskit', protocol=None,
                      applications=('ramsey', 'pauli')):
    config = deepcopy(protocol) if protocol is not None else json.loads(PROTOCOL.read_text(encoding='utf-8'))
    cases = []; clock = time.perf_counter()
    # Independent circuit verification is recorded, not passed off as executing
    # all large-system synthetic measurements on a quantum SDK.
    native = import_module(f'qml_tricks.learning.application_backends.{framework}')
    word = 'XYZ'; a = .63
    verification = dict(framework=framework, word=word, amplitude=a,
        bell_probability_max_error=float(np.max(np.abs(native.bell_probabilities(pauli_state(word, a))-bell_distribution(word, a)))))
    if verification['bell_probability_max_error'] > 1e-8:
        raise ValueError('Native Bell verification failed')
    for seed in config['seeds']:
        if 'ramsey' in applications:
            r = config['ramsey']
            for size in r['train_sizes']:
                for shots in r['shots_per_setting']:
                    cases.append(ramsey_case(r, seed, size, shots, framework))
        if 'pauli' in applications:
            p = config['pauli']
            for n in p['qubits']:
                for copies in p['state_copies']:
                    for noise in p['readout_bit_flip']:
                        cases.append(pauli_case(p, seed, n, copies, noise))
        print(f'Completed application seed {seed}: {len(cases)} configurations', flush=True)
    summary, paired = summarize(cases)
    source = Path(__file__).resolve().parents[1]
    result = dict(protocol=config, selected_applications=list(applications), native_verification=verification,
        cases=cases, summary=summary, paired_comparisons=paired, elapsed_seconds=time.perf_counter()-clock,
        platform=platform.platform(), threadpools=threadpool_info(),
        versions={p:version(p) for p in ('numpy', 'scipy', 'scikit-learn', {'cirq':'cirq-core'}.get(framework,framework))},
        source_hashes={str(p.relative_to(source)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in source.rglob('*.py')},
        protocol_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        limits='No hardware runs or computational speedup. Periodic task has an identical cheap classical representation. Quantum-state task compares acquisition protocols, not the power of regularization; general POVM lower bounds belong to the cited paper, not this finite simulation. All cells are retained; intervals are descriptive, uncorrected for multiple comparisons.')
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/opportunities.json')
    parser.add_argument('--framework', choices=['qiskit', 'pennylane', 'cirq'], default='qiskit')
    parser.add_argument('--application', choices=['all', 'ramsey', 'pauli'], default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_opportunities(args.output, args.framework,
            applications=('ramsey', 'pauli') if args.application=='all' else (args.application,))


if __name__ == '__main__':
    main()
