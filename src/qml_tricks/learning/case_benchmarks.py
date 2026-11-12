"""Small, honest integration benchmarks for the new paper-inspired models."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qml_tricks.learning.case_models import (
    AmplitudeQVCClassifier, MulticlassQuantumKernel, TfidfQuantumMatcher,
    classical_product_kernel, projected_gradient_attack,
)

CASES = ("entity_matching", "multiclass_kernel", "adversarial_qvc")
DOIS = {
    "entity_matching": "10.1038/s41598-025-88177-z",
    "multiclass_kernel": "10.1103/PhysRevA.111.062410",
    "adversarial_qvc": "10.1103/PhysRevResearch.5.023186",
}


def _score(y, prediction):
    return {"accuracy":float(accuracy_score(y,prediction)), "predictions":np.asarray(prediction).tolist()}


def _choose(candidates, y_validation):
    # candidates contain a fitted estimator, its validation predictions and config.
    index = int(np.argmax([accuracy_score(y_validation,item[1]) for item in candidates]))
    estimator, _, config = candidates[index]
    return estimator, config, [float(accuracy_score(y_validation,item[1])) for item in candidates]


def entity_fixture(seed):
    """Authored template data, grouped by entity; not the Bischof dataset."""
    rng = np.random.default_rng(seed)
    groups = rng.permutation(24)
    partitions = (groups[:14],groups[14:19],groups[19:])
    output = []
    for group_ids in partitions:
        pairs, labels, ids = [], [], []
        for group in group_ids:
            name = f"device{group}"
            examples = [
                ((f"{name} red sensor",f"{name} red sensor"),1),
                ((f"{name} red sensor",f"sensor red {name}"),1),
                ((f"company {name} sells sensors",f"sensors are sold by company {name}"),1),
                ((f"{name} is active",f"{name} is not active"),0),
                ((f"{name} red sensor",f"{name} blue engine"),0),
            ]
            for k,(pair,label) in enumerate(examples):
                pairs.append(pair); labels.append(label); ids.append(f"{group}:{k}")
        output.append((pairs,np.asarray(labels),ids))
    return output


def _entity(framework, seed, steps):
    (train,y_train,train_ids),(val,y_val,val_ids),(test,y_test,test_ids) = entity_fixture(seed)
    model = TfidfQuantumMatcher(framework=framework,steps=steps,seed=seed).fit(train,y_train)
    training_evaluations = model.state_evaluations_
    quantum_predictions = model.predict(test)
    inference_evaluations = model.state_evaluations_-training_evaluations
    train_features, val_features, test_features = model.features(train),model.features(val),model.features(test)
    candidates = []
    for c in (.1,1.,10.):
        estimator = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500)).fit(train_features,y_train)
        candidates.append((estimator,estimator.predict(val_features),{"C":c}))
    classical, config, scores = _choose(candidates,y_val)
    return {
        "dataset":"authored sentence templates; held-out entity groups; not author data",
        "split_ids":{"train":train_ids,"validation":val_ids,"test":test_ids},
        "test_targets":y_test.tolist(),
        "models":{
            "tfidf_quantum_adaptation":_score(y_test,quantum_predictions),
            "tfidf_threshold":_score(y_test,(test_features[:,0]>=model.threshold_).astype(int)),
            "regularized_logistic_same_descriptors":_score(y_test,classical.predict(test_features)),
        },
        "selection":{"quantum_candidates":1,"classical_candidates":3,"classical_validation_scores":scores,"classical_selected":config},
        "training_loss":model.training_loss_,
        "resources":{"qubits":model.n_qubits_,"trainable_parameters":len(model.theta_),"objective_evaluations":1+3*steps,
                     "training_state_evaluations":training_evaluations,"test_state_evaluations":inference_evaluations,
                     "triage_interval":list(model.uncertain_interval_),"tfidf_threshold":model.threshold_},
    }


def _multiclass(framework, seed, steps):
    dataset = load_iris()
    train, rest = train_test_split(np.arange(len(dataset.target)),train_size=60,stratify=dataset.target,random_state=seed)
    val, test = train_test_split(rest,train_size=30,stratify=dataset.target[rest],random_state=seed+1)
    x,y = dataset.data,dataset.target
    candidates = []
    for topology in ("y","linear","full"):
        for scale in (.5,1.):
            model = MulticlassQuantumKernel(framework=framework,topology=topology,scale=scale).fit(x[train],y[train])
            candidates.append((model,model.predict(x[val]),{"topology":topology,"scale":scale,"C":1.}))
    quantum, q_config, q_scores = _choose(candidates,y[val])
    quantum_search_evaluations = sum(item[0].state_evaluations_ for item in candidates)
    classical_candidates = []
    for kernel in ("linear","rbf","poly"):
        for c in (.3,3.):
            estimator = make_pipeline(StandardScaler(),SVC(kernel=kernel,C=c)).fit(x[train],y[train])
            classical_candidates.append((estimator,estimator.predict(x[val]),{"kernel":kernel,"C":c}))
    classical, c_config, c_scores = _choose(classical_candidates,y[val])
    # A control with exactly the selected Pauli model's preprocessing, C and scale.
    pauli = next(item[0] for item in candidates if item[2]["topology"] == "y" and item[2]["scale"] == q_config["scale"])
    from sklearn.multiclass import OneVsRestClassifier
    a,b = pauli.angles(x[train]),pauli.angles(x[test])
    exact_control = OneVsRestClassifier(SVC(kernel="precomputed",C=1.)).fit(classical_product_kernel(a,a),y[train])
    pauli_predictions = pauli.predict(x[test])
    quantum_predictions = quantum.predict(x[test])
    return {
        "dataset":"scikit-learn Iris; new 60/30/60 stratified split; not paper split",
        "split_ids":{"train":train.tolist(),"validation":val.tolist(),"test":test.tolist()},
        "test_targets":y[test].tolist(),
        "models":{
            "selected_quantum_kernel":_score(y[test],quantum_predictions),
            "selected_classical_svc":_score(y[test],classical.predict(x[test])),
            "pauli_quantum_control":_score(y[test],pauli_predictions),
            "identical_closed_form_pauli":_score(y[test],exact_control.predict(classical_product_kernel(b,a))),
        },
        "selection":{"quantum_candidates":6,"classical_candidates":6,"quantum_validation_scores":q_scores,
                     "classical_validation_scores":c_scores,"quantum_selected":q_config,"classical_selected":c_config},
        "resources":{"qubits":x.shape[1],"search_state_evaluations":quantum_search_evaluations,
                     "test_state_evaluations":sum(item[0].state_evaluations_ for item in candidates)-quantum_search_evaluations,
                     "train_gram_entries":len(train)**2,"train_overlap_pairs_per_candidate":len(train)*(len(train)-1)//2,
                     "test_overlap_pairs_per_candidate":len(test)*len(train),"cached_states_per_candidate":len(train),
                     "one_statevector_bytes":16*2**x.shape[1]},
    }


def _robustness(framework, seed, steps):
    dataset = load_digits()
    eligible = np.flatnonzero(np.isin(dataset.target,[0,1]))
    selected, _ = train_test_split(eligible,train_size=96,stratify=dataset.target[eligible],random_state=seed)
    train, rest = train_test_split(selected,train_size=48,stratify=dataset.target[selected],random_state=seed)
    val, test = train_test_split(rest,train_size=24,stratify=dataset.target[rest],random_state=seed+1)
    x = dataset.images.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)/16
    y = dataset.target
    model = AmplitudeQVCClassifier(framework=framework,steps=steps,seed=seed).fit(x[train],y[train])
    training_evaluations = model.state_evaluations_
    candidates = []
    for c in (.1,1.,10.):
        source = LogisticRegression(C=c,max_iter=500).fit(x[train],y[train])
        candidates.append((source,source.predict(x[val]),{"C":c}))
    source, config, scores = _choose(candidates,y[val])
    target = MLPClassifier(hidden_layer_sizes=(8,),alpha=.01,max_iter=300,random_state=seed+13).fit(x[train],y[train])
    classical_attack = projected_gradient_attack(source.predict_proba,x[test],y[test],epsilon=.2,steps=3,seed=seed)
    quantum_attack = projected_gradient_attack(model.predict_proba,x[test],y[test],epsilon=.2,steps=3,seed=seed)
    models = {}
    for name,estimator in (("quantum_qvc",model),("classical_source_logistic",source),("independent_classical_target_mlp",target)):
        models[name] = {"clean":_score(y[test],estimator.predict(x[test])),
                        "classical_source_attack":_score(y[test],estimator.predict(classical_attack)),
                        "quantum_whitebox_attack":_score(y[test],estimator.predict(quantum_attack))}
    return {
        "dataset":"scikit-learn digits 0/1; pooled 4x4 images; not the full-sized paper datasets",
        "split_ids":{"train":train.tolist(),"validation":val.tolist(),"test":test.tolist()},
        "test_targets":y[test].tolist(),"models":models,"training_loss":model.training_loss_,
        "selection":{"quantum_candidates":1,"classical_source_candidates":3,"classical_source_validation_scores":scores,"classical_source_selected":config,"classical_target_candidates":1},
        "attack":{"norm":"L-infinity","epsilon":.2,"steps":3,"random_restarts":1,
                  "gradient":"central differences of simulator cross-entropy before amplitude normalization",
                  "clean_inputs":x[test].tolist(),"classical_perturbed_inputs":classical_attack.tolist(),"quantum_perturbed_inputs":quantum_attack.tolist(),
                  "scope":"Diagnostic only; stronger attacks and convergence sweeps remain required"},
        "resources":{"qubits":model.n_qubits_,"trainable_parameters":len(model.theta_),"objective_evaluations":1+3*steps,
                     "training_state_evaluations":training_evaluations,
                     "prediction_and_attack_state_evaluations":model.state_evaluations_-training_evaluations,
                     "one_statevector_bytes":16*2**model.n_qubits_},
    }


def run_reference(case="multiclass_kernel", framework="qiskit", seed=7, steps=4):
    if case not in CASES or framework not in ("qiskit","pennylane","cirq"):
        raise ValueError("Unknown reference case or framework")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        report = {"entity_matching":_entity,"multiclass_kernel":_multiclass,"adversarial_qvc":_robustness}[case](framework,seed,steps)
    return {
        "case":case,"framework":framework,"seed":seed,"spsa_steps":steps,
        "evidence_status":"local_reference_adaptation_not_paper_reproduction",
        "related_publication_doi":DOIS[case],"execution":"exact classical statevector simulation",
        "shots":None,"qpu_jobs":0,"wall_seconds":time.perf_counter()-started,
        "warnings":[str(item.message) for item in captured],**report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=CASES,default="multiclass_kernel")
    parser.add_argument("--framework",choices=("qiskit","pennylane","cirq"),default="qiskit")
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--output",type=Path,default=Path("results/local/case_model.json"))
    args = parser.parse_args()
    report = run_reference(args.case,args.framework,args.seed,args.steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved {args.case} reference adaptation to {args.output}")


if __name__ == "__main__":
    main()
