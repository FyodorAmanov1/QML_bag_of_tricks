"""Build website evidence data and README tables from canonical reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/assets/showcase.json"


def build_data():
    summary = json.loads((ROOT/"results/paper_summary.json").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT/"docs/use_cases/catalog.json").read_text(encoding="utf-8"))
    replay = summary["alloys"]["fresh_replays"]["10"]
    classical = max((row["mean"],name) for name,row in replay.items() if name != "quantum")
    channel = summary["communications"]
    cases = [
        dict(id="alloys",title="Predict alloy phases",domain="Materials",status="reproduced",
             badge="Reproduced comparator",metric="Accuracy ↑",value=f"{100*replay['quantum']['mean']:.2f}%",
             baseline=f"{100*classical[0]:.2f}% · best original classical comparator",
             mechanism="A fidelity kernel offers a different similarity geometry for small materials datasets.",
             scope="10-qubit replay, 10 splits. Stronger tuned classical models lead in follow-up comparisons.",
             guide="reproductions/alloys/",paper="https://doi.org/10.1016/j.matt.2024.05.035",venue="Matter · 2024"),
        dict(id="communications",title="Recover a wireless signal",domain="Communications",status="reproduced",
             badge="Reproduced comparator",metric="Symbol error ↓",value=f"{100*(1-channel['quantum_reservoir']['mean']):.3f}%",
             baseline=f"{100*(1-channel['memoryless_logistic']['mean']):.3f}% · memoryless logistic error",
             mechanism="A fixed quantum reservoir supplies nonlinear temporal features for a classical readout.",
             scope="6 qubits, 3 fresh simulated datasets at 20 dB. Stronger causal classical controls lead.",
             guide="reproductions/communications/",paper="https://doi.org/10.1038/s41467-024-51162-7",venue="Nature Communications · 2024"),
    ]
    descriptions = {
        "earth_observation":("Classify satellite images","Earth observation","A compact hybrid image head combines learned encoding with a reconstruction objective."),
        "molecular_toxicity":("Screen a molecular endpoint","Molecular screening","Annealer-based selection tests a regularized ensemble of weak learners."),
        "hyperparameter_search":("Spend fewer training epochs","ML infrastructure","A quantum-trained learning-curve predictor helps prioritize neural-network trials."),
        "entity_matching":("Match ambiguous records","Entity matching","TF-IDF handles easy pairs; a jointly trained hybrid head handles an ambiguous region."),
        "multiclass_kernel":("Choose a multiclass kernel","Tabular learning","Select a task-dependent fidelity geometry and fit a classical one-vs-rest classifier."),
        "adversarial_qvc":("Study attack transfer","Adversarial ML","A different learned representation can resist perturbations from a classical surrogate."),
    }
    for app in catalog["applications"]:
        title, domain, mechanism = descriptions[app["id"]]
        qualitative = app.get("comparison_kind") == "qualitative_figure"
        if qualitative:
            value = "Higher mean CV accuracy" if app["id"] == "multiclass_kernel" else "Greater transfer resilience"
            baseline = "Tested linear, polynomial, sigmoid and Gaussian SVMs" if app["id"] == "multiclass_kernel" else "Tested ConvNet / ResNet18 targets"
        else:
            metric = app["metric"]
            choose = max if metric["direction"] == "higher" else min
            baseline_row = choose(app["classical_comparators"],key=lambda item:item["value"])
            if metric["unit"] == "epochs":
                value = f"{app['quantum']['value']:,}"
                baseline = f"{baseline_row['value']:,} · {baseline_row['model']} epochs"
            elif metric["name"] == "test accuracy":
                value = f"{100*app['quantum']['value']:.1f}%"
                baseline = f"{100*baseline_row['value']:.1f}% · {baseline_row['model']}"
            else:
                value = f"{app['quantum']['value']:.4f}"
                baseline = f"{baseline_row['value']:.4f} · {baseline_row['model']}"
        adaptation = bool(app.get("local_model"))
        cases.append(dict(id=app["id"],title=title,domain=domain,
            status="adaptation" if adaptation else "reported",
            badge="Paper result · adaptation available" if adaptation else "Paper lead",
            metric=("Figure-only comparison" if qualitative else app["metric"]["name"]+(" ↓" if app["metric"]["direction"]=="lower" else " ↑")),
            value=value,baseline=baseline,mechanism=mechanism,scope=app["headline_caveat"],
            guide="use_cases/"+Path(app["guide"]).stem+"/",paper=app["publication"]["url"],
            venue=app["publication"]["venue"]+" · "+str(app["publication"]["year"])))
    strategies = json.loads((ROOT/"src/qml_tricks/catalog.json").read_text(encoding="utf-8"))
    learning = json.loads((ROOT/"src/qml_tricks/learning/catalog.json").read_text(encoding="utf-8"))
    opportunities = json.loads((ROOT/"results/opportunities_summary.json").read_text(encoding="utf-8"))
    configuration = ["late_pauli_query",8,512,0.0]
    selected = {row["model"]:row for row in opportunities["summary"] if row["configuration"]==configuration}
    quantum_data = dict(quantum=f"{100*selected['bell_logistic_l2']['mean']:.0f}",
                        classical=f"{100*selected['local_bayes']['mean']:.2f}",qubits=8,coherent_qubits=16,copies=512,
                        source="results/opportunities_summary.json",measurement_noise=0.0)
    return dict(checked_on=catalog["checked_on"],cases=cases,quantum_data=quantum_data,
                counts=dict(reproductions=2,paper_leads=len(catalog["applications"]),
                            adaptations=sum(bool(a.get("local_model")) for a in catalog["applications"]),
                            strategies=len(strategies),learning=len(learning),sdks=3,
                            recipe_entrypoints=3*(len(strategies)+len(learning))))


def readme_table(cases):
    lines = ["| Application | Quantum result | Classical comparison | Evidence and scope |",
             "|---|---|---|---|"]
    for case in cases:
        guide = "docs/"+case["guide"].rstrip("/")+".md"
        label = "Local comparator replay" if case["status"]=="reproduced" else "Published result"
        lines.append(f"| [{case['title']}]({guide}) | **{case['value']}** · {case['metric']} | {case['baseline']} | {label}. {case['scope']} |")
    return "\n".join(lines)


def update_block(markdown, name, body):
    start,end = f"<!-- showcase:{name}:start -->",f"<!-- showcase:{name}:end -->"
    if markdown.count(start)!=1 or markdown.count(end)!=1:
        raise ValueError(f"README needs exactly one {name} block")
    before,rest = markdown.split(start,1)
    _,after = rest.split(end,1)
    return before+start+"\n"+body+"\n"+end+after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    data = build_data()
    serialized = json.dumps(data,indent=2,ensure_ascii=False)+"\n"
    readme = (ROOT/"README.md").read_text(encoding="utf-8")
    updated = update_block(readme,"reproductions",readme_table(data["cases"][:2]))
    updated = update_block(updated,"papers",readme_table(data["cases"][2:]))
    if args.check:
        if not DEST.exists() or DEST.read_text(encoding="utf-8")!=serialized or updated!=readme:
            raise SystemExit("Showcase is stale; run python scripts/build_showcase.py")
        print("Verified website evidence data and README comparisons against canonical reports")
    else:
        DEST.write_text(serialized,encoding="utf-8",newline="\n")
        (ROOT/"README.md").write_text(updated,encoding="utf-8",newline="\n")
        print("Built eight application cards and source-derived README tables")


if __name__ == "__main__":
    main()
