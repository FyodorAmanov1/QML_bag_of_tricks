"""Generate bounded cross-framework reference runs and a scientific figure."""
import hashlib
import json
from pathlib import Path
import numpy as np
from qml_tricks.core import RunConfig, catalog, experiment

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for row in catalog():
        for sdk in ("qiskit", "pennylane", "cirq"):
            result = experiment(row["id"], sdk, RunConfig(n_qubits=4, steps=1))
            runs.append(result.to_dict())
        print(f"Validated reference run: {row['id']}", flush=True)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / "src").rglob("*.py")}
    (output / "reference.json").write_text(json.dumps({"source_hashes": hashes, "runs": runs}, indent=2) + "\n", encoding="utf-8")
    bibliography = []
    for row in catalog():
        source = row["source"]
        fields = [f"  title = {{{row['title']}: primary result}}", f"  url = {{{source}}}", f"  note = {{{row['locator']}}}"]
        if "doi.org/" in source:
            fields.append(f"  doi = {{{source.split('doi.org/')[1]}}}")
        bibliography.append("@misc{"+row["id"]+",\n"+",\n".join(fields)+"\n}")
    (ROOT / "references.bib").write_text("\n\n".join(bibliography)+"\n", encoding="utf-8")
    n = np.arange(2, 65)
    local, glob = 1/(8*n**2), (3/8.)**(n-1)/8
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    ax.semilogy(n, local, label="Local objective: 1 / (8 n²)", linewidth=2.5)
    ax.semilogy(n, glob, label="Global objective: (3/8)ⁿ⁻¹ / 8", linewidth=2.5)
    ax.set(xlabel="Qubits n", ylabel="Gradient variance over independent uniform angles",
           title="Product-state warm-up: exact analytical variance")
    ax.grid(alpha=.2); ax.legend()
    assets = ROOT / "docs/assets"; assets.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assets / f"local_global_variance.{ext}", dpi=180)
    (output / "analytic_variance.json").write_text(json.dumps({"n":n.tolist(), "local":local.tolist(), "global":glob.tolist(),
        "source":"https://doi.org/10.1038/s41467-021-21728-w", "interpretation":"Exact product-state calculation, not an empirical scaling fit"}, indent=2)+"\n", encoding="utf-8")
    print(f"Wrote {len(runs)} native reference runs, bibliography, and figure.")


if __name__ == "__main__":
    main()
