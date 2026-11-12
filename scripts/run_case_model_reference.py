"""Run all nine SDK/case combinations; no quantum-win acceptance condition."""
from pathlib import Path
import hashlib
import importlib.metadata
import json

from qml_tricks.learning.case_benchmarks import CASES, run_reference

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [ROOT/"src/qml_tricks/learning/case_models.py",ROOT/"src/qml_tricks/learning/case_benchmarks.py",
               ROOT/"scripts/assemble_learning.py",Path(__file__).resolve(),
               *sorted((ROOT/"src/qml_tricks/learning/case_backends").glob("*.py"))]
    report = {
        "scope":"Executed reduced adaptations; published success claims are separate literature records.",
        "versions":{name:importlib.metadata.version(name) for name in ("numpy","scipy","scikit-learn","qiskit","pennylane","cirq-core")},
        "sources":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in sources},
        "runs":[],
    }
    for sdk in ("qiskit","pennylane","cirq"):
        for case in CASES:
            result = run_reference(case,sdk,seed=7,steps=4)
            report["runs"].append(result)
            print(f"{sdk}/{case}: saved adaptation predictions ({result['wall_seconds']:.1f}s)",flush=True)
    path = ROOT/"results/case_models_reference.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
