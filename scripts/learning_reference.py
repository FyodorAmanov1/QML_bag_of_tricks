"""Run every new native entrypoint and bounded end-to-end reference workflows."""
import json
import warnings
from pathlib import Path
from importlib.metadata import version
from qml_tricks.learning.demos import run_recipe
from qml_tricks.learning.benchmark import benchmark,generative_benchmark

ROOT=Path(__file__).resolve().parents[1]
catalog=json.loads((ROOT/'src/qml_tricks/learning/catalog.json').read_text())
rows=[]
for recipe in catalog:
    for sdk in ('qiskit','pennylane','cirq'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=run_recipe(recipe['id'],sdk)
        rows.append(dict(method=recipe['id'],framework=sdk,seed=7,result=result,warnings=[str(w.message) for w in caught]))
    print(f"Verified three SDKs: {recipe['id']}",flush=True)
workflows={}
for task in ('classification','regression','quantum'):
    workflows[task]=benchmark(task=task,samples=30,seeds=(7,17,27),steps=1,
        methods=('reuploading','frozen_features','fidelity_kernel','projected_kernel','classical_pretraining','classical_warm_start','nystrom','kernel_surrogate','spsa'))
    print(f'Completed benchmark: {task}',flush=True)
workflows['generative']=generative_benchmark(steps=1)
payload=dict(native_runs=rows,workflows=workflows,versions={x:version(x) for x in ('qiskit','pennylane','cirq-core','scikit-learn')})
(ROOT/'results/learning_reference.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(f'Wrote {len(rows)} native entrypoint results and four benchmark workflows.')
