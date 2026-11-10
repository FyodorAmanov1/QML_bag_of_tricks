"""Generate independent native preparation modules and linked learning guides."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
"qiskit": '''import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def state(theta, x, n, depth):
    c = QuantumCircuit(n)
    for l in range(depth):
        for i in range(n):
            c.ry(float(x[i]), i)
            c.rz(float(x[i]/2), i)
            c.ry(float(theta[l*n+i]), i)
        for i in range(n-1): c.cz(i,i+1)
    v = np.asarray(Statevector.from_instruction(c))
    order = [int(format(i, f'0{n}b')[::-1],2) for i in range(2**n)]
    return v[order]
''',
"pennylane": '''import numpy as np
import pennylane as qml

def state(theta, x, n, depth):
    dev = qml.device('default.qubit', wires=n)
    @qml.qnode(dev)
    def circuit():
        for l in range(depth):
            for i in range(n):
                qml.RY(float(x[i]), wires=i)
                qml.RZ(float(x[i]/2), wires=i)
                qml.RY(float(theta[l*n+i]), wires=i)
            for i in range(n-1): qml.CZ(wires=[i,i+1])
        return qml.state()
    return np.asarray(circuit(),dtype=complex)
''',
"cirq": '''import numpy as np
import cirq

def state(theta, x, n, depth):
    q = cirq.LineQubit.range(n); c = cirq.Circuit()
    for l in range(depth):
        for i in range(n):
            c.append(cirq.ry(float(x[i]))(q[i]))
            c.append(cirq.rz(float(x[i]/2))(q[i]))
            c.append(cirq.ry(float(theta[l*n+i]))(q[i]))
        for i in range(n-1): c.append(cirq.CZ(q[i],q[i+1]))
    return cirq.Simulator(dtype=np.complex128).simulate(c,qubit_order=q).final_state_vector
'''
}

RECIPES = [
("kipu_offline_surrogate", "Kipu-style offline quantum-feature surrogates", "https://kipu-quantum.com/blog/classical-surrogates-for-quantum-feature-extraction", "Public vendor workflow: acquire quantum feature pairs, fit a classical surrogate, and deploy with a classical predictor. Ridge is the student family reported in Kipu's example. The native demos here use small reference teachers, not Kipu's DQFE circuits, proprietary SDK, datasets or claimed accuracy figures. Held-out fidelity is an empirical acceptance gate, not a universal theorem.", "See [the full teacher-first workflow](kipu_workflow.md). Use an already-fitted quantum feature teacher or import real Kipu input/output feature pairs. Freeze the teacher before distillation, preserve the downstream predictor, and export a teacher-free classical artifact only after validation. This entry explicitly differs from density-embedding fidelity-kernel approximation and Nyström."),
("parameter_shift", "Parameter-shift differentiation", "https://doi.org/10.1103/PhysRevA.99.032331", "Exact expectation derivative identity for a single Pauli rotation occurrence per coordinate; unbiased sampled expectation gradients. Nonlinear losses require a classical chain rule; shifting the loss itself is generally incorrect.", "Use finite-shot training when numerical differences would amplify shot noise. Shared coordinates need occurrence-wise derivatives; arbitrary generators need generalized rules."),
("reuploading", "Angle encoding and data re-uploading", "https://doi.org/10.22331/q-2020-02-06-226", "Expressivity results for the published constructions; no universal convergence or advantage claim for the small executable.", "Use bounded continuous features and a small register. Repeated uploads trade depth for expressivity. The native example uses one trainable Y rotation per occurrence, with angle and phase data gates."),
("fidelity_kernel", "Fidelity quantum kernels", "https://doi.org/10.1038/s41586-019-0980-2", "The ideal kernel |<psi(x)|psi(y)>|^2 is PSD; the regularized downstream optimization is convex. No general trainability or predictive superiority guarantee.", "Use moderate sample counts: exact training uses quadratic pair evaluations. The finite-shot Gram matrix can be indefinite; training-only eigenspace features give a consistent PSD out-of-sample extension."),
("projected_kernel", "Projected quantum kernels", "https://doi.org/10.1038/s41467-021-22539-9", "A Gaussian kernel on fixed measured feature vectors is PSD. The source's advantage statements need its particular access and task assumptions, which this toy implementation does not establish.", "Measure local Z features then use an RBF kernel. Cache the measured feature vectors so sampling does not make the same feature map inconsistent between calls. Compare a classical readout on exactly those same features."),
("frozen_features", "Frozen quantum features and convex readouts", "https://doi.org/10.1103/PhysRevLett.122.040504", "For a fixed feature matrix and positive ridge regularization, the readout objective is strictly convex. This does not guarantee useful quantum features.", "Use a fixed quantum encoder to avoid variational circuit optimization. Fit ridge readouts for regression or one-hot least-squares classification; softmax scores are not calibrated probabilities."),
("spsa", "Simultaneous perturbation stochastic approximation", "https://doi.org/10.1109/9.119632", "Conditional stochastic-approximation convergence under the source's smoothness, noise, boundedness and gain-sequence assumptions. Two loss calls per perturbation, not two total shots.", "Use noisy objectives with many coordinates. The finite-step implementation uses diminishing a and c; it is not a barren-plateau cure or a guaranteed speedup."),
("shot_allocation", "Independent-term shot allocation", "https://doi.org/10.22331/q-2020-05-11-263", "For independent sample means, variance is sum c_i^2 sigma_i^2/N_i. Continuous optimal known-variance allocation is proportional to |c_i| sigma_i; integer allocation is rounded and not claimed exactly optimal.", "Use unequal observable variances under a total shot budget. The demo takes supplied variances; real use needs pilot measurements counted in the same total budget. Correlated grouped measurements need covariance analysis."),
("classical_shadows", "Classical shadows for local quantum features", "https://doi.org/10.1038/s41567-020-0932-7", "Unbiased local Pauli shadow estimators; median-of-means sample complexity depends on shadow norms, locality, accuracy and confidence. Arbitrary default shot counts do not certify the theorem's confidence.", "Use quantum-native states with many requested local observables. The executable prepares a native SDK state and simulates random Pauli measurement outcomes on the host. It does not assume full amplitudes are available on hardware."),
("born_mmd", "Born machines with characteristic-kernel MMD", "https://doi.org/10.1103/PhysRevA.98.062324", "Correct population-MMD parameter-shift gradients and unbiased MMD-squared U-statistics. A characteristic Gaussian kernel identifies distributions on finite binary support; no global optimization guarantee.", "Use generative learning with samples, including correlations missed by one-bit losses. Negative finite-sample MMD estimates are retained. Distinct samples are required for diagonal-free within-distribution terms."),
("classical_pretraining", "Classically pretrained feature extractors", "https://doi.org/10.22331/q-2020-10-09-340", "The frozen ridge readout is strictly convex conditional on the learned features. Supervised MLP pretraining is an empirical engineering companion explicitly requested by the user; no global pretraining or plateau guarantee is asserted.", "Fit an MLP on training inputs and labels, freeze its hidden representation, then feed it to a quantum head. Save the extractor and compare a classical ridge head on the SAME representation. PCA alone is preprocessing, not supervised pretraining."),
("classical_warm_start", "Classical simulator warm starts for quantum parameters", "https://doi.org/10.22331/q-2019-12-09-214", "The gradient identity remains exact on the simulator; warm-start effectiveness has no general guarantee here. This is an explicitly requested engineering companion, not a claimed theorem from the linked initialization paper.", "Optimize the same native small circuit with ideal classical simulation, then transfer its angles to the sampled phase. Report exponential simulator storage and separate pretraining calls. This does not scale past classical simulation limits."),
("nystrom", "Nyström approximation of quantum kernels", "https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf", "For PSD kernels, landmark features produce a PSD low-rank approximation; using all landmarks with positive eigenspaces recovers the ideal Gram matrix up to tolerance. Accuracy with few landmarks depends on spectrum and selection.", "Use a smaller landmark set to reduce kernel evaluations. This still needs quantum features or kernel queries for new inputs; it is not fully classical inference."),
("kernel_surrogate", "Density-embedding fidelity-kernel surrogate (not Kipu DQFE)", "https://www.nature.com/articles/s41467-024-49877-8", "The learned embedding inner product is PSD by construction. If density-embedding errors are uniformly <= epsilon and target norms are one, kernel errors are <= 2 epsilon + epsilon^2. The error premise is not guaranteed by finite training data. This concrete small-state student is not a reproduction of the cited shadow-model theorem.", "Collect quantum teacher embeddings on TRAINING data, fit a classical RBF-ridge student, and evaluate new kernels without quantum calls. Here teacher density matrices use small exact simulations, an exponential acquisition cost. Held-out kernel error must be audited separately. Random Fourier approximation of an ordinary RBF kernel alone is not approximation of an arbitrary quantum kernel."),
]

def main():
    folder = ROOT/'src/qml_tricks/learning/backends';folder.mkdir(parents=True,exist_ok=True)
    (folder/'__init__.py').write_text('"""Independent native state preparation backends."""\n')
    for sdk, code in BACKENDS.items(): (folder/f'{sdk}.py').write_text(code,encoding='utf-8')
    guides = ROOT/'docs/learning'; guides.mkdir(parents=True,exist_ok=True)
    for key,title,source,guarantee,use in RECIPES:
        directory = ROOT/'src/qml_tricks/learning/recipes'/key;directory.mkdir(parents=True,exist_ok=True)
        (directory/'__init__.py').write_text('')
        for sdk in BACKENDS:
            (directory/f'{sdk}_impl.py').write_text(f'''"""{title}: native {sdk} preparation and shared mathematical recipe."""
from qml_tricks.learning.backends.{sdk} import state
from qml_tricks.learning.demos import run_recipe

def run(seed=7):
    return run_recipe("{key}", "{sdk}", seed)

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
''',encoding='utf-8')
        (guides/f'{key}.md').write_text(f'''# {title}

## When to use it

{use}

## Mathematical and evidence contract

{guarantee}

Primary source: [{title}]({source}). The source supports the stated mechanism;
the executable's restricted scope is specified above. A formal identity, convex
readout, or PSD approximation is not a universal barren-plateau or advantage proof.

## Implementation contract for an agent

1. Read the data-access assumptions and source result before transferring the recipe.
2. Install `.[learning]` plus the selected SDK extra, or `.[all,learning]`.
3. Run the native example and inspect estimator error, resource counts and scope.
4. Fit preprocessing on training data only. Reuse the same splits for controls.
5. Record the quantum acquisition cost separately from classical fitting and inference.
6. Validate the mathematical property on a small independently calculable case.

```bash
python -m qml_tricks learning-demo {key} --framework qiskit
python -m qml_tricks learning-demo {key} --framework pennylane
python -m qml_tricks learning-demo {key} --framework cirq
```

The native implementations use [Qiskit](../../src/qml_tricks/learning/recipes/{key}/qiskit_impl.py),
[PennyLane](../../src/qml_tricks/learning/recipes/{key}/pennylane_impl.py), and
[Cirq](../../src/qml_tricks/learning/recipes/{key}/cirq_impl.py). Each imports its
own native preparation backend and shares only the estimator/learning mathematics.
Finite-shot experiments are host sampling of native SDK simulator distributions.
Simulator full-state access and density embeddings are not hardware shortcuts.

## Benchmarks and failure modes

See [the project workflow](index.md) for complete models, CSV input, repeated seeds,
classical controls, serialization and resource reports. Compare predictive error
and cost together. Kernel concentration, insufficient samples, feature collapse,
invalid parameter-shift generators, singular landmarks, regularization sensitivity,
data leakage and distribution shift can invalidate an apparent success.

Each method needs its own assumptions after composition. The small native
examples are tests of mechanism and integration, not empirical evidence for an
asymptotic computational advantage.
''',encoding='utf-8')
    (ROOT/'src/qml_tricks/learning/recipes/__init__.py').write_text('')
    (ROOT/'src/qml_tricks/learning/catalog.json').write_text(json.dumps([dict(id=r[0],title=r[1],source=r[2],guarantee=r[3]) for r in RECIPES],indent=2)+'\n')
    print(f'Generated {len(RECIPES)} learning guides and {3*len(RECIPES)} native entrypoints.')

if __name__ == '__main__': main()
