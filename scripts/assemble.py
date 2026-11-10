"""Authoring source for the catalog, guides, and independent native SDK modules.

Run from the repository root. Shared strings are expanded into readable native
Python files; runtime implementations do not convert between quantum SDKs.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

# id, title, category, DOI, published result locator, scope, use case,
# assumptions, mechanism/bound, implementation boundary
ROWS = [
('near_identity_encoding','Near-identity data encoding','Encoding','10.1007/s42484-023-00103-6','Data-encoding trainability analysis; the product-encoder derivative below is derived explicitly here','analytical special case; no universal encoding guarantee','Bounded classical features and tunable weak quantum embeddings',
 r'Bounded features $|x_i|\leq1$; $E_\epsilon(x)=\bigotimes_i R_y(\epsilon x_i)$; reference trainable angles $\theta_i=\pi/2$; normalized mean-Z objective.',
 r'For $C=n^{-1}\sum_i\cos(\theta_i+\epsilon x_i)$, $\partial_i C=-\sin(\theta_i+\epsilon x_i)/n$. At $\theta_i=\pi/2$ and $0\leq\epsilon<\pi/2$, $|\partial_i C|\geq\cos(\epsilon)/n$. This direct calculation establishes a signal for this example, not for arbitrary encoders, datasets, or losses. The primary paper motivates the need to audit data encoding; this particular bound is derived here.',
 'Native bounded-angle product encoder followed by trainable Y rotations. Strategy uses encoding_scale (default 0.1); baseline uses scale 1 with the same features and initialization. Encoding controls data gates, whereas identity initialization controls trainable circuit parameters.'),
('identity_block_init','Identity-block initialization','Initialization','10.22331/q-2019-12-09-214','Identity-block construction and initial-gradient analysis','initialization only; shallow-block conditions','Deeper variational models with a useful reference and shallow identity blocks',
 r'Each block has independent forward/backward parameters initialized to equal values; invert gate order and angles exactly. Preserve shallow effective causal cones and a nonzero state-observable commutator.',
 r'For $W(a,b)=B(b)^\dagger B(a)$, initialize $b=a$ so $W=I$. The derivatives with respect to the independent copies generally remain nonzero. Tying $b\equiv a$ throughout optimization instead gives $W\equiv I$ and zero gradients. In this executable, the forward Y angle on boundary wire zero has $\partial C=-\cos(a_0)\cos(a_1)$ at matched initialization, hence variance $1/4$ for independent uniform forward X angles, independent of qubit count and the number of matched shallow blocks. This is a directly derived special case, not a new statement attributed to the paper. An initial-gradient claim does not guarantee the training trajectory remains outside a plateau.',
 'Uses B=RY(c) CZ RX(a) per shallow block, with independently trainable inverse copies, matched only at initialization. Prepares |+> on wire zero and reads Z zero. Exact unitary cancellation is tested. A nonzero gradient is task dependent; no universal lower bound is asserted.'),
('natural_gradient','Quantum natural gradient descent','Optimization companions','10.22331/q-2020-05-25-269','Fubini-Study geometry and natural-gradient update; metric approximation discussion','geometric optimization method; no general barren-plateau avoidance guarantee','Ill-conditioned quantum-state parameterizations with resolvable gradients',
 r'Differentiable normalized pure state; consistently defined Fubini-Study metric; regularization or a pseudoinverse for singular directions; gradient and metric estimation budgets must be affordable.',
 r'$g_{ij}=\operatorname{Re}[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle]$, and $\theta^+=\theta-\eta(g+\lambda I)^{-1}\nabla C$. For pure states the QFIM is $F=4g$; switching conventions requires adjusting the learning rate and regularization. Preconditioning alone does not ensure the information needed to estimate these quantities is available with polynomially many shots.',
 'Computes the dense Fubini-Study metric from central derivatives of native SDK statevectors. Strategy uses a regularized QNG update; baseline uses Euclidean gradient descent with identical initialization, circuit, objective and learning rate. Even in sampled-loss mode, the metric is an exact-simulator resource, not a hardware-estimated metric.'),
('local_loss','Local losses and bounded causal cones','Objectives','10.1038/s41467-021-21728-w','Theorems 1 and 2; warm-up example, equations (1)-(3)','initialization ensemble','Quantum autoencoders and local state preparation',
 'Local observable; logarithmic alternating-layer depth; independent local 2-design blocks; nonvanishing input-dependent purity factors.',
 r'For the product-state warm-up, $C_G=1-\prod_j\cos^2(\theta_j/2)$ has $\mathrm{Var}(\partial_j C_G)=\frac18(\frac38)^{n-1}$, whereas $C_L=\frac1n\sum_j\sin^2(\theta_j/2)$ has $\mathrm{Var}(\partial_j C_L)=1/(8n^2)$. The general lower bound also contains input purity and observable factors: locality alone is insufficient.',
 'The executable uses the exact product-state warm-up. Both losses have the same zero set; that equivalence is specific to this target.'),
('qcnn','Quantum convolutional neural networks','Architectures','10.1103/PhysRevX.11.041011','Theorem 1 and Corollary 1; Theorem 2 for pooling-only networks','initialization ensemble','Classification and compression of structured quantum states',
 'Independent local 2-design convolution blocks; the specified pooling channel; a nontrivial output observable and non-exponentially-small data-dependent factors.',
 r'The gradient-variance lower bound is inverse polynomial in $n$ when the architecture has $O(\log n)$ pooling levels and its state/observable prefactors are inverse polynomial. Pooling reduces the active register rather than leaving a deep random unitary on every wire.',
 'Haar-dressed convolution and coherent controlled-Y pooling followed by discarding wires. The selected tangent is illustrative; the code does not reproduce the complete parameter ensemble in the theorem.'),
('mps','Isometric matrix product states','Architectures','10.1103/PhysRevA.109.L050402','Main gradient-scaling result; companion detailed analysis DOI 10.1007/s00220-024-05217-x','Haar tensor ensemble','Learning low-entanglement state models with local energy or moment objectives',
 'Open boundaries; isometric gauge; Haar-distributed unitary tensors; finite-range extensive Hamiltonian; controlled bond dimension.',
 r'For the specified isometric tensor ensemble, the energy-gradient second moment has a leading term independent of system size and algebraic dependence on bond dimension. Dividing an extensive energy by $n$ multiplies its gradient second moment by $n^{-2}$.',
 'Sequential two-qubit Haar blocks give a bond-dimension-two state-preparation circuit. A selected Pauli tangent is measured, not the full Riemannian tensor gradient.'),
('ttn','Isometric tree tensor networks','Architectures','10.1103/PhysRevA.109.L050402','Hierarchical tensor-network gradient scaling; detailed analysis DOI 10.1007/s00220-024-05217-x','Haar tensor ensemble','Multiscale state models and coarse-grained quantum features',
 'Isometric tree tensors; Haar tensor ensemble; finite-range extensive objective; fixed branching and controlled bond dimension.',
 r'At level $\tau$, the tensor-gradient variance scales as $(b\eta)^\tau$ in the published construction. Since $\tau=O(\log_b n)$, fixed nonzero $b\eta$ gives a power of $n$, rather than an exponential in $n$.',
 'A binary preparation tree with two-qubit Haar isometries. Diagnostic coordinate variance is distinct from the published tensor-gradient norm.'),
('mera','Isometric MERA circuits','Architectures','10.1103/PhysRevA.109.L050402','MERA gradient scaling; detailed analysis DOI 10.1007/s00220-024-05217-x','Haar tensor ensemble','Critical and multiscale quantum-state representations',
 'Published isometric MERA geometry; Haar tensor ensemble; finite-range extensive objective; bounded bond dimension and branching.',
 r'Isometries preserve normalization and disentanglers remove correlations between scales. The proven layer dependence is algebraic in system size when the number of scales is logarithmic and the transfer-channel factors remain controlled.',
 'A small binary tree with interbranch disentanglers illustrates the mechanism. It is not the complete translational MERA geometry of the proof.'),
('finite_local_depth','Finite local-depth circuits','Architectures','10.1103/PhysRevLett.132.150603','General gradient lower bound and finite-local-depth consequence','local 2-design ensemble','Long-range correlated state models with local objectives',
 'Local 2-design gates; bounded number of gates incident on each physical qubit; local Hamiltonian; nontrivial input and observable factors.',
 r'Bound local depth, the number of gates touching each qubit, rather than conventional scheduled depth. A sequential sweep can have $O(n)$ global depth while every wire participates in $O(1)$ blocks. The published lower bound remains non-exponential under the stated factors.',
 'The executable is a one-dimensional bounded-incidence Haar sweep. Its bond dimension is small and it is efficiently simulable; higher-dimensional generalizations require a separate geometry implementation.'),
('gaussian_init','Depth-dependent Gaussian initialization','Initialization','https://papers.nips.cc/paper_files/paper/2022/file/7611a3cb5d733e628081431445cb01fd-Paper-Conference.pdf','NeurIPS 2022 paper: local/global-observable gradient lower bounds','initialization only','Deep rotation-and-entangler circuits with a useful near-identity reference',
 'Specified rotation/entangler ordering; independent Gaussian angles with depth-dependent variance; observable support and nonzero initial-state overlap conditions in the appropriate bound.',
 r'Small-angle moments preserve a component of the backwards-propagated observable. Products of damping factors can remain inverse polynomial when the angle variance shrinks with depth. A fixed-variance Gaussian is not a generic cure; the bound concerns a gradient norm, not every coordinate.',
 'RX/RY and CZ circuit with conservative variance $1/(8Ln)$. This is a small-angle mechanism demonstration, not a reproduction of every local/global theorem case.'),
('gaussian_mixture','Observable-aware Gaussian mixture initialization','Initialization','10.1088/1367-2630/ae0823','Theorems 1-3; Table 1 for final-layer distributions','initialization only','Rotation circuits with a known Pauli target observable',
 'RX then RY after each CZ block; Gaussian variance $1/(2LS)$; last-layer distributions selected from Table 1 for each Pauli site; all-zero reference; theorem-specific observable coefficients.',
 r'For the single Pauli-string construction, $\mathbb E\|\nabla f\|_2^2\geq 1/4-1/(8L)$. Here $S$ is the support size. General Hamiltonian claims require the coefficient and cross-term analysis; they cannot be inferred by adding arbitrary terms.',
 r'Implements the allowed G1 choice for a $Z^{\otimes n}$ target: all angles Gaussian with variance $1/(2Ln)$. X/Y targets need the different final-layer mixture table, which this executable does not expose.'),
('correlated_parameters','Spatially correlated parameters','Initialization','10.1088/2058-9565/abd891','Separable-circuit examples and correlated-module constructions','specified parameter ensemble','Permutation-aligned learning targets and tied quantum feature blocks',
 'Correlation pattern and observable must match the construction. Parameter tying alone gives no general lower bound; resource cost can grow exponentially in some examples.',
 r'A transparent special case uses $U=\bigotimes_j R_y(\phi)$ and $C=n^{-1}\sum_j\langle Z_j\rangle=\cos\phi$. Uniform $\phi$ gives $\mathbb E\partial_\phi C=0$ and $\mathrm{Var}(\partial_\phi C)=1/2$, while independent angles give $1/(2n^2)$ per coordinate.',
 'Exactly implements this tied-angle special case. Unused coordinates are explicitly a redundant storage convention; diagnostic coordinate zero is the active tied parameter.'),
('constrained_hva','Short-time constrained Hamiltonian ansatz','Initialization','10.22331/q-2024-02-01-1239','Theorem 1, equation (30); short-time commutator analysis','specified short-time region','Physics-informed QML with a known local generator and reference state',
 r'Local generators and observable; reference commutator $g=|\mathrm{Tr}(\rho_0[H^{(m)},O])|=\Theta(1)$; nonnegative total evolution time below the theorem threshold $\tau_0=\Theta(1/n)$.',
 r'Under the threshold and nonzero-commutator condition, $|\partial_{n,m}C|\geq g/4$. Bounding evolution time controls how far the gradient can move from its reference value. Zero reference commutator invalidates this argument.',
 'Starts qubit zero in $|+\rangle$; Y rotations and ZZ couplings measure $Z_0$. Bounds every stored angle by $0.01/(nP)$ and projects updates. The constant 0.01 is a conservative demonstration choice, not a certified evaluation of the paper\'s threshold.'),
('low_excitation','Fixed low-excitation sectors','Symmetry','10.22331/q-2025-05-15-1745','Trainability analysis of Hamming-weight-preserving circuits','specified sector ensemble','Single-particle features and fixed-particle quantum data',
 'Weight-preserving circuit; reference in weight-k sector; sufficiently mixed ensemble within the relevant sector; suitable measurement and data factors. Fixed k is essential to the dimension argument.',
 r'The sector dimension is $d_k=\binom nk$. For fixed $k$, $d_k=O(n^k)$; an inverse-polynomial-in-$d_k$ variance bound is therefore inverse polynomial in $n$. At $k=\Theta(n)$ the sector can be exponentially large.',
 'Prepares one excitation and uses matched XX/YY rotations. Two layers need not mix to the theorem ensemble; exact excitation conservation is tested separately from trainability.'),
('permutation_equivariant','Permutation-equivariant quantum models','Symmetry','10.1038/s41534-024-00804-1','Trainability results and dataset-dependent conditions','specified equivariant ensemble','Permutation-invariant quantum data and unordered feature sets',
 'Full S_n-equivariant generators; compatible measurement; input projection onto relevant representation sectors must not be exponentially weak; use the published mixing/ensemble assumptions.',
 r'The permutation-invariant operator algebra has polynomial dimension, with the relevant dynamical algebra scaling as $O(n^3)$. The theorem also contains state and observable projections: polynomial algebra size is not sufficient on its own.',
 'Collective Y rotations and equal-angle all-pairs ZZ interactions commute with permutations. The short circuit verifies symmetry but does not certify convergence to the ensemble used in the theorem.'),
('matchgate','Observable-aligned matchgate circuits','Symmetry','10.1038/s41467-024-49909-3','Lie-algebra variance theorem; applications to free-fermionic algebras','specified mixed group ensemble','Free-fermion state models and quadratic observables',
 'Nearest-neighbor fermionic Gaussian generators; observable in a polynomial-size algebra/module; non-exponentially-small input projection; the group-ensemble assumptions of the variance theorem.',
 r'For quadratic fermionic generators, $\dim\mathfrak{so}(2n)=n(2n-1)$. The algebraic variance formula depends on inverse algebra dimension and state/observable generalized purities. Arbitrary spin observables can inhabit exponentially large modules.',
 'RZ and nearest-neighbor XX gates with local Z readout. Small-depth uniform angles demonstrate the architecture rather than sampling Haar over the complete Gaussian group.'),
('engineered_dissipation','Engineered dissipation and objective localization','Channels','10.1038/s41534-024-00875-0','Effective-Hamiltonian construction and Supplementary Note 1','specified engineered channel','State learning with a known product reference and controlled reset resource',
 'A tailored Liouvillian with known stationary state; non-exponentially-small effective-Hamiltonian coefficients; faithfulness to the original task must be checked independently.',
 r'Heisenberg evolution gives $H_{\mathrm{eff}}=\mathcal E^\dagger(H)$. A suitable dissipative layer can create useful local contributions on logarithmic timescales. This does not guarantee that minimizing the transformed objective solves the original task.',
 'Amplitude damping after each local layer demonstrates channel control. It does not implement the full paper-specific effective-Hamiltonian optimization or establish faithfulness; original and transformed task quality require independent evaluation.'),
('nonunital_noise','Non-unital noise and late-layer trainability','Channels','10.1038/s41567-026-03245-z','Local-observable variance results and effective-depth theorems','random noisy ensemble; parameter-position dependent','Studying late trainable layers on noisy processors',
 'Local random gates; specified local non-unital noise; local output observables. Bounds depend on distance from output; useful input information and earlier gradients can still be lost.',
 r'Noise contracts the influence of earlier layers. Non-unital channels can maintain a nonflat output landscape even when an effectively shallow tail controls the prediction. This does not establish useful gradients for every parameter at arbitrary depth.',
 'Amplitude damping with Haar-dressed local blocks, comparing damping with the same noiseless circuit. The diagnostic selects the first rotation of the final layer; other parameter positions can behave differently.'),
('mmd_loss','Local-moment MMD objectives','Objectives','10.1038/s41534-024-00902-0','Implicit-loss analysis and kernel bodyness discussion','specified loss and shallow architecture','Sample-based generative learning of selected local statistics',
 'An implicit sample loss; significant low-body kernel contributions; shallow compatible circuit and nonvanishing data-dependent factors. Full-distribution identification needs a richer kernel.',
 r'For $k(x,y)=n^{-1}\sum_i z_i(x)z_i(y)$, $\mathrm{MMD}^2=n^{-1}\sum_i(\mathbb E_p z_i-\mathbb E_q z_i)^2$. Local statistics avoid multiplying many independent overlaps. This degree-one kernel cannot distinguish distributions with identical one-bit marginals.',
 'Fits target moments 0.5, corresponding to independent bits with P(1)=0.25. Uses a plug-in finite-shot squared loss, whose finite-sample bias is disclosed. No asymptotic lower bound is asserted for this arbitrary dataset.'),
]

def source_url(source):
    return source if source.startswith('https:') else 'https://doi.org/' + source

BACKENDS = {
 'qiskit': ('''import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, Kraus
''', 'c = QuantumCircuit(n)', {
 'ry': 'c.ry({t}, {a})', 'rx':'c.rx({t}, {a})','rz':'c.rz({t}, {a})',
 'x':'c.x({a})','h':'c.h({a})','cz':'c.cz({a}, {b})',
 'xx':'c.rxx({t}, {a}, {b})','yy':'c.ryy({t}, {a}, {b})','zz':'c.rzz({t}, {a}, {b})',
 'pool':'c.cry({t}, {a}, {b})', 'haar':'c.unitary(haar(config.seed, {index}), [{b}, {a}])',
 'noise':'c.append(Kraus([np.diag([1, np.sqrt(1 - gamma)]), np.array([[0, np.sqrt(gamma)], [0, 0]])]).to_instruction(), [{a}])'},
 '''    c = build_circuit(theta, config)
    state = DensityMatrix.from_instruction(c) if CHANNEL else Statevector.from_instruction(c)
    p = state.probabilities()
    # Qiskit indexes qubit zero as the least significant bit.
    order = [int(format(i, f"0{config.n_qubits}b")[::-1], 2) for i in range(len(p))]
    return _normalize(p[order])
''', '''    c = build_circuit(theta, config)
    return {"qubits": c.num_qubits, "ancillas": 0, "native_operations": len(c.data),
            "two_qubit_operations": sum(len(x.qubits) == 2 for x in c.data),
            "native_depth": c.depth(), "gate_count_stage": "logical, before hardware synthesis"}
'''),
 'pennylane': ('''import numpy as np
import pennylane as qml
''', 'c = None', {
 'ry':'qml.RY({t}, wires={a})','rx':'qml.RX({t}, wires={a})','rz':'qml.RZ({t}, wires={a})',
 'x':'qml.PauliX(wires={a})','h':'qml.Hadamard(wires={a})','cz':'qml.CZ(wires=[{a}, {b}])',
 'xx':'qml.IsingXX({t}, wires=[{a}, {b}])','yy':'qml.IsingYY({t}, wires=[{a}, {b}])','zz':'qml.IsingZZ({t}, wires=[{a}, {b}])',
 'pool':'qml.CRY({t}, wires=[{a}, {b}])','haar':'qml.QubitUnitary(haar(config.seed, {index}), wires=[{a}, {b}])',
 'noise':'qml.AmplitudeDamping(gamma, wires={a})'},
 '''    dev = qml.device("default.mixed" if CHANNEL else "default.qubit", wires=config.n_qubits)
    @qml.qnode(dev)
    def circuit():
        build_circuit(theta, config)
        return qml.probs(wires=range(config.n_qubits))
    return _normalize(np.asarray(circuit(), dtype=float))
''', '''    with qml.queuing.AnnotatedQueue() as queue:
        build_circuit(theta, config)
    tape = qml.tape.QuantumScript.from_queue(queue)
    return {"qubits": config.n_qubits, "ancillas": 0, "native_operations": len(tape.operations),
            "two_qubit_operations": sum(len(op.wires) == 2 for op in tape.operations),
            "native_depth": tape.graph.get_depth(), "gate_count_stage": "logical, before hardware synthesis"}
'''),
 'cirq': ('''import numpy as np
import cirq
''', 'q = cirq.LineQubit.range(n)\n    c = cirq.Circuit()', {
 'ry':'c.append(cirq.ry({t})(q[{a}]))','rx':'c.append(cirq.rx({t})(q[{a}]))','rz':'c.append(cirq.rz({t})(q[{a}]))',
 'x':'c.append(cirq.X(q[{a}]))','h':'c.append(cirq.H(q[{a}]))','cz':'c.append(cirq.CZ(q[{a}], q[{b}]))',
 'xx':'c.append(cirq.XXPowGate(exponent=({t})/np.pi, global_shift=-0.5)(q[{a}], q[{b}]))',
 'yy':'c.append(cirq.YYPowGate(exponent=({t})/np.pi, global_shift=-0.5)(q[{a}], q[{b}]))',
 'zz':'c.append(cirq.ZZPowGate(exponent=({t})/np.pi, global_shift=-0.5)(q[{a}], q[{b}]))',
 'pool':'c.append(cirq.ry({t})(q[{b}]).controlled_by(q[{a}]))',
 'haar':'c.append(cirq.MatrixGate(haar(config.seed, {index}))(q[{a}], q[{b}]))',
 'noise':'c.append(cirq.amplitude_damp(gamma)(q[{a}]))'},
 '''    c = build_circuit(theta, config)
    order = cirq.LineQubit.range(config.n_qubits)
    if CHANNEL:
        result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(c, qubit_order=order)
        p = np.real(np.diag(result.final_density_matrix))
    else:
        result = cirq.Simulator(dtype=np.complex128).simulate(c, qubit_order=order)
        p = np.abs(result.final_state_vector)**2
    return _normalize(p)
''', '''    c = build_circuit(theta, config)
    ops = list(c.all_operations())
    return {"qubits": config.n_qubits, "ancillas": 0, "native_operations": len(ops),
            "two_qubit_operations": sum(len(op.qubits) == 2 for op in ops),
            "native_depth": len(c), "gate_count_stage": "logical, before hardware synthesis"}
''')
}

def body(strategy, ops):
    lines=[]
    def emit(code): lines.append(code)
    def op(name, **kw): emit(ops[name].format(**kw))
    def layer_rot(axis='ry', angle='theta[2*l*n+i]'):
        emit('for i in range(n):'); op(axis,t=angle,a='i'); lines[-1]='    '+lines[-1]
    if strategy in {'mps','finite_local_depth','ttn','mera','qcnn'}:
        emit('if config.variant == "baseline":')
        if strategy == 'qcnn':
            emit('    for i in range(n):'); op('ry',t='0.2 + 0.1*i',a='i');lines[-1]='        '+lines[-1]
        emit('    for l in range(config.depth):');emit('        for i in range(n-1):')
        op('haar',index='l*n+i',a='i',b='i+1');lines[-1]='            '+lines[-1]
        op('ry',t='theta[(l*n+i) % len(theta)]',a='i');lines[-1]='            '+lines[-1]
        emit('    return c')
    if strategy == 'near_identity_encoding':
        emit('features = np.linspace(-1.0, 1.0, n) if config.features is None else np.asarray(config.features)')
        emit('scale = config.encoding_scale if config.variant == "strategy" else 1.0')
        emit('for i in range(n):')
        op('ry',t='scale * features[i]',a='i');lines[-1]='    '+lines[-1]
        op('ry',t='theta[i]',a='i');lines[-1]='    '+lines[-1]
    elif strategy == 'identity_block_init':
        op('h',a='0')
        emit('for l in range(config.depth):')
        emit('    base = 4*l*n')
        emit('    for i in range(n):');op('rx',t='theta[base+i]',a='i');lines[-1]='        '+lines[-1]
        emit('    for i in range(n-1):');op('cz',a='i',b='i+1');lines[-1]='        '+lines[-1]
        emit('    for i in range(n):');op('ry',t='theta[base+n+i]',a='i');lines[-1]='        '+lines[-1]
        emit('    for i in reversed(range(n)):');op('ry',t='-theta[base+2*n+i]',a='i');lines[-1]='        '+lines[-1]
        emit('    for i in reversed(range(n-1)):');op('cz',a='i',b='i+1');lines[-1]='        '+lines[-1]
        emit('    for i in reversed(range(n)):');op('rx',t='-theta[base+3*n+i]',a='i');lines[-1]='        '+lines[-1]
    elif strategy == 'natural_gradient':
        emit('for i in range(n):')
        op('ry',t='theta[i]',a='i');lines[-1]='    '+lines[-1]
        op('rz',t='theta[n+i]',a='i');lines[-1]='    '+lines[-1]
        emit('for i in range(n-1):');op('cz',a='i',b='i+1');lines[-1]='    '+lines[-1]
        emit('for i in range(n):');op('h',a='i');lines[-1]='    '+lines[-1]
    elif strategy == 'local_loss':
        emit('for i in range(n):'); op('ry',t='theta[i]',a='i'); lines[-1]='    '+lines[-1]
    elif strategy == 'correlated_parameters':
        emit('for i in range(n):'); op('ry',t='theta[0] if config.variant == "strategy" else theta[i]',a='i'); lines[-1]='    '+lines[-1]
    elif strategy in {'mps','finite_local_depth','ttn','mera','qcnn'}:
        if strategy == 'qcnn':
            emit('for i in range(n):'); op('ry',t='0.2 + 0.1*i',a='i'); lines[-1]='    '+lines[-1]
            emit('active = list(range(n))');emit('block = 0');emit('while len(active) > 1:');emit('    for j in range(0, len(active), 2):');emit('        a, b = active[j], active[j+1]')
            for name,kw in [('haar',dict(index='block',a='a',b='b')),('ry',dict(t='theta[block % len(theta)]',a='a')),('pool',dict(t='0.7 + theta[(block+1) % len(theta)]',a='b',b='a'))]:
                op(name,**kw);lines[-1]='        '+lines[-1]
            emit('        block += 1');emit('    active = active[::2]')
        else:
            if strategy in {'mps','finite_local_depth'}: emit('pairs = [(i, i+1) for i in range(n-1)]')
            else:
                emit('pairs = []');emit('stride = n//2');emit('while stride:');emit('    pairs.extend((a, a+stride) for a in range(0,n,2*stride))');emit('    stride //= 2')
            emit('for block, (a,b) in enumerate(pairs):')
            op('haar',index='block',a='a',b='b');lines[-1]='    '+lines[-1]
            op('ry',t='theta[block % len(theta)]',a='a');lines[-1]='    '+lines[-1]
            if strategy=='mera':
                emit('    if b+1 < n:');op('xx',t='0.3 + theta[(block+1) % len(theta)]',a='b',b='b+1');lines[-1]='        '+lines[-1]
    else:
        if strategy=='low_excitation':op('x',a='0')
        if strategy=='constrained_hva':op('h',a='0')
        emit('for l in range(config.depth):')
        start=len(lines)
        if strategy=='low_excitation':
            emit('if config.variant == "baseline":')
            emit('    for i in range(n):');op('ry',t='theta[2*l*n+n+i]',a='i');lines[-1]='        '+lines[-1]
            emit('for i in range(n-1):')
            for axis in ['xx','yy']:
                op(axis,t='theta[2*l*n+i]',a='i',b='i+1');lines[-1]='    '+lines[-1]
        elif strategy=='permutation_equivariant':
            layer_rot(angle='theta[2*l*n] if config.variant == "strategy" else theta[2*l*n+i]')
            emit('for i in range(n):');emit('    for j in range(i+1,n):');op('zz',t='theta[2*l*n+1] if config.variant == "strategy" else theta[(2*l*n+n+i+j) % len(theta)]',a='i',b='j');lines[-1]='        '+lines[-1]
        elif strategy=='matchgate':
            emit('if config.variant == "baseline":');start_rot=len(lines);layer_rot('ry');lines[start_rot:]=['    '+s for s in lines[start_rot:]]
            layer_rot('rz');emit('for i in range(n-1):');op('xx',t='theta[2*l*n+n+i]',a='i',b='i+1');lines[-1]='    '+lines[-1]
        else:
            if strategy in {'gaussian_init','gaussian_mixture'}:
                emit('for i in range(n-1):');op('cz',a='i',b='i+1');lines[-1]='    '+lines[-1]
                layer_rot('rx');layer_rot('ry', 'theta[2*l*n+n+i]')
            else:
                layer_rot()
                emit('for i in range(n-1):')
                if strategy=='constrained_hva':op('zz',t='theta[2*l*n+n+i]',a='i',b='i+1')
                elif strategy=='nonunital_noise':op('haar',index='l*n+i',a='i',b='i+1')
                else:op('cz',a='i',b='i+1')
                lines[-1]='    '+lines[-1]
            if strategy in {'engineered_dissipation','nonunital_noise'}:
                emit('gamma = 0.15 if config.variant == "strategy" else 0.0')
                emit('for i in range(n):');op('noise',a='i');lines[-1]='    '+lines[-1]
        lines[start:]=['    '+s for s in lines[start:]]
    return '\n'.join('    '+s for s in lines)

DETAILS = {
    'near_identity_encoding': r'''## Encoding versus initialization

This guide controls the **data** gates $E_\epsilon(x)$. Gaussian initialization
controls the starting **trainable** angles, and identity blocks pair trainable
branches. These are different decisions and can be combined only after checking
the actual data-dependent gradient.

For a globally near-identity encoder, the operator-norm telescoping bound is
$\|E_\epsilon(x)-I\|\leq\frac{|\epsilon|}{2}\sum_i|x_i|$.
Small individual angles do not imply a small global distance at growing $n$;
for a fixed global budget, scale epsilon with the feature L1 norm. The bundled
nonzero-gradient example initializes the trainable readout at pi/2, not at the
identity. Initializing that readout at zero can instead create a stationary point.
This reference initialization is deterministic, so its across-seed derivative
variance is zero even though the derivative is nonzero. Read the reported
mean-squared coordinate signal and the analytical bound; a zero variance without
a zero-mean condition is not a barren-plateau diagnosis.

Too small an encoding scale erases distinguishability: the overlap between two
encoded product states is $\prod_i\cos^2(\epsilon(x_i-y_i)/2)$. At epsilon zero,
the model contains no information about the features even though the trainable
readout can have a large gradient. Select scale using both signal and validation
performance; this implementation is classically simulable.

```bash
python -m qml_tricks run near_identity_encoding --framework cirq --encoding-scale 0.1 --features -1 -0.2 0.3 1
```

''',
    'identity_block_init': r'''## Correct independent-copy construction

The native time order of a block is RX(a), CZ, RY(c), RY(-d), reversed CZ,
RX(-b). Initialize b=a and d=c, but store and update all four arrays independently.
Every block then cancels at initialization on any input state. A perturbation
of c without the same perturbation of d is a real trainable direction; an
implementation that recomputes d=c after every update is not this algorithm.

For the boundary readout in the code, the conjugated Y generator involves only
wire zero and its neighbor. Evaluating its commutator with Z on the prepared
|+0...0> state gives the derivative stated above. Uniform independent a0,a1
give E[cos(a0)^2] E[cos(a1)^2]=1/4. Other blocks are identities at this point,
so they do not enlarge this causal cone. Coordinate zero, an X rotation on the
|+> reference, would instead have zero derivative; the diagnostic deliberately
selects coordinate n, the first forward Y rotation.

Use shallow identity blocks with a meaningful input/reference and a compatible
observable. Inserting a universal deep random unitary into each block can restore
the original effective-depth problem. Monitor gradients after updates: cancellation
is an initial condition, not a constraint that should hold for the entire run.

''',
    'natural_gradient': r'''## When QNG is useful, and what it does not establish

Use QNG when gradients are statistically resolvable but Euclidean coordinates
give poorly scaled or redundant directions. It is an optimizer companion, not
an independently proven general barren-plateau remedy. A regularized inverse
can amplify gradient estimation noise; a small eigenvalue is not a source of
missing measurement information. Compare wall time, shots and metric work,
not just the number of iterations.

The circuit prepares product RY/RZ states, applies fixed CZ gates, then fixed H
gates to measure mean X in the premeasurement state using computational-basis
probabilities. The fixed gates preserve FS geometry. With parameters ordered
(alpha_0,...,alpha_n-1,phi_0,...,phi_n-1), the analytic metric is
$g=\frac14\operatorname{diag}(1,...,1,\sin^2\alpha_0,...,\sin^2\alpha_{n-1})$.
The implementations compute this through native statevectors rather than
hardcoding the formula; tests compare against the formula and verify invariance
under parameter-dependent global phase.

The default uses lambda=0.001 and solves the regularized linear system without
forming an explicit inverse. It performs 2P+1 native state evaluations per step,
stores a dense Jacobian and metric, and has dense O(P^3) solve cost. Metric
regularization and condition numbers are reported. At alpha=0 a phase direction
is redundant; regularization handles this singularity without inventing a signal.

On hardware, replace the simulator metric with a measured full, block-diagonal,
or diagonal estimator and count its shots and errors. Computational-basis output
probabilities alone do not determine the full pure-state FS metric. Mixed states
require a suitable mixed-state information geometry. The sampled-loss mode here
retains an exact-simulator metric and is explicitly labeled as such.

PennyLane also supplies [QNGOptimizer](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNGOptimizer.html)
and metric transforms. Its lambda and metric-approximation settings must be
matched when comparing with this dense reference implementation. Classical
natural gradient uses a classical Fisher metric; the pure-state quantum metric
used here is FS, with QFIM=4g.

```bash
python -m qml_tricks run natural_gradient --framework pennylane --metric-regularization 0.001
python -m qml_tricks run natural_gradient --framework qiskit --variant baseline
```

'''
}

records=[]
for sid,title,category,source,locator,scope,use,assumptions,math,boundary in ROWS:
    baseline = 'Same circuit and task, uniformly initialized parameters.'
    if sid=='local_loss':baseline='Global all-zero infidelity versus its local surrogate; same zero set.'
    if sid in {'mps','ttn','mera','qcnn','finite_local_depth'}:baseline='Replace the structured architecture with depth-configured full-register Haar sweeps and the same input and observable. Gate counts differ and must be compared; a short control need not exhibit a plateau.'
    if sid in {'engineered_dissipation','nonunital_noise'}:baseline='Same circuit and observable with damping disabled; not evidence that all noise helps.'
    if sid=='correlated_parameters':baseline='Independent angles versus one spatially shared angle.'
    if sid=='low_excitation':baseline='Add nonconserving Y rotations to the same excitation-preserving circuit; compare task quality and leakage as well as gradients.'
    if sid=='permutation_equivariant':baseline='Replace collective parameters with site/pair-dependent angles, breaking full permutation equivariance.'
    if sid=='matchgate':baseline='Add Y rotations that leave the quadratic-fermion gate family. No plateau is guaranteed at the small default depth.'
    if sid=='mmd_loss':baseline='Use the global delta kernel, whose MMD squared equals the full probability-vector squared distance to the same independent-bit target. It distinguishes more distributions but can concentrate.'
    if sid=='near_identity_encoding':baseline='Use encoding scale 1 instead of encoding_scale, keeping the bounded features and trainable reference identical.'
    if sid=='identity_block_init':baseline='Independently randomize forward and inverse branches instead of matching their initial values.'
    if sid=='natural_gradient':baseline='Euclidean gradient descent; same circuit, objective, initialization, learning rate and gradient estimator. QNG adds separately counted exact metric state evaluations.'
    status='restricted demonstration; inspect guide before transferring the published guarantee'
    if sid in {'local_loss','correlated_parameters'}:status='analytically checked special case'
    if sid=='gaussian_mixture':status='single Z-string G1 special case; full-norm theorem, coordinate diagnostic only'
    if sid=='near_identity_encoding':status='analytically checked product-encoder special case; bounded features required'
    if sid=='identity_block_init':status='analytically checked shallow-block special case; independent copies; initialization only'
    if sid=='natural_gradient':status='optimizer companion; exact-simulator metric; not a proven general plateau remedy'
    record=dict(id=sid,title=title,category=category,source=source_url(source),locator=locator,scope=scope,
                use_case=use,assumptions=assumptions,baseline=baseline,implementation_status=status,
                role='optimization_companion' if sid=='natural_gradient' else 'encoding_design' if sid=='near_identity_encoding' else 'trainability_strategy')
    records.append(record)
    directory=ROOT/'src/qml_tricks/strategies'/sid
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'__init__.py').write_text(f'"""{title}."""\n',encoding='utf-8')
    for sdk,(imports,create,ops,probs,resources) in BACKENDS.items():
        state_body = {
            'qiskit': '''    state = np.asarray(Statevector.from_instruction(build_circuit(theta, config)))
    order = [int(format(i, f"0{config.n_qubits}b")[::-1], 2) for i in range(len(state))]
    return state[order]
''',
            'pennylane': '''    dev = qml.device("default.qubit", wires=config.n_qubits)
    @qml.qnode(dev)
    def circuit():
        build_circuit(theta, config)
        return qml.state()
    return np.asarray(circuit(), dtype=complex)
''',
            'cirq': '''    c = build_circuit(theta, config)
    return cirq.Simulator(dtype=np.complex128).simulate(c, qubit_order=cirq.LineQubit.range(config.n_qubits)).final_state_vector
'''
        }[sdk]
        text=f'''r"""{title}: native {sdk} implementation.

Published result scope: {scope}.
Implementation boundary: {boundary}
"""
{imports}from qml_tricks.fixtures import haar
from qml_tricks.core import RunConfig, experiment

CHANNEL = {sid in {'engineered_dissipation','nonunital_noise'}}


def build_circuit(theta, config):
    """Build the native circuit; PennyLane operations are queued in a QNode."""
    n = config.n_qubits
    {create}
{body(sid,ops)}
    return c


def _normalize(p):
    p = np.maximum(np.real(p), 0)
    return p / p.sum()


def probabilities(theta, config):
    """Exact simulator probabilities in wire-0-first bit order."""
{probs}


def statevector(theta, config):
    """Native pure-state amplitudes; never a hardware-access assumption."""
    if CHANNEL:
        raise ValueError("Pure-state metric is unavailable for channel recipes")
{state_body}


def resources(theta, config):
    """Native logical counts, including preparation and channel operations."""
{resources}


def run(config: RunConfig = RunConfig()):
    return experiment("{sid}", "{sdk}", config)


if __name__ == "__main__":
    import json
    print(json.dumps(run().to_dict(), indent=2))
'''
        (directory/f'{sdk}_impl.py').write_text(text,encoding='utf-8')
    guide=fr'''# {title}

**Family:** {category} · **Result scope:** {scope} · **Code:** {status}

## When to use it

{use}. Check the assumptions below before selecting this method. Use a simpler classical model when its representation and data access already solve the same task economically.

## Mathematical contract

{math}

**Required assumptions:** {assumptions}

The bounded quantity and its ensemble belong to the cited construction. An inverse-polynomial average second moment does not establish that every parameter has a useful gradient, that every initialization succeeds, or that optimization reaches a good minimum. Keep objective normalization fixed while varying system size.

## Logic and proof route

Read **{locator}** in the [primary publication]({source_url(source)}). Identify the surviving operator component or tensor transfer channel, insert the actual state/observable factors, and verify that every factor remains inverse polynomial. Check the parameter convention $R_P(t)=\exp(-itP/2)$ before translating constants. The mathematical statement above is a summary or explicitly derived special case; the source supplies the full theorem and proof.

## Native implementation

{boundary}

1. Choose the circuit size and the data/state preparation that satisfies the input assumptions.
2. Construct the native circuit and initialize with the specified ensemble. The bundled numerical fixtures are reproducible across SDKs.
3. Evaluate the objective and record the actual coordinate or gradient norm being measured.
4. Compare the control with the same size and measurement budget; explain any objective or architecture changes.
5. Train briefly, then recheck the region, symmetry, and data conditions. An initialization guarantee does not automatically survive updates.

```bash
python -m pip install -e ".[qiskit]"
python -m qml_tricks run {sid} --framework qiskit --steps 3
python -m qml_tricks run {sid} --framework pennylane --steps 3
python -m qml_tricks run {sid} --framework cirq --steps 3
python -m qml_tricks sweep {sid} --framework qiskit --sizes 2 4 8 --steps 0 --initializations 16
```

Install the matching extra for PennyLane/Cirq, or use `.[all]`. The three native implementations are distributed as [Qiskit Python files](../../src/qml_tricks/strategies/{sid}/qiskit_impl.py), [PennyLane](../../src/qml_tricks/strategies/{sid}/pennylane_impl.py), and [Cirq](../../src/qml_tricks/strategies/{sid}/cirq_impl.py). These downloads use the installed shared package.

**Control:** {baseline}

Output includes a finite training trace, exact coordinate-gradient variance, measurement-noise variance, resource counts, and the implementation boundary. No monotonic improvement or asymptotic separation is expected from every small run. `--shots 4096` samples the exact native output distribution on the classical host; it does not submit a hardware job.

## Learning-task transfer

For a linear empirical objective $C(\theta)=\sum_s w_s\mathrm{{Tr}}[O_s U_\theta\rho_s U_\theta^\dagger]$, derive the bound for this entire sum. Label-weighted state differences and gradient cancellations can invalidate a per-example statement. A nonlinear cross-entropy or squared loss requires its own chain-rule and signal analysis. The default executable is a small state/moment objective rather than an accuracy benchmark on a real dataset. Classical features can supply preparation angles; quantum-native inputs require copies of the actual state and appropriate purity/projection factors. Do not assume access to their full amplitudes.

## Space and computational-advantage audit

The native register uses $n$ logical qubits and no added ancillas in this example. Exact pure-state simulation stores $16\cdot2^n$ bytes for one complex128 statevector; a dense mixed-state matrix stores $16\cdot4^n$ bytes before temporary arrays. The executable reports both estimates and identifies the simulator regime. Its parameter array is polynomial size; some architectures deliberately contain unused coordinates, so inspect which gates depend on each parameter.

These storage comparisons do not prove a quantum memory separation. Account for preparation, repeat copies, precision, loading classical features, gates, and measured outputs. Compare tensor-network, symmetry-sector, free-fermion, causal-cone, and quantum-data-assisted classical surrogates when applicable. The small bundled instances are classically simulated by construction. **No end-to-end quantum advantage is established by this recipe.**

## Failure modes, composition, and checks

Check zero commutators, irrelevant coordinates, maximally mixed inputs, cancellation between data points, oversized symmetry sectors, and changes of loss support. The designated diagnostic coordinate may have zero gradient even if the published full gradient norm is nonzero. Exact simulation tests wiring and algebra; repeated seeds cannot prove an asymptotic theorem.

Combine methods only after intersecting their state, ensemble, observable, and noise assumptions. Adding parameter tying can destroy independence; adding noise changes a unitary theorem; changing a loss can change its optimum. Use independent derivative checks and cross-SDK probability comparisons before interpreting a training curve.

## Evidence and maintenance

- Primary publication: [{source}]({source_url(source)}).
- Result locator: {locator}.
- Literature cutoff: 2026-11-09. See the coverage ledger for search limitations and exclusions.
- See [foundations](../foundations.md) for definitions, measurement accounting, and statistical limitations.
'''
    # Backslashes in ordinary prose must survive Python's escape processing.
    guide=guide.replace('\x08','\\b').replace('\x0c','\\f').replace('\x0d','\\r').replace('\x09','\\t')
    if sid in DETAILS:
        guide=guide.replace('## Logic and proof route', DETAILS[sid]+'## Logic and proof route')
    target=ROOT/'docs/strategies'/f'{sid}.md';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(guide,encoding='utf-8')
(ROOT/'src/qml_tricks/catalog.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print(f'Wrote {len(records)} guides and {3*len(records)} native modules.')
