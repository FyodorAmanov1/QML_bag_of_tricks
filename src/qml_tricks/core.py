"""Framework-independent experiments and uncertainty-aware diagnostics."""
from dataclasses import dataclass, asdict, replace
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import time
import numpy as np
from .fixtures import initial_parameters, loss_from_probabilities
from .geometry import fubini_study_metric, natural_gradient_direction


def catalog():
    return json.loads(Path(__file__).with_name("catalog.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RunConfig:
    n_qubits: int = 4
    depth: int = 2
    seed: int = 7
    shots: int | None = None
    steps: int = 3
    variant: str = "strategy"
    learning_rate: float = 0.1
    initializations: int = 4
    repetitions: int = 2
    encoding_scale: float = 0.1
    features: tuple[float, ...] | None = None
    metric_regularization: float = 1e-3

    def validate(self, strategy):
        if not np.isfinite(self.encoding_scale) or not 0 <= self.encoding_scale < np.pi/2:
            raise ValueError("encoding_scale must be finite and in [0, pi/2)")
        if self.features is not None and (len(self.features) != self.n_qubits or not np.all(np.isfinite(self.features)) or np.max(np.abs(self.features)) > 1):
            raise ValueError("features must have n_qubits finite values in [-1, 1]")
        if not np.isfinite(self.metric_regularization) or self.metric_regularization <= 0:
            raise ValueError("metric_regularization must be finite and positive")
        if not 2 <= self.n_qubits <= 12 or not 1 <= self.depth <= 32:
            raise ValueError("Use 2..12 qubits and 1..32 layers; expensive runs require editing the explicit resource guard.")
        if strategy in {"qcnn", "ttn", "mera"} and self.n_qubits & (self.n_qubits - 1):
            raise ValueError("This hierarchical implementation requires a power-of-two qubit count.")
        if self.shots is not None and self.shots < 2:
            raise ValueError("shots must be None (exact) or at least 2")
        if self.steps < 0 or self.initializations < 2 or self.repetitions < 2 or self.seed < 0:
            raise ValueError("Require nonnegative steps/seed and at least two initializations/repetitions")
        if self.variant not in {"strategy", "baseline"} or not 0 < self.learning_rate <= 1:
            raise ValueError("Invalid variant or learning rate")
        if strategy in {"engineered_dissipation", "nonunital_noise"} and self.n_qubits > 8:
            raise ValueError("Density-matrix recipes are limited to 8 qubits.")


@dataclass
class ExperimentResult:
    strategy: str
    framework: str
    configuration: dict
    loss: float
    training_trace: list
    gradient_statistics: dict
    resources: dict
    assumption_checks: dict
    provenance: dict

    def to_dict(self):
        return asdict(self)


def module(strategy, framework):
    if strategy not in {r["id"] for r in catalog()}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if framework not in {"qiskit", "pennylane", "cirq"}:
        raise ValueError(f"Unknown framework: {framework}")
    return import_module(f"qml_tricks.strategies.{strategy}.{framework}_impl")


def gradient(evaluate, theta):
    """Independent central derivative. Valid for tied and non-Pauli parameters.

    Used only on exact simulator objectives. Finite-shot gradients below use
    finite differences with a larger, disclosed step; they are biased estimates.
    """
    return finite_difference(evaluate, theta, 1e-5)


def finite_difference(evaluate, theta, step):
    out = np.empty(len(theta))
    for j in range(len(theta)):
        delta = np.zeros_like(theta); delta[j] = step
        out[j] = (evaluate(theta + delta) - evaluate(theta - delta)) / (2 * step)
    return out


def experiment(strategy, framework, config):
    config.validate(strategy)
    sdk = module(strategy, framework)
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed + 100003)
    evaluations = 0
    sampled_evaluations = 0
    metric_state_evaluations = 0
    metric_trace = []
    active_config = config

    def evaluate(theta, sample=True):
        nonlocal evaluations, sampled_evaluations
        probabilities = sdk.probabilities(theta, active_config)
        evaluations += 1
        if sample and config.shots is not None:
            sampled_evaluations += 1
            # Classical sampling of the SDK's exact output distribution is an
            # explicitly simulated finite-shot experiment, not a hardware job.
            probabilities = rng.multinomial(config.shots, probabilities) / config.shots
        return loss_from_probabilities(probabilities, strategy, config.n_qubits, config.variant)

    theta = initial_parameters(strategy, config.n_qubits, config.depth, config.seed, config.variant)
    trace = [evaluate(theta)]
    differentiation_step = 0.1 if config.shots else 1e-5
    for _ in range(config.steps):
        g = finite_difference(evaluate, theta, differentiation_step)
        direction = g
        if strategy == "natural_gradient" and config.variant == "strategy":
            metric = fubini_study_metric(lambda t: sdk.statevector(t, config), theta)
            metric_state_evaluations += 2*len(theta) + 1
            direction = natural_gradient_direction(metric, g, config.metric_regularization)
            eigenvalues = np.linalg.eigvalsh(metric)
            metric_trace.append({"minimum_eigenvalue": float(eigenvalues[0]),
                                 "maximum_eigenvalue": float(eigenvalues[-1]),
                                 "regularized_condition_number": float(np.linalg.cond(metric + config.metric_regularization*np.eye(len(theta))))})
        theta = theta - config.learning_rate * direction
        if strategy == "constrained_hva" and config.variant == "strategy":
            # Preserve the conservative nonnegative total-time region.
            theta = np.clip(theta, 0, 0.01 / (config.n_qubits * len(theta)))
        trace.append(evaluate(theta))
    exact_gradients, estimates = [], []
    # One designated coordinate, not an inferred full Riemannian norm.
    index = (2 * (config.depth - 1) * config.n_qubits if strategy == "nonunital_noise"
             else config.n_qubits if strategy == "identity_block_init" else 0)
    for k in range(config.initializations):
        # Resample fixed Haar factors too, not just trainable angles. Otherwise
        # Haar-dressed zero-angle initializations would repeat the same circuit.
        active_config = replace(config, seed=config.seed + k)
        t = initial_parameters(strategy, config.n_qubits, config.depth, config.seed + k, config.variant)
        delta = np.zeros_like(t); delta[index] = differentiation_step
        exact_gradients.append((evaluate(t + delta, False) - evaluate(t - delta, False)) / (2 * differentiation_step))
        estimates.append([(evaluate(t + delta) - evaluate(t - delta)) / (2 * differentiation_step)
                          for _ in range(config.repetitions)])
    active_config = config
    estimates = np.asarray(estimates)
    within = float(np.var(estimates, axis=1, ddof=1).mean())
    between = float(np.var(estimates.mean(axis=1), ddof=1))
    corrected = between - within / config.repetitions
    boot_rng = np.random.default_rng(config.seed + 991)
    boot = np.var(boot_rng.choice(exact_gradients, size=(400, len(exact_gradients))), axis=1, ddof=1)
    row = next(r for r in catalog() if r["id"] == strategy)
    stats = {"coordinate": index, "exact_coordinate_variance": float(np.var(exact_gradients, ddof=1)),
             "mean_squared_exact_coordinate": float(np.mean(np.square(exact_gradients))),
             "between_initialization_variance_of_means": between,
             "within_initialization_sampling_variance": within,
             "noise_corrected_variance": corrected,
             "signal_status": "resolved_on_exact_simulator" if config.shots is None else "finite_sample_estimate",
             "difference_step": differentiation_step, "gradient_estimator": "central_finite_difference",
             "finite_difference_bias": "O(h^2) for a smooth loss; not an unbiased parameter-shift estimate",
             "initializations": config.initializations, "repetitions": config.repetitions,
             "bootstrap_exact_variance_interval_95": np.quantile(boot, [0.025, 0.975]).tolist(),
             "interval_caveat": "percentile bootstrap; small ensembles give unreliable coverage",
             "asymptotic_conclusion": "none"}
    if strategy == "natural_gradient":
        stats["optimizer"] = {"name": "quantum_natural_gradient" if config.variant == "strategy" else "euclidean_gradient_descent",
                              "metric_convention": "Fubini-Study; QFIM = 4g",
                              "regularization": config.metric_regularization,
                              "metric_estimator": "exact statevector central differences" if metric_trace else "not evaluated",
                              "metric_trace": metric_trace,
                              "barren_plateau_avoidance_guarantee": False}
    circuits = sdk.resources(theta, config)
    circuits.update({"objective_evaluations": evaluations, "sampled_objective_evaluations": sampled_evaluations,
                     "simulated_measurement_shots": sampled_evaluations * (config.shots or 0),
                     "elapsed_seconds": time.perf_counter() - started,
                     "statevector_bytes": 16 * 2**config.n_qubits,
                     "density_matrix_bytes": 16 * 4**config.n_qubits,
                     "classical_parameter_bytes": theta.nbytes,
                     "stored_parameter_count": len(theta),
                     "simulator": "density_matrix" if strategy in {"engineered_dissipation", "nonunital_noise"} else "statevector",
                     "hardware_executed": False})
    if strategy == "natural_gradient":
        circuits.update({"metric_state_evaluations": metric_state_evaluations,
                         "metric_matrix_bytes": 8*len(theta)**2 if metric_trace else 0,
                         "metric_jacobian_bytes": 16*(2**config.n_qubits)*len(theta) if metric_trace else 0,
                         "metric_hardware_measurements": "not implemented; simulator-only metric"})
    return ExperimentResult(strategy, framework, asdict(config), trace[-1], trace, stats, circuits,
        {"published_result": row["source"], "guarantee_scope": row["scope"],
         "role": row["role"],
         "implementation_status": row["implementation_status"],
         "automatic_theorem_certification": False,
         "assumptions": row["assumptions"], "baseline_changes": row["baseline"]},
        {"sdk_version": version({"cirq": "cirq-core"}.get(framework, framework)),
         "numpy_version": version("numpy"), "literature_cutoff": "2026-11-09"})
