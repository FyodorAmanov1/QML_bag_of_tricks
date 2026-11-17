"""JSON-producing CLI. Install only the SDK extra you intend to run."""
import argparse
import json
from pathlib import Path
from .core import RunConfig, catalog, experiment


def main():
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='benchmark':
        from .learning.benchmark import main as benchmark_main
        return benchmark_main(sys.argv[2:])
    if len(sys.argv)>1 and sys.argv[1]=='learning-demo':
        from .learning.demos import run_recipe
        p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--framework',choices=['qiskit','pennylane','cirq'],default='qiskit');p.add_argument('--seed',type=int,default=7)
        args=p.parse_args(sys.argv[2:]);print(json.dumps(run_recipe(args.method,args.framework,args.seed),indent=2));return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("run", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("strategy", choices=[x["id"] for x in catalog()])
        p.add_argument("--framework", required=True, choices=["qiskit", "pennylane", "cirq"])
        p.add_argument("--n-qubits", type=int, default=4)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--shots", type=int)
        p.add_argument("--steps", type=int, default=3)
        p.add_argument("--initializations", type=int, default=4)
        p.add_argument("--repetitions", type=int, default=2)
        p.add_argument("--variant", choices=["strategy", "baseline"], default="strategy")
        p.add_argument("--encoding-scale", type=float, default=0.1)
        p.add_argument("--features", type=float, nargs="+")
        p.add_argument("--metric-regularization", type=float, default=1e-3)
        p.add_argument("--output", type=Path)
        if name == "sweep":
            p.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    if args.command == "list":
        print(json.dumps(catalog(), indent=2)); return
    rows = []
    for n in args.sizes if args.command == "sweep" else [args.n_qubits]:
        for variant in ("strategy", "baseline") if args.command == "sweep" else [args.variant]:
            cfg = RunConfig(n_qubits=n, depth=args.depth, seed=args.seed, shots=args.shots,
                            steps=args.steps, initializations=args.initializations,
                            repetitions=args.repetitions, variant=variant,
                            encoding_scale=args.encoding_scale,
                            features=tuple(args.features) if args.features is not None else None,
                            metric_regularization=args.metric_regularization)
            rows.append(experiment(args.strategy, args.framework, cfg).to_dict())
    payload = rows if args.command == "sweep" else rows[0]
    result = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
