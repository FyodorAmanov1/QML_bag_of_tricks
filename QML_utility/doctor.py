"""Read-only environment and optional-SDK checks; does not install anything."""
import argparse
import importlib.metadata
import json
import platform
import sys


def diagnose():
    dependencies = {}
    for name in ('numpy','scipy','scikit-learn','joblib','qiskit','pennylane','cirq-core'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return dict(python=platform.python_version(),supported_python=sys.version_info>=(3,12),
                dependencies=dependencies,
                core_ready=all(dependencies[name] is not None for name in ('numpy','scipy','scikit-learn','joblib')),
                available_frameworks=[sdk for sdk,dist in [('qiskit','qiskit'),('pennylane','pennylane'),('cirq','cirq-core')] if dependencies[dist]],
                scope='Package metadata only; run quickstart or tests to verify native execution',
                installation='From the unified repository root: python -m pip install -e ".[qiskit]" (or pennylane/cirq/all)')


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(diagnose(),indent=2))


if __name__ == '__main__':
    main()
