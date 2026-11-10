"""Record only the installed dependency closure, not unrelated host packages.

Run in the validated reference environment. The result is platform specific;
use pyproject extras to resolve on other platforms.
"""
from importlib.metadata import distribution, version
from pathlib import Path
import platform
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

roots = ["numpy", "scipy", "qiskit", "qiskit-aer", "pennylane", "cirq-core",
         "pytest", "build", "mkdocs-material", "scikit-learn", "nbformat",
         "nbclient", "ipykernel", "matplotlib"]
seen = set()
queue = list(roots)
while queue:
    name = canonicalize_name(queue.pop())
    if name in seen:
        continue
    seen.add(name)
    for value in distribution(name).requires or []:
        dep = Requirement(value)
        if dep.marker is None or dep.marker.evaluate({"extra": ""}):
            queue.append(dep.name)
header = f"# Validated dependency closure: {platform.system()}, Python {platform.python_version()}\n"
header += "# Install after creating a fresh virtual environment. Other platforms: resolve pyproject extras.\n"
lines = [f"{name}=={version(name)}" for name in sorted(seen)]
Path("requirements-reference.txt").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
print(f"Recorded {len(lines)} dependency pins.")
