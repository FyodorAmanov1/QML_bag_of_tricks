"""Classical shadows for local quantum features: native cirq preparation and shared mathematical recipe."""
from qml_tricks.learning.backends.cirq import state
from qml_tricks.learning.demos import run_recipe

def run(seed=7):
    return run_recipe("classical_shadows", "cirq", seed)

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
