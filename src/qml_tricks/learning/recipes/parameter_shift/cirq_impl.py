"""Parameter-shift differentiation: native cirq preparation and shared mathematical recipe."""
from qml_tricks.learning.backends.cirq import state
from qml_tricks.learning.demos import run_recipe

def run(seed=7):
    return run_recipe("parameter_shift", "cirq", seed)

if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
