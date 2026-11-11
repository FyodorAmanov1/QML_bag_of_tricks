# QML Bag of Tricks 0.2.0 — release candidate

Agent-friendly QML recipes with explicit assumptions and classical comparisons.

- Three executed project notebooks: real-data classification, real-data regression
  and offline quantum-feature distillation with a classical exported student.
- A bounded real-data study: three datasets, ten repeated splits, eight models,
  three validation candidates per model and 240 untouched-test evaluations.
- Five agent-produced project solutions, independently re-executed and graded.
- Two application prototypes with 280 configurations and 1,720 base evaluations:
  periodic calibration and late-query quantum-state learning. Includes tuned
  regularized controls, optimal local-record inference and controls that remove
  the apparent advantage.
- 19 circuit-strategy guides and 14 learning/estimation guides with Qiskit,
  PennyLane and Cirq entrypoints, scientific sources and evidence boundaries.
- Release artwork, a clearer entry point, package artifacts and a review packet.

Observed real-data results favor the tested classical configurations over these
small quantum-feature configurations. They do not establish conclusions about
all QML, quantum advantage or production readiness. Source reports retain failures,
warnings, uncertainty and costs rather than selecting only favorable cases.

The new state-learning result is a conditional two-copy acquisition benefit on
synthetic quantum inputs. The periodic feature benefit has an identical cheap
classical Fourier implementation. Neither establishes a practical runtime gain
or predictive superiority on general real-world classical datasets.

Owner scientific review is pending. The GitHub repository is private for this
review; public documentation and Colab access remain pending. This candidate is
not presented as an already-published GitHub or PyPI release.
