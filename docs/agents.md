---
template: agents.html
title: Use with your coding agent
---

# Use QML Bag of Tricks with your coding agent

Give your agent a task, a dataset description, a compute budget and one of the
copyable briefs below. The same workflow applies to any coding agent that can
read files or browse documentation and run Python in your project environment.

| Access route | What your agent can do | What it needs |
|---|---|---|
| **Public website** | Read methods, follow primary citations, inspect linked native source files and adapt a method in your project | Browsing tools; local file writing and Python execution for implementation and benchmarking |
| **Local checkout** | Read `AGENTS.md`, import the package, run recipes, reuse tests and save benchmark artifacts | Authorized private-repository access, Python 3.12 and the selected SDK dependencies |

The website is a static documentation library. It does not execute your agent,
train models, run QPU jobs or provide a hosted inference API. Downloaded reference
files can import shared package helpers; a source download alone is not a complete
installation. For website-only access, inspect dependencies and adapt the needed
method into your own project, or obtain authorized checkout access.

## Find the right technical material

1. Browse the [complete documentation directory](explore.md), then use the
   [method selector](method_selector.md) to choose a candidate.
2. Read its assumptions and mathematical contract. In particular, inspect
   [near-identity encoding](strategies/near_identity_encoding.md),
   [local objectives](strategies/local_loss.md) and
   [quantum natural gradient](strategies/natural_gradient.md) when relevant.
3. Use [learning workflows](learning/index.md) and [application APIs](use_cases/models.md)
   to connect the construction to a dataset. For published application replays,
   start at [paper reproductions](reproductions/index.md).
4. Open the guide's linked SDK source. Website source links download ordinary
   Python files; they are reference implementations with the scope stated in the guide.

A compact [plain-text entry point](llms.txt) and downloadable
[website brief](assets/agent-website.txt) / [local brief](assets/agent-local.txt)
are available for agents that prefer text over a page layout.

## Set up a local checkout

Cloning currently requires access to the **private** source repository. The
public website does not grant that access. In an authorized shell:

```bash
git clone https://github.com/FyodorAmanov1/QML_bag_of_tricks.git
cd QML_bag_of_tricks
python -m venv .venv
# Activate .venv using your shell before the following commands.
python -m pip install -e ".[qiskit,learning]"
python -m qml_tricks list
python -m qml_tricks run local_loss --framework qiskit --output results/local/local_loss.json
python -m qml_tricks.learning.case_models --case multiclass_kernel --framework qiskit --output results/local/first_run.json
```

Use Python 3.12. Install `.[pennylane,learning]` or `.[cirq,learning]` and use the
matching `--framework` to choose another SDK. The `local_loss` run is a circuit
diagnostic; the multiclass run is a small application reference. Neither is a
benchmark of your dataset. For custom data, follow the
[model APIs](use_cases/models.md#use-your-own-data) or the
[CSV workflow](learning/index.md#csv-workflow), and design appropriate splits.

## What to ask your agent to deliver

Ask for a runnable training/evaluation script, environment and seed records,
split identifiers, held-out predictions, a metrics report and model/resource
artifacts where supported. Keep train-only preprocessing and validation-only
selection, and include tuned regularized classical controls with the same data
access. For grouped or temporal data, use a suitable splitter before fitting.

Report loading, readout, shots, simulation time and memory; preserve classical
wins and failed configurations. Do not turn an initialization result into a
training guarantee, or a simulator comparison into a hardware advantage.
For an offline quantum-feature surrogate, train the teacher first, gate export
on held-out fidelity and task quality, and check zero-query student inference.

The [five executed agent tasks](agent_evaluation.md) illustrate concrete outputs
and acceptance checks. They are a bounded integration evaluation, not an estimate
of general agent success.

## Copy a project brief

Choose the access route your agent actually has. Fill in the bracketed fields
before pasting it into your agent's conversation.
