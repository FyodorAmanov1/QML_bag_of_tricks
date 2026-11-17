---
template: agents.html
title: Use with your coding agent
---

# Use QML Bag of Tricks with your coding agent

Give your agent a task, a dataset description, a compute budget and one of the
copyable briefs below. The same workflow applies to any coding agent that can
read files or browse documentation and run Python in your project environment.

The entry point is now [QML Utility](utility/unified_setup.md), shipped in the
same distribution as the research library. Start with `qml-utility setup`; it
exposes published cases, QNG, local costs, identity-block initialization and weak
encoding. It runs canonical research implementations rather than disconnected
copies. See [the utility project brief](utility/project_agent_brief.md).

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

1. Start with [QML Utility setup](utility/unified_setup.md),
   [case selection](utility/case_selection.md) and the
   [practical project workflow](utility/practical_workflow.md). Use the
   [complete documentation directory](explore.md) and
   [method selector](method_selector.md) for research depth.
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
python -m pip install -e ".[qiskit]"
python -m QML_utility doctor
python -m QML_utility setup --framework qiskit --output QML_utility/results/local/agent_setup --check-essentials
python -m QML_utility quickstart --framework qiskit --output QML_utility/results/local/agent_first_run
```

Use Python 3.12. Install `.[pennylane]` or `.[cirq]` and use the matching
`--framework` on setup and quickstart to choose another SDK. Setup prepares a
contract and executes circuit diagnostics; quickstart fits a small numeric
integration example against classical controls. Neither establishes an advantage
on your dataset. Use fresh output directories. For custom data, follow the
[Utility workflow](utility/practical_workflow.md) and
[evaluation guide](utility/evaluation.md), with appropriate splits. The original
[model APIs](use_cases/models.md#use-your-own-data),
[learning workflows](learning/index.md) and `qml-tricks` CLI remain available.

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
