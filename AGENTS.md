# Repository instructions

- Preserve the user's peer-reviewed-only evidence policy. Separate proof scope
  from implementation scope; never promote a demonstration by changing its label.
- Read the relevant strategy guide and primary result before changing a circuit.
- Update scripts/assemble.py for generated strategies, regenerate, then inspect.
- Keep native SDK construction independent. Share numerical fixtures and statistics.
- Test probability ordering, rotation conventions, gradients, and relevant structure.
- Record loading, readout, shots, classical simulation, and memory assumptions.
- Run pytest and strict MkDocs build for changes to executable recipes or docs.
- Do not infer full training, generalization, or advantage from an initialization bound.
- Kipu's intended surrogate is offline feature-map distillation. Keep it distinct
  from density-embedding approximation and Nyström. Native demos are reference
  teachers, not replicas of proprietary DQFE. Gate export on held-out fidelity.
- Preserve the explicitly requested encoding-design and QNG optimizer companions.
  Their role labels distinguish them from claims of general plateau avoidance.
- Learning guides/backends/entrypoints are authored in scripts/assemble_learning.py.
  Keep the user-requested classical pretraining and simulator warm starts labeled
  as engineering companions. Kernel surrogates need held-out error and zero-query
  inference checks; Nyström still requires quantum queries on new inputs.
