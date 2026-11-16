# Admission and promotion

The unit of evidence is a **task–data–access–comparator–metric tuple**, not a model name.

1. Admit a success only with a peer-reviewed primary publication and a result locator.
2. Record whether it is numerical, hardware experimental, or a conditional theorem.
3. Record inputs, targets, data-generation constraints, preprocessing, splits,
   model, training schedule, baseline selection and uncertainty. Use `unknown`
   when a necessary artifact is unavailable.
4. Distinguish predictive quality, new-task plasticity, old-task retention,
   samples/copies, trainable parameters, memory and end-to-end compute.
5. State the proposed mechanism separately from conditions proven sufficient.
   Frequency support is not arbitrary Fourier-coefficient freedom; compact
   parameterization alone does not eliminate barren plateaus.
6. Preserve every prespecified dataset, setting and outcome. Add strong classical
   controls; do not tune until a favorable test result appears.
7. Locally promote a published claim only after recovering original data and
   protocol, executing the target scale, independently recomputing metrics,
   checking group/time leakage, and retaining all seeds and compute budgets.
8. Generalize a recommendation only after separate unseen tasks support it.
9. Scope a non-winning result to its data, preprocessing, model/training settings,
   baseline access and budget. It is not a proof that the method cannot help
   elsewhere. Better training or different data geometry is a prospective
   hypothesis; test it without retroactively changing the original outcome.

No real-classical-data paper win has passed step 7 in this prototype. A native
circuit identity test is valuable but cannot satisfy that gate. The late-query
audit is a separate restricted result, not a new broad advantage claim.

Distinguish a finite experimental non-win from a mathematical constraint. More
training may improve an optimization-limited fit; it cannot recover information
already discarded by PCA, change an exact classical kernel equivalence, or remove
an output-size lower bound. See [contextual results](CONTEXTUAL_RESULTS.md).
