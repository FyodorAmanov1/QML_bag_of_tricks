# Launch material

Use this draft only after owner review and a public repository/docs destination
are complete. Replace the placeholders with working URLs. No post has been sent.

## LinkedIn draft

I built QML Bag of Tricks to help ML engineers and their coding agents try
quantum machine learning with clear assumptions and measurable comparisons.

It includes Qiskit, PennyLane and Cirq implementations, three executed project
notebooks, and a small real-data benchmark with classical controls. Methods cover
initialization, local costs, structured circuits, quantum natural gradient,
kernels and classical deployment surrogates. Each guide explains when the method
fits and where its guarantees stop.

The first real-data study is useful precisely because it reports the complete
results: the tested classical configurations outperform these small quantum
feature models. The repository helps you reproduce and investigate that outcome,
as well as decide what to try next.

Application-specific tests also show where structure helps: a two-copy quantum
state-learning protocol improves sample efficiency for late queries. The report
includes tuned regularized models, optimal inference for the tested classical
records, and a query-known control that removes the gap. Periodic calibration
shows a useful representation benefit that a classical Fourier model also gets.

Repository: [PUBLIC REPOSITORY URL]
Start with a notebook: [PUBLIC WALKTHROUGH URL]

If you try it on a dataset, share the settings, baselines and results in an issue.
Contributions that improve evidence, usability or reproducibility are welcome.

## Launch checklist

- Finish the owner scientific review and record decisions in a reviewed commit.
- Check every public link and run the notebooks from the public checkout.
- Set a concise repository description, QML/SDK topics, license and social image.
- Publish a reviewed release and link the walkthroughs prominently.
- Share one notebook demonstration and its actual comparison, then invite a
  concrete contribution such as a new dataset report or a documented failure.
- Follow up with fixes and reproducible community results. Track whether people
  can complete a project, alongside stars, issues and repeat contributors.

The existing LinkedIn audience is a useful distribution channel; it does not
guarantee stars or adoption. Give readers a short path to an executable example
and a clear way to report what happened.
