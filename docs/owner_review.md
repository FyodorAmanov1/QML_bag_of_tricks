# Owner review packet

**Status: pending owner scientific review.** Automated checks and agent task
replays are complete records of software behavior, not a substitute for your
review. No external QML-engineer review has been requested.

Review these artifacts in order:

1. [Three walkthroughs](tutorials.md): terminology, explanations, task usefulness,
   expected outputs and adaptation guidance.
2. [Real-data study](study.md): dataset selection, target scaling, candidate grids,
   split isolation, uncertainty interpretation and complete negative findings.
   Also review the [application comparisons](applications/results.md): late-query
   access, matched copy counts, restricted local measurements, the optimal
   likelihood control, classical Fourier equivalence and simulated-noise scope.
   The [paper reproductions](reproductions/results.md) separately identify
   archived results, fresh replay, author preprocessing and stronger controls.
   Review the [additional published applications](use_cases/index.md) against
   their primary tables: these remain reported leads, with no local training
   runs. Check the best-classical-comparator selection, the one-of-ten molecular
   endpoint context, epoch-versus-wall-time distinction and artifact readiness.
3. [Agent implementation evaluation](agent_evaluation.md): prompts, acceptance
   checks, task restrictions, self-reported logs and limits of inference.
4. [Evidence coverage](coverage.md): theorem conditions, proof-to-code gaps and
   the distinction between a formal result and an engineering companion.
5. [Kipu workflow](learning/kipu_workflow.md): reference-teacher scope, supplied
   measured-pair contract, high-fidelity gates and separate task-quality criteria.

Before approving a public scientific claim, check that its precise objective,
ensemble, generator conventions, state projections, noise and resource model
match the cited result. Identify any unverified transfer to a full learning loss.
Check every headline against its raw report. Assess whether proposed use cases
fit the small example rather than generalizing from a method name.

Record your review decisions in a follow-up commit or release notes. This packet
intentionally has no fabricated reviewer name, approval date or sign-off.
