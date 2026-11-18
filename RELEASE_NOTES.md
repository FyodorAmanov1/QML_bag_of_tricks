# QML Bag of Tricks 0.2.0 — release candidate

## Published-evidence workflow

- Peer-reviewed findings and documented author implementations support inclusion
  and use without requiring an independent local rerun of every paper.
- Project planning and case setup default to `intent="adapt"`; `replicate` is
  an explicit optional track. New-project performance still requires evaluation.
- Selector matches use `published_candidates` / `evidence_matched_published_candidate`
  status labels and include publication and `implementation_access` records.
- `require_author_code=true` restricts matching to the five pinned inspected
  author-source routes; it does not certify reuse terms or a local training result.
- Existing local-reproduction flags, benchmark outcomes and scientific limits
  remain unchanged. The evidence guide distinguishes source, archive, component
  and theory access rather than treating every case as an unfinished local replay.

## License and attribution update

- Project code uses Apache-2.0 across both packages.
- Added NOTICE and preserved third-party licenses and attribution.
- Citation and package metadata identify Fyodor Amanov and project contributors.
- Added a citation guide; a DOI awaits a real archived release.
- Historical reports and scientific evidence labels remain intact.
  This update does not create a tagged scientific release.

## Unreleased unified entry point

One root distribution now provides `qml-utility` and the existing `qml-tricks`
CLI. Start with `qml-utility setup` for published-case protocols and the canonical
QNG, local-cost, identity-block and weak-encoding strategy checks. Supervised QNG
reuses the main geometry implementation. Utility documentation is included in
the same website build from its canonical files. Default CI/test discovery covers
both layers. Original studies and research implementations remain available;
this local integration does not publish a release or change source visibility.

## Earlier release-candidate contents

Reproduced QML applications, trainable methods and agent-friendly implementation
recipes in Qiskit, PennyLane and Cirq.

The application entry point is motivated by two peer-reviewed classical-input
studies: the alloy replay reaches 89.12% accuracy versus 87.73% for its best
original comparator, while reservoir equalization reaches 0.313% symbol error
versus 4.437% for direct memoryless logistic regression. These are bounded
numerical reproductions of the original comparisons; stronger tuned classical
models lead in the follow-up comparisons. See the
[motivation guide](docs/why_qml.md) and [full reproduction report](docs/reproductions/results.md).

- Three executed project notebooks: real-data classification, real-data regression
  and offline quantum-feature distillation with a classical exported student.
- A bounded real-data study: three datasets, ten repeated splits, eight models,
  three validation candidates per model and 240 untouched-test evaluations.
- Five agent-produced project solutions, independently re-executed and graded.
- Bounded numerical reproductions of two classical-input application papers:
  alloy-phase classification and wireless channel equalization, with pinned
  author artifacts, fresh circuit/dynamics runs and stronger classical controls.
- Six additional peer-reviewed application guides: satellite imagery,
  molecular carcinogenicity, hyperparameter-search resource reduction,
  hybrid entity matching, multiclass quantum kernels and QVC attack transfer.
  Includes a structured evidence catalog, result locators, artifact readiness
  and agent benchmark contracts. These are reported leads, not local reproductions.
- Three runnable reduced application models with independent Qiskit, PennyLane
  and Cirq construction, nine executed adaptation runs, classical controls,
  training/query costs and quantum white-box attack diagnostics. Their outcomes
  remain separate from the publications' successes.
- Two application prototypes with 280 configurations and 1,720 base evaluations:
  periodic calibration and late-query quantum-state learning. Includes tuned
  regularized controls, optimal local-record inference and controls that remove
  the apparent advantage.
- 19 circuit-strategy guides and 14 learning/estimation guides with Qiskit,
  PennyLane and Cirq entrypoints, scientific sources and evidence boundaries.
- Release artwork, a clearer entry point, package artifacts and a review packet.
- A redesigned README and responsive application showcase, with evidence
  filters, a project selector, SDK commands and source-derived comparison data.
  The Pages workflow produces a private downloadable preview by default;
  public documentation publication is a separate option.

The earlier tabular study remains available as a detailed benchmark report.
All source reports retain failures, warnings, uncertainty and costs. Application
gains are scoped to their stated comparators and data-access assumptions.

The new state-learning result is a conditional two-copy acquisition benefit on
synthetic quantum inputs. The periodic feature benefit has an identical cheap
classical Fourier implementation. Neither establishes a practical runtime gain
or predictive superiority on general real-world classical datasets.

Owner scientific review is pending. The GitHub repository is private for this
review; the documentation website is public on GitHub Pages with owner approval.
Public source checkout and Colab installation access remain pending. This candidate is
not presented as an already-published GitHub or PyPI release.
