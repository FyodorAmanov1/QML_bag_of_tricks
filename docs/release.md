# Review and release

Version 0.2.0 is a local release candidate. Scientific review belongs to the
repository owner and is pending; no external QML-engineer review is required.
The [review packet](owner_review.md) links the evidence and checks.

## Reproduce and package

From a cloned checkout with Python 3.12:

```bash
python -m pip install -e ".[all,learning,notebooks,dev,docs]"
python scripts/run_notebooks.py
python -m pytest -q
python scripts/grade_agent_tasks.py
python -m mkdocs build --strict
python -m build
python scripts/prepare_release.py
```

The ignored `release/` directory contains wheel/source distributions, a static
documentation ZIP, the social preview, validation record, release notes and a
readiness manifest. Preparing it does not commit, push or publish anything.

## Configure the public destination

The repository is hosted privately at
[FyodorAmanov1/QML_bag_of_tricks](https://github.com/FyodorAmanov1/QML_bag_of_tricks)
for owner review. Public documentation and Colab access remain pending. No Pages
deployment is triggered by a push; its workflow is manual.

After the owner chooses to make the repository public, run
`python scripts/prepare_release.py --repo-url https://github.com/FyodorAmanov1/QML_bag_of_tricks`.
It writes a public-links checklist without changing the source notebooks. The
generated links assume the `main` branch and project-style GitHub Pages; check
those assumptions against the actual repository. In Colab, set `REPO_URL` in
the setup cell to that URL. For a reviewed immutable install, append the
reviewed tag or commit to the Git installation URL.

The Pages workflow requires Pages to use GitHub Actions in repository settings
when public deployment is intended.
Its configured workflow is a deployment mechanism, not evidence that a live
site exists. Set the repository's About description, relevant topics, and
social preview using `docs/assets/social-preview.png` after publication.
An editable [LinkedIn draft and launch checklist](launch.md) are included for
use after review; no post has been sent.

After reviewing and tagging a commit, the manual **Prepare release artifacts**
workflow checks out that exact existing tag, executes the notebooks and tests,
and packages documentation and distributions. Its optional `create_draft`
input creates a draft GitHub release; the default only uploads build artifacts.
Review the draft and links before publishing. No workflow has been run remotely
as part of this local validation.
