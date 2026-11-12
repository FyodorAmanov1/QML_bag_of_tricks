# Review and release

Version 0.2.0 is a local release candidate. Scientific review belongs to the
repository owner and is pending; no external QML-engineer review is required.
The [review packet](owner_review.md) links the evidence and checks.

The documentation is [live on GitHub Pages](https://fyodoramanov1.github.io/QML_bag_of_tricks/).
The source repository remains private. The owner explicitly approved public
documentation publication, including the built examples and notebooks.
Cloning the full source checkout still requires repository access.

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
for owner review. No Pages deployment is triggered by a push; its workflow is
manual and builds a downloadable preview by default.

## Preview the website privately

[Desktop design preview](assets/website-preview.png) ·
[Mobile design preview](assets/website-mobile-preview.png).

The designed homepage includes eight application cards, evidence filters, a
project selector, SDK commands and links to the complete guide library. Its
numbers are generated from the canonical reports and application catalog:

```bash
python scripts/build_showcase.py
python scripts/build_showcase.py --check
python -m mkdocs serve
```

On GitHub, run **Website preview and publication** with **publish = false**.
Download the `qml-website-preview` artifact from that run. In this private
repository the artifact remains available to authorized repository users;
this creates no public Pages deployment. Serve the downloaded directory with
`python -m http.server` to preview all relative links and assets.

## Publish the documentation site

GitHub Pages publication and repository visibility are separate decisions.
The built documentation includes downloadable Python examples, notebooks and
result artifacts. Public deployment exposes those built files even if the
Git source repository stays private. Confirm that audience before selecting
**publish = true**.

GitHub Pages can publish from a private repository on supported paid plans;
private visitor access requires the relevant GitHub Enterprise organization
setup. See [GitHub Pages availability](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
and [site access control](https://docs.github.com/en/enterprise-cloud@latest/pages/getting-started-with-github-pages/changing-the-visibility-of-your-github-pages-site).
An unsupported account needs a hosting or plan decision; do not change repository
visibility merely to make deployment succeed.

For an approved public documentation deployment, configure Pages to use
**GitHub Actions** in repository settings, then run the workflow with
**publish = true**. Verify the deployment's returned URL before adding a live
website link to the README or repository About section. The current site URL
is `https://fyodoramanov1.github.io/QML_bag_of_tricks/`. Ordinary repository
pushes leave the live site unchanged; run the workflow with **publish = true**
to publish a new documentation build.

## Public source and release links

After the owner chooses to make the repository public, run
`python scripts/prepare_release.py --repo-url https://github.com/FyodorAmanov1/QML_bag_of_tricks`.
It writes a public-links checklist without changing the source notebooks. The
generated links assume the `main` branch and project-style GitHub Pages; check
those assumptions against the actual repository. In Colab, set `REPO_URL` in
the setup cell to that URL. For a reviewed immutable install, append the
reviewed tag or commit to the Git installation URL.

Set the repository's About description, relevant topics, and
social preview using `docs/assets/social-preview.png` after publication.
An editable [LinkedIn draft and launch checklist](launch.md) are included for
use after review; no post has been sent.

After reviewing and tagging a commit, the manual **Prepare release artifacts**
workflow checks out that exact existing tag, executes the notebooks and tests,
and packages documentation and distributions. Its optional `create_draft`
input creates a draft GitHub release; the default only uploads build artifacts.
Review the draft and links before publishing. Website preview runs and public
deployment are distinct from the release workflow.
