"""MkDocs hook: copy canonical SDK files into downloadable site assets."""
from pathlib import Path
import json
from mkdocs.structure.files import File


def on_page_context(context, page, config, nav):
    root = Path(config['config_file_path']).parent
    if page.is_homepage:
        context['showcase'] = json.loads((root/'docs/assets/showcase.json').read_text(encoding='utf-8'))
    if page.is_homepage or page.file.src_uri == 'agents.md':
        context['agent_briefs'] = {
            mode: (root/f'docs/assets/agent-{mode}.txt').read_text(encoding='utf-8').strip()
            for mode in ('website', 'local')
        }
    if page.file.src_uri == 'explore.md':
        groups = {key: dict(id=key, title=title, pages=[]) for key, title in (
            ('essentials', 'Essentials & project guides'),
            ('strategies', 'Circuit strategies'),
            ('learning', 'Learning workflows'),
            ('use_cases', 'Published applications & model APIs'),
            ('reproductions', 'Paper reproductions'),
            ('applications', 'Application benchmarks'),
        )}
        for guide in nav.pages:
            if guide.is_homepage or guide.file.src_uri == 'explore.md':
                continue
            folder = guide.file.src_uri.split('/')[0]
            groups[folder if folder in groups else 'essentials']['pages'].append(guide)
        context['documentation_groups'] = list(groups.values())
        # These anchors are rendered by explore.html after Markdown conversion.
        # Register them so MkDocs can validate links to the complete directory.
        page.present_anchor_ids.update(groups)
        page.present_anchor_ids.update(f'directory-{key}' for key in groups)
    return context


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri.startswith("learning/"):
        return markdown.replace("../../src/qml_tricks/learning/recipes/", "../source/learning/")
    if page.file.src_uri.startswith("strategies/"):
        return markdown.replace("../../src/qml_tricks/strategies/", "../source/")
    return markdown


def on_files(files, config):
    root = Path(config["config_file_path"]).parent
    source = root / "src/qml_tricks/strategies"
    for path in source.glob("*/*_impl.py"):
        dest = f"source/{path.parent.name}/{path.name}"
        # Paths are rooted at docs; virtual sources use the actual canonical file.
        files.append(File.generated(config, dest, abs_src_path=str(path)))
    for path in (root / 'src/qml_tricks/learning/recipes').glob('*/*_impl.py'):
        files.append(File.generated(config,f'source/learning/{path.parent.name}/{path.name}',abs_src_path=str(path)))
    for path in (root/'notebooks').glob('*.ipynb'):
        files.append(File.generated(config,f'source/notebooks/{path.name}',abs_src_path=str(path)))
    for path in (root/'src/qml_tricks/learning/application_backends').glob('*.py'):
        if path.name != '__init__.py':
            files.append(File.generated(config,f'source/applications/{path.name}',abs_src_path=str(path)))
    for name in ('opportunities.json.gz', 'opportunities_summary.json'):
        path = root/'results'/name
        if path.exists():
            files.append(File.generated(config,f'source/results/{name}',abs_src_path=str(path)))
    for path in (root/'src/qml_tricks/learning/replication_backends').glob('*.py'):
        if path.name != '__init__.py':
            files.append(File.generated(config,f'source/replications/{path.name}',abs_src_path=str(path)))
    for path in (root/'src/qml_tricks/learning/case_backends').glob('*.py'):
        if path.name != '__init__.py':
            files.append(File.generated(config,f'source/case_models/{path.name}',abs_src_path=str(path)))
    for name, dest in (('case_models.py','models.py'),('case_benchmarks.py','benchmarks.py')):
        path = root/'src/qml_tricks/learning'/name
        files.append(File.generated(config,f'source/case_models/{dest}',abs_src_path=str(path)))
    path = root/'results/case_models_reference.json'
    if path.exists():
        files.append(File.generated(config,'source/results/case_models_reference.json',abs_src_path=str(path)))
    return files
