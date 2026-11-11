"""MkDocs hook: copy canonical SDK files into downloadable site assets."""
from pathlib import Path
from mkdocs.structure.files import File


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
    return files
