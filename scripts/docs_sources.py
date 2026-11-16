"""MkDocs hook: copy canonical SDK files into downloadable site assets."""
from pathlib import Path
import json
import posixpath
import re
from mkdocs.structure.files import File


def utility_sources(root):
    folder = root/'QML_utility'
    paths = list(folder.glob('*.md'))+list(folder.glob('*.json'))
    for name in ('evidence','recipes','protocols','results'):
        paths += [p for p in (folder/name).glob('*') if p.is_file() and p.suffix in ('.md','.json','.gz')]
    sources = {}
    for path in paths:
        relative = path.relative_to(folder)
        if path.suffix=='.md':
            dest = 'utility/'+str(relative.with_name('index.md' if path.name=='README.md' else path.name.lower())).replace('\\','/')
        else:
            dest = 'utility/'+relative.as_posix()
        sources[path.resolve()] = dest
    for path in list(folder.glob('*.py'))+list((folder/'models').glob('*.py'))+list((folder/'benchmarks').glob('*.py')):
        sources[path.resolve()] = 'utility/source/'+path.relative_to(folder).as_posix()
    return sources


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
            ('utility', 'Start with QML Utility'),
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
    if page.file.src_uri.startswith('utility/'):
        root = Path(config['config_file_path']).parent
        mapping = utility_sources(root)
        canonical = next(path for path,dest in mapping.items() if dest==page.file.src_uri)
        def link(match):
            target,separator,anchor = match.group(2).partition('#')
            if not target or '://' in target or target.startswith(('mailto:','/')):
                return match.group(0)
            path = (canonical.parent/target).resolve()
            dest = mapping.get(path)
            if dest is None and path.is_relative_to((root/'docs').resolve()):
                dest = path.relative_to((root/'docs').resolve()).as_posix()
            if dest is None:
                return match.group(0)
            relative = posixpath.relpath(dest,posixpath.dirname(page.file.src_uri))
            return match.group(1)+'('+relative+(separator+anchor if separator else '')+')'
        return re.sub(r'(\[[^\]]*\])\(([^)]+)\)',link,markdown)
    if page.file.src_uri.startswith("learning/"):
        return markdown.replace("../../src/qml_tricks/learning/recipes/", "../source/learning/")
    if page.file.src_uri.startswith("strategies/"):
        return markdown.replace("../../src/qml_tricks/strategies/", "../source/")
    return markdown


def on_files(files, config):
    root = Path(config["config_file_path"]).parent
    for path,dest in utility_sources(root).items():
        files.append(File.generated(config,dest,abs_src_path=str(path)))
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
