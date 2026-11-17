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


def on_nav(nav, config, files):
    """Keep the practical entry point first without dropping auto-discovered guides."""
    utility = next((item for item in nav.items if item.is_section and item.title == 'Utility'), None)
    if utility is None:
        return nav
    utility.title = 'Start here: QML Utility'
    first = ['index.md', 'unified_setup.md', 'start_here.md', 'project_agent_brief.md',
             'case_selection.md', 'practical_workflow.md', 'evaluation.md', 'composing_methods.md']
    priority = {f'utility/{name}': index for index, name in enumerate(first)}
    utility.children.sort(key=lambda item: priority.get(getattr(getattr(item, 'file', None), 'src_uri', ''), len(first)))
    nav.items.remove(utility)
    nav.items.insert(1 if nav.items and nav.items[0].is_page and nav.items[0].is_homepage else 0, utility)

    def pages(items):
        for item in items:
            if item.is_page:
                yield item
            elif item.is_section:
                yield from pages(item.children)

    nav.pages[:] = list(pages(nav.items))
    for index, page in enumerate(nav.pages):
        page.previous_page = nav.pages[index-1] if index else None
        page.next_page = nav.pages[index+1] if index+1 < len(nav.pages) else None
    return nav


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
    for name, dest in (('LICENSE', 'LICENSE.txt'), ('NOTICE', 'NOTICE.txt'), ('CITATION.cff', 'CITATION.cff')):
        files.append(File.generated(config, f'assets/legal/{dest}', abs_src_path=str(root/name)))
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
