"""Prepare a reviewable local release package; never push or publish."""
import argparse,hashlib,json,shutil,tomllib,zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
version=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo-url');a=p.parse_args()
target=None
if a.repo_url:
    parsed=urlparse(a.repo_url);parts=parsed.path.removesuffix('.git').strip('/').split('/')
    if parsed.scheme!='https' or parsed.netloc!='github.com' or len(parts)!=2 or parsed.query or parsed.fragment or not all(parts):raise ValueError('Expected https://github.com/OWNER/REPOSITORY')
    target='/'.join(parts)
out=ROOT/'release';out.mkdir(exist_ok=True);links={}
distributions=list((ROOT/'dist').glob(f'qml_bag_of_tricks-{version}*'))
if not any(p.suffix=='.whl' for p in distributions) or not any(p.name.endswith('.tar.gz') for p in distributions):
    raise ValueError('Build the current wheel and source distribution first: python -m build')
if not (ROOT/'site/index.html').is_file():raise ValueError('Build documentation first: python -m mkdocs build --strict')
if any(not p.name.startswith(f'qml_bag_of_tricks-{version}') for p in out.glob('qml_bag_of_tricks-*')):
    raise ValueError('Release directory contains another package version; use a clean output directory')
if target:
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        links[path.name]=f'https://colab.research.google.com/github/{target}/blob/main/notebooks/{path.name}'
    (out/'PUBLIC_LINKS.md').write_text('# Public destinations\n\n'+f'Repository: https://github.com/{target}\n\n'+f'Docs: https://{parts[0]}.github.io/{parts[1]}/\n\n'+'\n'.join(f'- [{k}]({v})' for k,v in links.items())+'\n\nLinks assume a public main branch and GitHub Pages deployment. In Colab, set REPO_URL in the setup cell to the repository URL above. This script does not publish or modify source notebooks.\n',encoding='utf-8')
if (ROOT/'site').exists():
    with zipfile.ZipFile(out/'documentation.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in (ROOT/'site').rglob('*'):
            if path.is_file():z.write(path,path.relative_to(ROOT/'site'))
for path in distributions:
    if path.is_file():shutil.copy2(path,out/path.name)
for name in ('RELEASE_NOTES.md','VALIDATION.md'):shutil.copy2(ROOT/name,out/name)
shutil.copy2(ROOT/'docs/assets/social-preview.png',out/'social-preview.png')
manifest=dict(version=version,status='local candidate; owner scientific review pending',repository=target,
              public_links_configured=bool(target),published=False,colab_links=links,
              artifacts={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.is_file() and p.name!='readiness.json'},
              next_action='owner review, select destination, push and trigger the draft-release workflow')
(out/'readiness.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
