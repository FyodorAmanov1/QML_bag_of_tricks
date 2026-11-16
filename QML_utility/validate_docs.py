"""Validate utility JSON syntax and local Markdown file links without SDK imports."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    excluded = {'.cache','local','build','dist','.pytest_cache','.venv','venv','__pycache__'}
    documents = [path for path in ROOT.rglob('*.md') if not set(path.relative_to(ROOT).parts)&excluded]
    broken = []
    for path in documents:
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            target = match.group(1).split('#',1)[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            if not (path.parent/target).exists():
                broken.append((str(path.relative_to(ROOT)),target))
    configs = [path for path in ROOT.rglob('*.json') if not set(path.relative_to(ROOT).parts)&excluded]
    for path in configs:
        json.loads(path.read_text(encoding='utf-8'))
    if broken:
        raise ValueError('Broken local links: '+repr(broken))
    print(json.dumps(dict(markdown_files=len(documents),json_files=len(configs),broken_links=0),indent=2))


if __name__ == '__main__':
    main()
