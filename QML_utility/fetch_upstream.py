"""Fetch hash-checked inspected source files only; never execute upstream code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def main():
    sources = json.loads((ROOT / 'upstream_manifest.json').read_text())['sources']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=sorted(sources), required=True)
    args = p.parse_args()
    source = sources[args.case]
    destination = ROOT / '.cache' / args.case / source['commit']
    repository = source['repository'].removeprefix('https://github.com/')
    saved = []
    for item in source['files']:
        path = PurePosixPath(item['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe source path')
        url = f'https://raw.githubusercontent.com/{repository}/{source["commit"]}/{path}'
        with urlopen(url, timeout=30) as response:
            payload = response.read(2_000_001)
        if len(payload) > 2_000_000 or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Source size/hash mismatch: ' + str(path))
        target = destination / str(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        saved.append(str(path))
    print(json.dumps(dict(case=args.case, commit=source['commit'], files=saved,
                         verified=True, upstream_executed=False), indent=2))


if __name__ == '__main__':
    main()
