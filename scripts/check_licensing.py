"""Verify legal and citation files in a built wheel and source archive.

Run after `python -m build` in the development/docs environment.
"""
import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import tomllib
import zipfile
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist-dir', type=Path, default=Path('dist'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']
    citation = yaml.safe_load((root/'CITATION.cff').read_text(encoding='utf-8'))
    assert citation['license'] == project['license'] == 'Apache-2.0'
    assert citation['version'] == project['version']
    assert citation['authors'][0] == {'family-names': 'Amanov', 'given-names': 'Fyodor'}
    normalize = lambda data: data.replace(b'\r\n', b'\n')
    legal = ['LICENSE', 'NOTICE', 'LICENSES/MIT-legacy.txt']
    stem = 'qml_bag_of_tricks-'+project['version']
    with zipfile.ZipFile(args.dist_dir/(stem+'-py3-none-any.whl')) as wheel:
        prefix = stem+'.dist-info/'
        metadata = Parser().parsestr(wheel.read(prefix+'METADATA').decode())
        assert metadata['License-Expression'] == 'Apache-2.0'
        assert set(metadata.get_all('License-File')) == set(legal)
        for name in legal:
            assert normalize(wheel.read(prefix+'licenses/'+name)) == normalize((root/name).read_bytes())
        assert normalize(wheel.read('QML_utility/LICENSE')) == normalize((root/'LICENSE').read_bytes())
    with tarfile.open(args.dist_dir/(stem+'.tar.gz')) as source:
        for name in legal+['CITATION.cff', 'QML_utility/LICENSE']:
            assert normalize(source.extractfile(stem+'/'+name).read()) == normalize((root/name).read_bytes())
    print('Verified Apache-2.0 metadata, author citation and legal files in wheel and source archive.')


if __name__ == '__main__':
    main()
