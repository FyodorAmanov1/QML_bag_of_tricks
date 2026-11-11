"""Fetch pinned author artifacts; preserve their licenses and exact byte hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'alloys': ('peizong/quantum-machine-learning-metallurgy', '406f405980456c69213eae16d5df257f3fce0323', [
        ('simulator_data_process/alloy-property-avg.csv', 'reproductions/alloys/data.csv'),
        ('LICENSE', 'reproductions/alloys/LICENSE.upstream'),
        ('simulator_data_process/res/zz_1.2_8.npy', 'reproductions/alloys/published_8.npy'),
        ('simulator_data_process/res/zz_1.2_10.npy', 'reproductions/alloys/published_10.npy'),
        ('simulator_data_process/scripts/run_1.2_split.py', '.cache/paper_sources/alloy_script.py')]),
    'nisqrc': ('skhanCC/NISQRC-Codes', '153cd59aef951710f52acef764edae1c2564a26d', [
        ('3_Channel_Equalization_Error_vs_SNR.ipynb', '.cache/paper_sources/nisqrc.ipynb')])
}


def main():
    artifacts = []
    for study, (repo, commit, files) in SOURCES.items():
        for original, destination in files:
            url = f'https://raw.githubusercontent.com/{repo}/{commit}/{original}'
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            path = ROOT/destination; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            artifacts.append(dict(study=study, repository='https://github.com/'+repo,
                commit=commit, path=original, url=url, local_path=destination,
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    (ROOT/'reproductions/sources.json').write_text(json.dumps(artifacts, indent=2)+'\n', encoding='utf-8')
    print(f'Fetched and hashed {len(artifacts)} pinned artifacts; no upstream code executed.')


if __name__ == '__main__':
    main()
