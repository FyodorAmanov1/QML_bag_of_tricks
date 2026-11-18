"""Separate published evidence, author-source access and local execution scope."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OTHER_ACCESS = {
    'late_query_bell': ('local_reference', None, 'Native acquisition references and the restricted local audit; no pinned complete author experiment pipeline.'),
    'photonic_process': ('author_archive_link', 'https://doi.org/10.11583/dtu.29517107.v1', 'Published institutional data/code archive; no file-level pin inspected here.'),
    'discrete_log_kernel': ('theory', None, 'Conditional theoretical result; no practical implementation entry point asserted.'),
    'molecular_reservoir': ('source_not_recovered', None, 'Published paper and supporting information; exact regression code/artifacts not recovered here.'),
    'pauli_channel_learning': ('code_on_request', None, 'Inspected publisher manuscript describes analysis code on request; local native references cover ideal components.'),
}


def implementation_access(case):
    """A public source link is not a license grant or a local training result."""
    key = case.get('upstream_key')
    if key:
        source = json.loads((ROOT/'upstream_manifest.json').read_text(encoding='utf-8'))['sources'][key]
        return dict(status='public_author_source_inspected', public_author_source_verified=True,
                    repository=source['repository'], commit=source['commit'],
                    entrypoint=source['repository']+'/blob/'+source['commit']+'/'+source['files'][0]['path'],
                    license_recorded=source['license_recorded'], source_manifest='upstream_manifest.json',
                    scope='Inspected author source; check the evidence guide for code/paper differences and reuse terms. Not executed here.')
    status, url, scope = OTHER_ACCESS[case['id']]
    return dict(status=status, public_author_source_verified=False, entrypoint=url,
                license_recorded=None, scope=scope)
