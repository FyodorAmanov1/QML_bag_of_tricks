"""Keep literature claims scoped when generating the application entry point."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "application_catalog", ROOT / "scripts/build_application_catalog.py"
)
catalog_builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog_builder)


@pytest.fixture
def catalog():
    return json.loads((ROOT / "docs/use_cases/catalog.json").read_text(encoding="utf-8"))


def test_catalog_index_preserves_strongest_comparator_and_status(catalog):
    catalog_builder.validate_catalog(catalog, ROOT / "docs/use_cases")
    actual = (ROOT / "docs/use_cases/index.md").read_text(encoding="utf-8")
    assert actual == catalog_builder.render(catalog)
    assert "DenseNet: 94.9%" in actual  # Not only the weaker matched head.
    assert "Swift-Hyperband SVR: 3,014 epochs" in actual  # Not only plain Hyperband.
    assert "not locally reproduced" in actual
    assert "other nine" in actual


@pytest.mark.parametrize("field,value,error", [
    ("status", "locally_reproduced", "local reproduction"),
    ("classical_comparators", [], "classical comparator"),
    ("headline_caveat", "", "headline_caveat"),
    ("guide", "../index.md", "inside the catalog"),
])
def test_reject_evidence_promotion_or_missing_scope(catalog, field, value, error):
    catalog["applications"][0][field] = value
    with pytest.raises(ValueError, match=error):
        catalog_builder.validate_catalog(catalog, ROOT / "docs/use_cases")


@pytest.mark.parametrize("value", [95.8, float("nan"), float("inf"), True])
def test_reject_invalid_score_units(catalog, value):
    catalog["applications"][0]["quantum"]["value"] = value
    with pytest.raises(ValueError):
        catalog_builder.validate_catalog(catalog, ROOT / "docs/use_cases")


def test_reject_non_peer_reviewed_record(catalog):
    catalog["applications"][0]["publication"]["peer_reviewed"] = False
    with pytest.raises(ValueError, match="peer-reviewed"):
        catalog_builder.validate_catalog(catalog, ROOT / "docs/use_cases")


def test_renderer_does_not_require_quantum_to_win(catalog):
    changed = deepcopy(catalog)
    changed["applications"][0]["quantum"]["value"] = 0.5
    catalog_builder.validate_catalog(changed, ROOT / "docs/use_cases")
    assert "MLTQNN: **50.0%**" in catalog_builder.render(changed)
