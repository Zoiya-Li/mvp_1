from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/audit_benchmark_fixture.py"
)
SPEC = importlib.util.spec_from_file_location("audit_benchmark_fixture", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_metadata_audit_requires_explicit_synthetic_provenance():
    issues = audit.validate_fixture_metadata({
        "identity_id": "fixture-a",
        "gender": "female",
        "synthetic": True,
    })

    assert "real_person_status_not_explicitly_false" in issues
    assert "internal_qa_use_not_declared" in issues
    assert "missing_provenance:generator" in issues


def test_complete_internal_qa_provenance_passes_metadata_audit():
    issues = audit.validate_fixture_metadata({
        "identity_id": "fixture-a",
        "gender": "female",
        "synthetic": True,
        "contains_real_person": False,
        "source_kind": "generated",
        "generator": "test-generator",
        "created_at": "2026-07-21",
        "allowed_uses": ["internal_qa"],
    })

    assert issues == []
