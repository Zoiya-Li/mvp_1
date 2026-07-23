from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/benchmark_pipeline_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_pipeline_matrix", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _fixture(root: Path, name: str, gender: str = "female") -> None:
    fixture = root / name
    fixture.mkdir()
    (fixture / "fixture.json").write_text(json.dumps({
        "identity_id": name,
        "gender": gender,
        "synthetic": True,
        "contains_real_person": False,
        "allowed_uses": ["internal_qa"],
    }))
    for index in range(4):
        Image.new("RGB", (16, 16), (index * 20, 30, 40)).save(
            fixture / f"{index}.jpg"
        )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(fixture.glob("*.jpg"))
    }
    (fixture / "audit.json").write_text(json.dumps({
        "schema_version": 1,
        "pass": True,
        "source_sha256": hashes,
    }))


def test_discovery_and_matrix_cross_every_identity_with_every_theme(tmp_path):
    _fixture(tmp_path, "person-a")
    _fixture(tmp_path, "person-b", "male")

    fixtures = benchmark.discover_fixtures(tmp_path)
    cases = benchmark.build_cases(fixtures, ["theme-a", "theme-b"])

    assert len(fixtures) == 2
    assert len(cases) == 4
    assert {case["identity_id"] for case in cases} == {"person-a", "person-b"}


def test_theme_capabilities_filter_incompatible_presentations(tmp_path):
    _fixture(tmp_path, "person-a")
    _fixture(tmp_path, "person-b", "male")
    fixtures = benchmark.discover_fixtures(tmp_path)
    catalog = {
        "female-a": {"slug": "female-a", "presentation": "female"},
        "female-b": {"slug": "female-b", "presentation": "female"},
        "male-a": {"slug": "male-a", "presentation": "male"},
        "male-b": {"slug": "male-b", "presentation": "male"},
    }

    cases = benchmark.validate_theme_coverage(
        fixtures,
        list(catalog),
        catalog,
        minimum_themes_per_identity=2,
    )

    assert len(cases) == 4
    assert all(
        case["gender"] == case["theme_presentation"] for case in cases
    )


def test_theme_preflight_rejects_unknown_slug_before_execution(tmp_path):
    _fixture(tmp_path, "person-a")
    fixtures = benchmark.discover_fixtures(tmp_path)

    try:
        benchmark.validate_theme_coverage(
            fixtures,
            ["typo"],
            {"real": {"slug": "real", "presentation": "female"}},
            minimum_themes_per_identity=1,
        )
    except ValueError as exc:
        assert "Unknown production themes: typo" in str(exc)
    else:
        raise AssertionError("Unknown theme was not rejected")


def test_single_identity_can_never_be_promotion_eligible():
    summary = benchmark.summarize_reports(
        [{
            "identity_id": "person-a",
            "status": "pass",
            "report": {},
        }],
        fixture_count=1,
        minimum_identities=3,
        pass_rate_threshold=0.8,
        executed=True,
    )

    assert summary["pass_rate"] == 1.0
    assert summary["promotion_eligible"] is False
    assert "requires_at_least_3_identities" in summary["promotion_blockers"]


def test_summary_counts_failure_taxonomy_across_cases():
    summary = benchmark.summarize_reports(
        [{
            "identity_id": "person-a",
            "status": "failed",
            "report": {
                "support_snapshot": {
                    "generation": {
                        "events": [{
                            "evaluation": {
                                "failure_diagnosis": {
                                    "failure_class": "synthetic_texture"
                                }
                            }
                        }]
                    }
                }
            },
        }],
        fixture_count=3,
        minimum_identities=3,
        pass_rate_threshold=0.8,
        executed=True,
    )

    assert summary["failure_classes"] == {"synthetic_texture": 1}
    assert summary["promotion_eligible"] is False


def test_summary_infers_failure_class_from_compact_gate_diagnostics():
    summary = benchmark.summarize_reports(
        [{
            "identity_id": "person-a",
            "status": "failed",
            "report": {
                "failure_diagnostics": {
                    "failed_evaluations": [{
                        "evaluation": {
                            "hard_failures": ["skin_over_smoothed"],
                            "gate_status": {
                                "hard_gate_failures": ["quality_below_threshold"]
                            },
                        }
                    }]
                }
            },
        }],
        fixture_count=3,
        minimum_identities=3,
        pass_rate_threshold=0.8,
        executed=True,
    )

    assert summary["failure_classes"] == {"synthetic_texture": 1}


def test_passed_case_attempt_failures_do_not_count_as_terminal_failure():
    summary = benchmark.summarize_reports(
        [{
            "identity_id": "person-a",
            "status": "pass",
            "theme_slug": "theme-a",
            "report": {"hard_failures": ["identity_fail"]},
        }],
        fixture_count=1,
        minimum_identities=1,
        pass_rate_threshold=1.0,
        executed=True,
    )

    assert summary["failure_classes"] == {}
    assert summary["observed_attempt_failure_classes"] == {
        "identity_similarity": 1
    }


def test_worker_interruption_is_classified_for_bounded_infrastructure_retry():
    report = {
        "failure_diagnostics": {
            "failed_generation_reasons": {"worker_interrupted": 1},
            "events": [{"failure_reason": "worker_interrupted"}],
        }
    }

    assert benchmark.infrastructure_failure_reasons(report) == [
        "worker_interrupted"
    ]


def test_quality_failure_is_never_reclassified_as_infrastructure():
    report = {
        "failure_diagnostics": {
            "failed_generation_reasons": {"delivery_gate_failed": 1},
            "events": [{
                "failure_reason": "delivery_gate_failed",
                "evaluation": {"hard_failures": ["skin_over_smoothed"]},
            }],
        }
    }

    assert benchmark.infrastructure_failure_reasons(report) == []
