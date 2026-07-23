#!/usr/bin/env python3
"""Run and aggregate a multi-identity x multi-theme production benchmark.

This is the release gate for portrait-pipeline changes. A single successful
fixture can produce evidence, but it can never make a release promotion-
eligible. Fixture folders must contain a ``fixture.json`` consent/audit label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
INFRASTRUCTURE_FAILURE_REASONS = {
    "worker_interrupted",
    "worker_unavailable",
    "provider_transport_error",
    "service_unavailable",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_fixtures(root: Path) -> list[dict[str, Any]]:
    fixtures = []
    for metadata_path in sorted(root.glob("*/fixture.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fixture_dir = metadata_path.parent
        images = sorted(
            path for path in fixture_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not 4 <= len(images) <= 6:
            raise ValueError(
                f"{fixture_dir.name} must contain 4-6 reference images"
            )
        if metadata.get("gender") not in {"female", "male"}:
            raise ValueError(f"{metadata_path} must declare female or male gender")
        if metadata.get("synthetic") is not True:
            raise ValueError(f"{metadata_path} is not marked synthetic")
        if metadata.get("contains_real_person") is not False:
            raise ValueError(f"{metadata_path} does not exclude a real person")
        if "internal_qa" not in (metadata.get("allowed_uses") or []):
            raise ValueError(f"{metadata_path} does not allow internal QA")
        audit_path = fixture_dir / "audit.json"
        if not audit_path.exists():
            raise ValueError(f"{fixture_dir.name} has no audit.json")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("pass") is not True:
            raise ValueError(f"{fixture_dir.name} failed strict fixture audit")
        expected_hashes = audit.get("source_sha256") or {}
        actual_hashes = {path.name: _sha256(path) for path in images}
        if expected_hashes != actual_hashes:
            raise ValueError(f"{fixture_dir.name} changed after its audit")
        fixtures.append({
            "identity_id": str(metadata.get("identity_id") or fixture_dir.name),
            "fixture_dir": fixture_dir.resolve(),
            "gender": metadata["gender"],
            "fixture_kind": str(
                metadata.get("fixture_kind")
                or "repository-owned synthetic adult"
            ),
            "reference_count": len(images),
            "audit_schema_version": audit.get("schema_version"),
        })
    return fixtures


def build_cases(
    fixtures: list[dict[str, Any]],
    themes: list[str],
    theme_catalog: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    cases = []
    for fixture in fixtures:
        for theme in themes:
            descriptor = (theme_catalog or {}).get(theme, {})
            presentation = descriptor.get("presentation")
            if (
                presentation in {"female", "male"}
                and presentation != fixture["gender"]
            ):
                continue
            cases.append({
                **fixture,
                "theme_slug": theme,
                "theme_presentation": presentation,
            })
    return cases


def fetch_theme_catalog(base_url: str) -> dict[str, dict[str, Any]]:
    """Read the production capability catalog before spending on a matrix."""
    response = httpx.get(
        f"{base_url.rstrip('/')}/api/v2/themes",
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    themes = payload.get("themes") if isinstance(payload, dict) else None
    if not isinstance(themes, list):
        raise ValueError("Theme catalog response has no themes list")
    catalog = {
        str(theme.get("slug")): theme
        for theme in themes
        if isinstance(theme, dict) and theme.get("slug")
    }
    if not catalog:
        raise ValueError("Theme catalog is empty")
    return catalog


def validate_theme_coverage(
    fixtures: list[dict[str, Any]],
    themes: list[str],
    theme_catalog: dict[str, dict[str, Any]],
    *,
    minimum_themes_per_identity: int,
) -> list[dict[str, Any]]:
    missing = [theme for theme in themes if theme not in theme_catalog]
    if missing:
        raise ValueError("Unknown production themes: " + ", ".join(missing))
    cases = build_cases(fixtures, themes, theme_catalog)
    coverage = Counter(case["identity_id"] for case in cases)
    undercovered = [
        fixture["identity_id"]
        for fixture in fixtures
        if coverage[fixture["identity_id"]] < minimum_themes_per_identity
    ]
    if undercovered:
        raise ValueError(
            "Theme selection does not provide at least "
            f"{minimum_themes_per_identity} compatible themes for: "
            + ", ".join(undercovered)
        )
    return cases


def _walk_diagnoses(value: Any):
    if isinstance(value, dict):
        diagnosis = value.get("failure_diagnosis")
        if isinstance(diagnosis, dict) and diagnosis.get("failure_class"):
            yield diagnosis
        for child in value.values():
            yield from _walk_diagnoses(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_diagnoses(child)


FAILURE_SIGNAL_CLASS = {
    "unsafe_content": "safety",
    "no_face": "face_detection",
    "no_face_detected": "face_detection",
    "no_usable_face_detected": "face_detection",
    "multiple_faces": "face_detection",
    "identity_no_generated_face": "face_detection",
    "identity_geometry_drift": "identity_geometry",
    "identity_too_low": "identity_similarity",
    "identity_fail": "identity_similarity",
    "synthetic_appearance": "synthetic_texture",
    "skin_over_smoothed": "synthetic_texture",
    "wrong_composition": "composition",
    "anti_selfie_composition": "composition",
    "unreadable_image": "image_integrity",
    "bad_resolution": "image_integrity",
    "too_blurry": "image_integrity",
    "face_distorted": "image_integrity",
    "severe_artifacts": "image_integrity",
    "bad_artifacts": "local_artifact",
    "judge_failed": "judge_uncertain",
}


def _walk_failure_signals(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"hard_failures", "hard_gate_failures"} and isinstance(
                child, list
            ):
                for signal in child:
                    if signal:
                        yield str(signal)
            elif key == "failure_reason" and child:
                yield str(child)
            yield from _walk_failure_signals(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_failure_signals(child)


def failure_classes_for_report(report: dict[str, Any]) -> set[str]:
    explicit = {
        str(diagnosis["failure_class"])
        for diagnosis in _walk_diagnoses(report)
        if diagnosis.get("failure_class") not in {None, "none"}
    }
    inferred = {
        FAILURE_SIGNAL_CLASS[signal]
        for signal in _walk_failure_signals(report)
        if signal in FAILURE_SIGNAL_CLASS
    }
    return explicit | inferred


def infrastructure_failure_reasons(report: dict[str, Any]) -> list[str]:
    diagnostics = report.get("failure_diagnostics") or {}
    reasons = set((diagnostics.get("failed_generation_reasons") or {}).keys())
    for event in diagnostics.get("events") or []:
        reason = event.get("failure_reason") if isinstance(event, dict) else None
        if reason:
            reasons.add(str(reason))
    return sorted(reasons & INFRASTRUCTURE_FAILURE_REASONS)


def summarize_reports(
    results: list[dict[str, Any]],
    *,
    fixture_count: int,
    minimum_identities: int,
    pass_rate_threshold: float,
    executed: bool,
    minimum_themes_per_identity: int = 1,
) -> dict[str, Any]:
    status_counts = Counter(result.get("status", "missing") for result in results)
    failure_classes: Counter[str] = Counter()
    identities_with_pass: set[str] = set()
    observed_attempt_failure_classes: Counter[str] = Counter()
    for result in results:
        if result.get("status") == "pass":
            identities_with_pass.add(str(result.get("identity_id")))
        classes = failure_classes_for_report(result.get("report") or {})
        for failure_class in classes:
            observed_attempt_failure_classes[failure_class] += 1
            if result.get("status") == "failed":
                failure_classes[failure_class] += 1
    case_count = len(results)
    passed = status_counts.get("pass", 0)
    pass_rate = passed / case_count if case_count else 0.0
    promotion_blockers = []
    if not executed:
        promotion_blockers.append("matrix_not_executed")
    if fixture_count < minimum_identities:
        promotion_blockers.append(
            f"requires_at_least_{minimum_identities}_identities"
        )
    if pass_rate < pass_rate_threshold:
        promotion_blockers.append("case_pass_rate_below_threshold")
    if len(identities_with_pass) < fixture_count:
        promotion_blockers.append("at_least_one_identity_has_no_passing_case")
    theme_coverage: dict[str, set[str]] = {}
    for result in results:
        identity_id = str(result.get("identity_id"))
        theme_slug = result.get("theme_slug")
        if theme_slug:
            theme_coverage.setdefault(identity_id, set()).add(str(theme_slug))
    if any(
        len(theme_coverage.get(str(result.get("identity_id")), set()))
        < minimum_themes_per_identity
        for result in results
    ):
        promotion_blockers.append("identity_theme_coverage_below_threshold")
    return {
        "fixture_count": fixture_count,
        "case_count": case_count,
        "status_counts": dict(sorted(status_counts.items())),
        "pass_rate": round(pass_rate, 4),
        "pass_rate_threshold": pass_rate_threshold,
        "minimum_themes_per_identity": minimum_themes_per_identity,
        "failure_classes": dict(sorted(failure_classes.items())),
        "observed_attempt_failure_classes": dict(
            sorted(observed_attempt_failure_classes.items())
        ),
        "promotion_eligible": not promotion_blockers,
        "promotion_blockers": promotion_blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--theme", action="append", dest="themes", required=True)
    parser.add_argument("--base-url", default="https://flashshot.top")
    parser.add_argument("--ssh-host", default="root@38.76.165.9")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-synthetic-fixtures", action="store_true")
    parser.add_argument("--minimum-identities", type=int, default=3)
    parser.add_argument("--pass-rate-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-themes-per-identity", type=int, default=2)
    parser.add_argument(
        "--infrastructure-retries",
        type=int,
        default=1,
        help="Retry service/transport interruptions without masking QA failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixtures = discover_fixtures(args.fixture_root.resolve())
    if not fixtures:
        raise SystemExit("No audited fixture.json files found")
    if args.execute and not args.confirm_synthetic_fixtures:
        raise SystemExit(
            "Refusing to execute without --confirm-synthetic-fixtures"
        )

    themes = list(dict.fromkeys(args.themes))
    theme_catalog = fetch_theme_catalog(args.base_url)
    cases = validate_theme_coverage(
        fixtures,
        themes,
        theme_catalog,
        minimum_themes_per_identity=args.minimum_themes_per_identity,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    e2e_script = (
        Path(__file__).resolve().parents[1]
        / "deploy/overseas-vps/validate_portrait_v2_e2e.py"
    )
    results = []
    started_at = time.time()

    for case in cases:
        case_id = f"{case['identity_id']}--{case['theme_slug']}"
        artifact_dir = output_dir / case_id
        result = {
            "case_id": case_id,
            "identity_id": case["identity_id"],
            "theme_slug": case["theme_slug"],
            "gender": case["gender"],
            "reference_count": case["reference_count"],
            "status": "planned",
        }
        results.append(result)
        if not args.execute:
            continue

        execution_attempts = []
        completed = None
        report = {}
        for attempt in range(args.infrastructure_retries + 1):
            attempt_artifact_dir = (
                artifact_dir
                if attempt == 0
                else output_dir / f"{case_id}--infra-retry-{attempt}"
            )
            command = [
                sys.executable,
                str(e2e_script),
                str(case["fixture_dir"]),
                "--base-url",
                args.base_url,
                "--ssh-host",
                args.ssh_host,
                "--theme-slug",
                case["theme_slug"],
                "--gender",
                case["gender"],
                "--fixture-kind",
                case["fixture_kind"],
                "--artifact-dir",
                str(attempt_artifact_dir),
            ]
            if args.preview_only:
                command.append("--preview-only")
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            report_path = attempt_artifact_dir / "report.json"
            report = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.exists() else {}
            )
            infra_reasons = infrastructure_failure_reasons(report)
            execution_attempts.append({
                "attempt": attempt + 1,
                "returncode": completed.returncode,
                "artifact_dir": str(attempt_artifact_dir),
                "infrastructure_failure_reasons": infra_reasons,
            })
            if completed.returncode == 0 or not infra_reasons:
                break
        assert completed is not None
        result.update({
            "status": "pass" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "report": report,
            "execution_attempts": execution_attempts,
            "stdout_tail": completed.stdout[-2_000:],
            "stderr_tail": completed.stderr[-2_000:],
        })
        (output_dir / "matrix-report.json").write_text(
            json.dumps(
                {"cases": results},
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    summary = summarize_reports(
        results,
        fixture_count=len(fixtures),
        minimum_identities=args.minimum_identities,
        pass_rate_threshold=args.pass_rate_threshold,
        executed=args.execute,
        minimum_themes_per_identity=args.minimum_themes_per_identity,
    )
    report = {
        "mode": "preview_only" if args.preview_only else "full_set",
        "executed": args.execute,
        "themes": themes,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "summary": summary,
        "cases": results,
    }
    (output_dir / "matrix-report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if (not args.execute or summary["promotion_eligible"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
