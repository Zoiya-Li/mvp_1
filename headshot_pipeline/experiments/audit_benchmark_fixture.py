#!/usr/bin/env python3
"""Strictly audit one synthetic identity fixture before benchmark admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from server.input_quality import (
    assess_reference_diversity,
    assess_reference_identity_consistency,
    assess_reference_photo,
    summarize_reference_set,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_PROVENANCE_FIELDS = {
    "source_kind",
    "generator",
    "created_at",
    "contains_real_person",
    "allowed_uses",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fixture_metadata(metadata: dict[str, Any]) -> list[str]:
    issues = []
    if metadata.get("synthetic") is not True:
        issues.append("fixture_not_marked_synthetic")
    if metadata.get("contains_real_person") is not False:
        issues.append("real_person_status_not_explicitly_false")
    if metadata.get("gender") not in {"female", "male"}:
        issues.append("invalid_gender")
    if not str(metadata.get("identity_id") or "").strip():
        issues.append("missing_identity_id")
    for field in sorted(REQUIRED_PROVENANCE_FIELDS):
        if field not in metadata:
            issues.append(f"missing_provenance:{field}")
    allowed_uses = metadata.get("allowed_uses")
    if not isinstance(allowed_uses, list) or "internal_qa" not in allowed_uses:
        issues.append("internal_qa_use_not_declared")
    return issues


def audit_fixture(fixture_dir: Path) -> dict[str, Any]:
    fixture_dir = fixture_dir.resolve()
    metadata_path = fixture_dir / "fixture.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {}
    )
    images = sorted(
        path for path in fixture_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    metadata_issues = validate_fixture_metadata(metadata)
    if not 4 <= len(images) <= 6:
        metadata_issues.append("reference_count_must_be_4_to_6")

    photo_quality = {
        path.name: assess_reference_photo(path)
        for path in images
    }
    identity = assess_reference_identity_consistency(images)
    pose = identity.get("pose_diversity") or {}
    diversity = assess_reference_diversity(images, min_unique=4)
    set_gate = summarize_reference_set(
        photo_quality,
        min_photos=4,
        identity_consistency=identity,
        diversity=diversity,
        pose_diversity=pose,
    )

    strict_issues = list(metadata_issues)
    if not all(record.get("pass") for record in photo_quality.values()):
        strict_issues.append("individual_photo_quality_failed")
    if identity.get("status") != "pass":
        strict_issues.append("identity_consistency_not_proven")
    if pose.get("status") != "pass":
        strict_issues.append("pose_diversity_not_proven")
    diversity_exception = set_gate.get("diversity_exception") or {}
    if (
        diversity.get("status") != "pass"
        and not diversity_exception.get("applied")
    ):
        strict_issues.append("reference_diversity_not_proven")
    if not set_gate.get("pass"):
        strict_issues.append("production_reference_gate_failed")
    strict_issues = list(dict.fromkeys(strict_issues))

    return {
        "schema_version": 1,
        "identity_id": metadata.get("identity_id"),
        "status": "pass" if not strict_issues else "fail",
        "pass": not strict_issues,
        "strict_issues": strict_issues,
        "reference_count": len(images),
        "source_sha256": {
            path.name: file_sha256(path) for path in images
        },
        "photo_quality": photo_quality,
        "identity_consistency": identity,
        "pose_diversity": pose,
        "diversity": diversity,
        "production_reference_gate": set_gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_fixture(args.fixture_dir)
    if args.write_report:
        (args.fixture_dir / "audit.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
