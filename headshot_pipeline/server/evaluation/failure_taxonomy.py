"""Structured failure diagnosis for the portrait recovery policy.

The evaluator emits low-level gate names.  This module turns those signals
into stable product-level failure classes so recovery decisions can be
measured across identities, styles, models, and releases.
"""

from __future__ import annotations

from typing import Any


FAILURE_PRIORITIES: tuple[tuple[str, set[str]], ...] = (
    ("safety", {"unsafe_content"}),
    (
        "face_detection",
        {
            "no_face",
            "no_face_detected",
            "no_usable_face_detected",
            "identity_no_generated_face",
            "multiple_faces",
        },
    ),
    ("identity_geometry", {"identity_geometry_drift"}),
    ("identity_similarity", {"identity_too_low", "identity_fail"}),
    (
        "synthetic_texture",
        {"synthetic_appearance", "skin_over_smoothed"},
    ),
    (
        "composition",
        {"wrong_composition", "anti_selfie_composition"},
    ),
    (
        "image_integrity",
        {
            "unreadable_image",
            "bad_resolution",
            "too_blurry",
            "face_distorted",
            "severe_artifacts",
        },
    ),
    (
        "local_artifact",
        {"bad_artifacts"},
    ),
    ("judge_uncertain", {"judge_failed"}),
)


RECOVERY_BY_CLASS = {
    "none": ("ACCEPT", "accept"),
    "safety": ("DROP_CANDIDATE", "drop_unsafe"),
    "face_detection": (
        "REGENERATE_FROM_ORIGINAL",
        "rebuild_face_and_composition",
    ),
    "identity_geometry": (
        "REGENERATE_WITH_POSE_REFERENCE",
        "pose_matched_regeneration",
    ),
    "identity_similarity": ("IDENTITY_REPAIR", "identity_writeback"),
    "synthetic_texture": (
        "REGENERATE_FROM_ORIGINAL",
        "photoreal_regeneration",
    ),
    "composition": (
        "REGENERATE_FROM_ORIGINAL",
        "composition_regeneration",
    ),
    "image_integrity": (
        "REGENERATE_FROM_ORIGINAL",
        "clean_source_regeneration",
    ),
    "local_artifact": ("LOCAL_EDIT", "localized_artifact_repair"),
    "judge_uncertain": (
        "REGENERATE_FROM_ORIGINAL",
        "fail_closed_regeneration",
    ),
    "unknown_quality": (
        "REGENERATE_FROM_ORIGINAL",
        "quality_regeneration",
    ),
}


TARGETED_CONSTRAINTS = {
    "identity_geometry": (
        "Use the reference closest to the requested head angle as the primary "
        "identity anchor; preserve face-width, eye spacing, jaw, and head turn."
    ),
    "synthetic_texture": (
        "Target an unretouched camera photograph with ordinary skin-frequency "
        "detail, natural asymmetry, fine hair, and local tonal variation."
    ),
    "composition": (
        "Rebuild the frame from the written ShotSpec; prioritize crop, pose, "
        "camera distance, and readable environment over styling."
    ),
    "face_detection": (
        "Keep one unobstructed, naturally proportioned face at a readable size."
    ),
    "image_integrity": (
        "Produce a clean, sharp exposure with an undistorted face and sufficient "
        "native detail for delivery."
    ),
}


def classify_failure(
    judgement: dict[str, Any] | None,
    *,
    gate_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Return one stable primary diagnosis plus all observed failure signals."""
    judgement = judgement if isinstance(judgement, dict) else {}
    observed = {
        str(item)
        for item in (judgement.get("hard_failures") or [])
        if item
    }
    observed.update(str(item) for item in (gate_failures or []) if item)

    primary = "none"
    matched: set[str] = set()
    for failure_class, signals in FAILURE_PRIORITIES:
        overlap = observed & signals
        if overlap:
            primary = failure_class
            matched = overlap
            break

    if primary == "none" and observed:
        primary = "unknown_quality"
        matched = set(observed)

    action, strategy = RECOVERY_BY_CLASS[primary]
    return {
        "failure_class": primary,
        "recovery_action": action,
        "recovery_strategy": strategy,
        "matched_failures": sorted(matched),
        "observed_failures": sorted(observed),
        "targeted_constraint": TARGETED_CONSTRAINTS.get(primary),
    }


def classify_selected_failure(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Diagnose the final selected candidate in persisted pipeline metadata."""
    metadata = metadata if isinstance(metadata, dict) else {}
    selected = metadata.get("selected_candidate")
    selected = selected if isinstance(selected, dict) else {}
    gate = selected.get("gate_status")
    gate = gate if isinstance(gate, dict) else {}
    judgement = selected.get("final_judgement")
    if not isinstance(judgement, dict):
        diagnosis = metadata.get("failure_diagnosis")
        if isinstance(diagnosis, dict) and diagnosis.get("failure_class"):
            return dict(diagnosis)
    return classify_failure(
        judgement if isinstance(judgement, dict) else {},
        gate_failures=gate.get("hard_gate_failures") or [],
    )
