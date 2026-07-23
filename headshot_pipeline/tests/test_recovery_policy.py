from __future__ import annotations

import pytest

from server.evaluation.failure_taxonomy import (
    classify_failure,
    classify_selected_failure,
)
from server.gemini_worker import (
    append_recovery_constraint,
    order_references_for_recovery,
)
from server.shot_planner import (
    build_recovery_shot_spec,
    compose_recovery_shot_prompt,
)


@pytest.mark.parametrize(
    ("failure", "failure_class", "action"),
    [
        ("unsafe_content", "safety", "DROP_CANDIDATE"),
        ("no_face_detected", "face_detection", "REGENERATE_FROM_ORIGINAL"),
        (
            "identity_geometry_drift",
            "identity_geometry",
            "REGENERATE_WITH_POSE_REFERENCE",
        ),
        ("identity_too_low", "identity_similarity", "IDENTITY_REPAIR"),
        (
            "skin_over_smoothed",
            "synthetic_texture",
            "REGENERATE_FROM_ORIGINAL",
        ),
        ("wrong_composition", "composition", "REGENERATE_FROM_ORIGINAL"),
        ("too_blurry", "image_integrity", "REGENERATE_FROM_ORIGINAL"),
        ("bad_artifacts", "local_artifact", "LOCAL_EDIT"),
    ],
)
def test_failure_taxonomy_routes_distinct_failure_classes(
    failure: str,
    failure_class: str,
    action: str,
):
    diagnosis = classify_failure({"hard_failures": [failure]})

    assert diagnosis["failure_class"] == failure_class
    assert diagnosis["recovery_action"] == action


def test_pose_recovery_promotes_angled_reference_without_dropping_pack():
    paths, indexes = order_references_for_recovery(
        ["front.jpg", "smile.jpg", "angle.jpg", "other.jpg"],
        {"action": "REGENERATE_WITH_POSE_REFERENCE"},
        {"shot_id": "profile"},
    )

    assert paths == ["angle.jpg", "front.jpg", "smile.jpg", "other.jpg"]
    assert indexes == [2, 0, 1, 3]


def test_recovery_constraint_is_diagnosis_specific():
    prompt = append_recovery_constraint(
        "base prompt",
        {
            "failure_class": "identity_geometry",
            "targeted_constraint": "Preserve measured face geometry.",
        },
    )

    assert "identity_geometry" in prompt
    assert "Preserve measured face geometry" in prompt


def test_profile_fallback_keeps_slot_but_changes_photographic_solution():
    recovered = build_recovery_shot_spec(
        {
            "shot_id": "profile",
            "framing": "strict side profile",
            "pose": "90 degree turn",
            "environment": "roof terrace",
            "lighting": "window light",
            "lens": "85mm",
        },
        failure_class="identity_geometry",
        attempt=1,
    )
    prompt = compose_recovery_shot_prompt("old strict profile", recovered)

    assert recovered["shot_id"] == "profile"
    assert recovered["canonical_shot_id"] == "profile"
    assert recovered["shot_variant"] == "soft_turned_portrait"
    assert "20 to 35 degrees" in recovered["pose"]
    assert "supersedes earlier" in prompt


def test_selected_failure_prefers_persisted_final_judgement():
    diagnosis = classify_selected_failure({
        "selected_candidate": {
            "final_judgement": {
                "hard_failures": ["skin_over_smoothed"],
            },
            "gate_status": {
                "hard_gate_failures": ["severe_quality_failure"],
            },
        },
    })

    assert diagnosis["failure_class"] == "synthetic_texture"
    assert "skin_over_smoothed" in diagnosis["observed_failures"]
