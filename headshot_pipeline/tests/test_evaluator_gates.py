"""Delivery-gate regression tests for the evaluation service.

Covers the fail-closed judge contract and the previously untested local
appearance detectors:

1. ``_parse_quality_judge_response`` — empty VLM responses and unparseable
   verdicts (when the legacy score-regex fallback also fails) must mark
   ``judge_failed`` instead of silently producing a scoreless pass.
2. ``_candidate_gate_status`` — a missing VLM realism score must fail the
   quality gate even when the local compatibility fallback can synthesize a
   value for diagnostics/ranking.
3. ``_local_identity_similarity_check`` — the ``skin_over_smoothed`` and
   ``identity_geometry_drift`` detectors, exercised with synthetic numpy
   images and fabricated face keypoints (no real photos, no InsightFace
   model download).

Run:  python -m pytest headshot_pipeline/tests/test_evaluator_gates.py -q
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

# Make the package importable whether run from the repo root or the pipeline dir.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.evaluation import EvaluationService  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# judge_failed: empty / unparseable VLM verdicts fail closed
# ──────────────────────────────────────────────────────────────────────

def test_empty_judge_response_marks_judge_failed():
    out = EvaluationService._parse_quality_judge_response("")
    assert "judge_failed" in out["hard_failures"]
    assert out["scores"]["realism"] is None


def test_none_judge_response_marks_judge_failed():
    out = EvaluationService._parse_quality_judge_response(None)
    assert "judge_failed" in out["hard_failures"]


def test_unparseable_judge_response_marks_judge_failed():
    # No JSON object and no legacy "评分：N/10" score anywhere.
    out = EvaluationService._parse_quality_judge_response(
        "I cannot evaluate this image."
    )
    assert "judge_failed" in out["hard_failures"]
    assert out["scores"]["identity"] is None


def test_judge_failed_verdict_never_passes_delivery_gate():
    for text in ("", "not a json verdict at all"):
        judgement = EvaluationService._parse_quality_judge_response(text)
        judgement["scores"]["identity"] = 9  # local identity still scored
        gate = EvaluationService._candidate_gate_status(judgement)
        assert gate["hard_gates_pass"] is False
        assert gate["severe_quality_fail"] is True
        assert "severe_quality_failure" in gate["hard_gate_failures"]


def test_legacy_score_fallback_stays_compatible_but_blocks_quality_gate():
    # The old-score regex still rescues a plain-language verdict, but without
    # a VLM realism score the quality gate must not pass.
    judgement = EvaluationService._parse_quality_judge_response(
        "评分：9/10\n非常相似，无需调整。"
    )
    assert "judge_failed" not in judgement["hard_failures"]
    assert judgement["scores"]["identity"] == 9
    assert judgement["scores"]["realism"] is None

    gate = EvaluationService._candidate_gate_status(judgement)
    assert gate["quality_pass"] is False
    assert gate["realism_score_missing"] is True
    assert gate["hard_gates_pass"] is False


# ──────────────────────────────────────────────────────────────────────
# realism gate: missing VLM realism must not pass quality
# ──────────────────────────────────────────────────────────────────────

def test_missing_realism_marks_local_fallback_and_fails_quality_gate():
    judgement = EvaluationService._parse_quality_judge_response(
        '{"scores":{"identity":null,"face_quality":9,"style_match":9,'
        '"artifact":9,"commercial_readiness":9},'
        '"hard_failures":[],"recommended_action":"accept","notes":"ok"}'
    )
    # Local compatibility value retained for diagnostics/ranking …
    assert judgement["scores"]["realism"] == 9
    # … but clearly marked as not coming from the VLM.
    assert judgement["realism_source"] == "local_fallback"

    judgement["scores"]["identity"] = 9
    gate = EvaluationService._candidate_gate_status(judgement)
    assert gate["quality_pass"] is False
    assert gate["realism_score_missing"] is True
    assert gate["hard_gates_pass"] is False
    assert "quality_below_threshold" in gate["hard_gate_failures"]


def test_vlm_realism_passes_gate_unchanged():
    judgement = EvaluationService._parse_quality_judge_response(
        '{"scores":{"identity":null,"face_quality":9,"style_match":9,'
        '"realism":9,"artifact":9,"commercial_readiness":9},'
        '"hard_failures":[],"recommended_action":"accept","notes":"ok"}'
    )
    assert judgement["realism_source"] == "vlm"

    judgement["scores"]["identity"] = 9
    gate = EvaluationService._candidate_gate_status(judgement)
    assert gate["quality_pass"] is True
    assert gate["realism_score_missing"] is False
    assert gate["hard_gates_pass"] is True


def test_low_vlm_realism_still_fails_quality_gate():
    judgement = EvaluationService._parse_quality_judge_response(
        '{"scores":{"identity":null,"face_quality":9,"style_match":9,'
        '"realism":6,"artifact":9,"commercial_readiness":9},'
        '"hard_failures":[],"recommended_action":"retry","notes":"plastic"}'
    )
    assert judgement["realism_source"] == "vlm"
    judgement["scores"]["identity"] = 9
    gate = EvaluationService._candidate_gate_status(judgement)
    assert gate["quality_pass"] is False
    assert gate["realism_score_missing"] is False


# ──────────────────────────────────────────────────────────────────────
# Local appearance detectors: skin_over_smoothed / identity_geometry_drift
# ──────────────────────────────────────────────────────────────────────

def _fake_face(bbox, kps, embedding):
    face = SimpleNamespace(
        bbox=bbox,
        normed_embedding=embedding,
    )
    if kps is not None:
        face.kps = kps
    return face


class _FakeIdentityApp:
    """Serve one fabricated face per get() call (references first, generated last)."""

    def __init__(self, faces):
        self._faces = list(faces)

    def get(self, _img):
        index = min(getattr(self, "_calls", 0), len(self._faces) - 1)
        self._calls = index + 1
        return [self._faces[index]]


def _identity_check(eval_svc, generated_path, reference_paths):
    return eval_svc._local_identity_similarity_check(
        str(generated_path),
        [str(path) for path in reference_paths],
    )


def _identity_check_with_shot(eval_svc, generated_path, reference_paths, shot_spec):
    return eval_svc._local_identity_similarity_check(
        str(generated_path),
        [str(path) for path in reference_paths],
        shot_spec=shot_spec,
    )


# Canonical frontal proportions inside a 384x384 bbox (see evaluator's
# appearance metrics). Distances are chosen so the aligned case passes every
# geometry threshold with a wide margin.
_REF_KPS = [
    (200.0, 220.0),  # left eye
    (312.0, 220.0),  # right eye
    (256.0, 300.0),  # nose
    (216.0, 360.0),  # left mouth corner
    (296.0, 360.0),  # right mouth corner
]
# Same pose, but eyes narrowed (inter-eye 0.292→0.208) and mouth narrowed
# (0.208→0.156): two ratios drift past their thresholds → drift fires.
_DRIFTED_KPS = [
    (216.0, 220.0),
    (296.0, 220.0),
    (256.0, 300.0),
    (226.0, 360.0),
    (286.0, 360.0),
]
_BBOX = [64.0, 64.0, 448.0, 448.0]


def test_identity_geometry_drift_fires_on_shifted_proportions(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    flat = np.full((512, 512, 3), 160, dtype=np.uint8)
    ref_path = tmp_path / "ref.png"
    gen_path = tmp_path / "gen.png"
    cv2.imwrite(str(ref_path), flat)
    cv2.imwrite(str(gen_path), flat)

    embedding = np.array([1.0, 0.0, 0.0])
    app = _FakeIdentityApp([
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
        _fake_face(_BBOX, np.array(_DRIFTED_KPS, dtype=np.float32), embedding),
    ])
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: app  # type: ignore[assignment]

    result = _identity_check(eval_svc, gen_path, [ref_path])

    assert "identity_geometry_drift" in result["hard_failures"]
    failed = result["measurements"]["appearance_geometry_failed"]
    assert "inter_eye_to_face_width" in failed
    assert "mouth_to_face_width" in failed
    # Untouched ratios must still pass.
    assert result["measurements"]["appearance_geometry"][
        "face_width_to_height"
    ]["pass"] is True


def test_identity_geometry_drift_stays_silent_when_proportions_match(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    flat = np.full((512, 512, 3), 160, dtype=np.uint8)
    ref_path = tmp_path / "ref.png"
    gen_path = tmp_path / "gen.png"
    cv2.imwrite(str(ref_path), flat)
    cv2.imwrite(str(gen_path), flat)

    embedding = np.array([1.0, 0.0, 0.0])
    app = _FakeIdentityApp([
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
    ])
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: app  # type: ignore[assignment]

    result = _identity_check(eval_svc, gen_path, [ref_path])

    assert "identity_geometry_drift" not in result["hard_failures"]
    assert result["measurements"]["appearance_geometry_failed"] == []
    assert result["hard_failures"] == []


def test_identity_geometry_selects_best_same_pose_reference(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    flat = np.full((512, 512, 3), 160, dtype=np.uint8)
    ref_a = tmp_path / "ref-a.png"
    ref_b = tmp_path / "ref-b.png"
    gen_path = tmp_path / "gen.png"
    for path in (ref_a, ref_b, gen_path):
        cv2.imwrite(str(path), flat)

    embedding = np.array([1.0, 0.0, 0.0])
    app = _FakeIdentityApp([
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
        _fake_face(_BBOX, np.array(_DRIFTED_KPS, dtype=np.float32), embedding),
        _fake_face(_BBOX, np.array(_DRIFTED_KPS, dtype=np.float32), embedding),
    ])
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: app  # type: ignore[assignment]

    result = _identity_check(eval_svc, gen_path, [ref_a, ref_b])

    assert result["measurements"]["appearance_reference_selection"] == (
        "pose_then_geometry"
    )
    assert result["measurements"]["appearance_reference_candidate_count"] == 2
    assert result["measurements"]["appearance_geometry_failed"] == []
    assert "identity_geometry_drift" not in result["hard_failures"]


def test_profile_geometry_uses_pose_matched_reference_thresholds(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    flat = np.full((512, 512, 3), 160, dtype=np.uint8)
    ref_path = tmp_path / "profile-ref.png"
    gen_path = tmp_path / "profile-gen.png"
    cv2.imwrite(str(ref_path), flat)
    cv2.imwrite(str(gen_path), flat)

    reference_kps = np.array([
        (180.0, 220.0), (338.0, 220.0), (318.0, 288.0),
        (230.0, 360.0), (370.0, 360.0),
    ], dtype=np.float32)
    generated_kps = np.array([
        (190.0, 220.0), (328.0, 220.0), (315.0, 288.0),
        (248.0, 360.0), (352.0, 360.0),
    ], dtype=np.float32)
    embedding = np.array([1.0, 0.0, 0.0])
    app = _FakeIdentityApp([
        _fake_face(_BBOX, reference_kps, embedding),
        _fake_face(_BBOX, generated_kps, embedding),
    ])
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: app  # type: ignore[assignment]

    result = _identity_check_with_shot(
        eval_svc,
        gen_path,
        [ref_path],
        {"shot_id": "profile", "pose": "three-quarter side profile"},
    )

    assert result["measurements"]["appearance_yaw_delta"] <= 0.10
    assert result["measurements"][
        "appearance_geometry_threshold_profile"
    ] == "pose_matched_profile"
    assert "identity_geometry_drift" not in result["hard_failures"]


def test_skin_over_smoothed_fires_when_generated_skin_loses_texture(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(7)
    textured = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    smooth = np.full((512, 512, 3), 160, dtype=np.uint8)
    ref_path = tmp_path / "ref.png"
    gen_path = tmp_path / "gen.png"
    cv2.imwrite(str(ref_path), textured)
    cv2.imwrite(str(gen_path), smooth)

    embedding = np.array([1.0, 0.0, 0.0])
    app = _FakeIdentityApp([
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
    ])
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: app  # type: ignore[assignment]

    result = _identity_check(eval_svc, gen_path, [ref_path])

    assert "skin_over_smoothed" in result["hard_failures"]
    measurements = result["measurements"]
    assert measurements["reference_skin_texture_p75_median"] >= 1.5
    assert measurements["skin_texture_ratio"] < measurements[
        "skin_texture_ratio_min"
    ]


def test_skin_over_smoothed_stays_silent_when_texture_is_preserved(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(7)
    textured_ref = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    textured_gen = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    ref_path = tmp_path / "ref.png"
    gen_path = tmp_path / "gen.png"
    cv2.imwrite(str(ref_path), textured_ref)
    cv2.imwrite(str(gen_path), textured_gen)

    embedding = np.array([1.0, 0.0, 0.0])
    app = _FakeIdentityApp([
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
        _fake_face(_BBOX, np.array(_REF_KPS, dtype=np.float32), embedding),
    ])
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: app  # type: ignore[assignment]

    result = _identity_check(eval_svc, gen_path, [ref_path])

    assert "skin_over_smoothed" not in result["hard_failures"]
    assert result["measurements"]["skin_texture_ratio"] >= result[
        "measurements"
    ]["skin_texture_ratio_min"]
