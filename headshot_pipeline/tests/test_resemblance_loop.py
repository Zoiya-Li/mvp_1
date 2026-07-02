"""Deterministic regression tests for the resemblance-agent loop (Task #54).

Two layers, NO Chrome / NO Gemini required:

1. ``_parse_judge_response`` — pure function. Covers every score-extraction
   pattern, range clamping, structured-feedback capture, and the score<8
   full-text fallback.

2. ``execute_generate_with_resemblance_loop`` — the ACTUAL control flow is
   driven by a FakeClient that returns canned judge texts, verifying:
     - accept-fast (score>=threshold on first judge),
     - revise-then-accept,
     - max-iterations-reached (accept last image),
     - judge-call failure (accept current, score=None).
   The worker is built via ``__new__`` so its ``__init__`` (which connects to
   Chrome) never runs.

Run:  python -m pytest headshot_pipeline/tests/test_resemblance_loop.py -q
  or:  python headshot_pipeline/tests/test_resemblance_loop.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

# Make the package importable whether run from mvp_1/ or the pipeline dir.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

import server.gemini_worker as gemini_worker_module  # noqa: E402
from server.evaluation import EvaluationService, AgentRouter  # noqa: E402
from server.gemini_worker import (  # noqa: E402
    GeminiWorker,
    IDENTITY_PASS_THRESHOLD,
    IDENTITY_REPAIR_THRESHOLD,
    MAX_RESEMBLANCE_ITERATIONS,
    RESEMBLANCE_THRESHOLD,
    build_identity_pack_metadata,
    build_shot_spec_metadata,
    identity_threshold_profile,
)
from server.repair import FaceSwapRepair  # noqa: E402
from server.models import (  # noqa: E402
    AgentAction,
    Candidate,
    EvaluationResult,
    FinalAsset,
    IdentityPack,
    PhotoJob,
    ProviderInvocation,
    ShotSpec,
    StyleTemplate,
    UserFeedbackRecord,
)


# ──────────────────────────────────────────────────────────────────────
# Layer 1: the parser (pure)
# ──────────────────────────────────────────────────────────────────────

PARSE_CASES = [
    # (label, judge_text, expected_score, expect_feedback_truthy)
    ("canonical full-width colon", "评分：8/10\n很好。", 8, False),
    ("half-width colon", "评分:7/10", 7, True),          # 7<8 → feedback=text
    ("bare x/10 no label", "对比下来我给6/10分。", 6, True),
    ("评分N分 (no slash)", "评分9分，相当像。", 9, False),
    ("相似度：N", "相似度：7", 7, True),
    ("N分，满分10", "整体8分，满分10。", 8, False),
    ("trigger keyword captures feedback", "评分：5/10\n需要调整：\n- 眼睛偏圆\n- 鼻翼宽", 5, True),
    ("clamps >10 down to 10", "评分：12/10", 10, False),
    ("clamps 0 up to 1", "评分：0/10", 1, True),         # 1<8 → feedback
    ("no score anywhere -> None", "我无法评估这张图片。", None, False),
    ("empty text -> (None,None)", "", None, False),
    ("score>=8 with no trigger -> feedback None", "评分：9/10\n非常相似，无需调整。", 9, False),
    # ── DECIMAL SCORES (regression guard for the 9.5→5 mis-score bug) ──
    # gemini-3.1-flash-image-preview returns decimals despite the integer prompt.
    # The old integer-only `(\d+)` patterns broke on these: pattern 1 failed
    # (the ".5" sat between the digit and "/10"), then the fallback greedily
    # matched the "5" in "9.5" → score=5 → needless revision of an excellent
    # image. Round-half-up via int(x+0.5): 9.5→10, 9.2→9, 8.5→9.
    ("decimal 9.5/10 rounds UP to 10", "评分：9.5/10", 10, False),
    ("decimal 9.2/10 rounds DOWN to 9", "评分：9.2/10\n细节很好。", 9, False),
    ("decimal 8.5/10 rounds UP to 9 (accepts)", "评分：8.5/10", 9, False),
]


def test_parse_judge_response_table():
    fails = []
    for label, text, exp_score, exp_fb in PARSE_CASES:
        score, feedback = GeminiWorker._parse_judge_response(text)
        if score != exp_score:
            fails.append(f"{label}: score got {score!r} want {exp_score!r}")
        if exp_fb is not None:
            # feedback truthiness expectation (bool)
            got_fb = bool(feedback)
            if got_fb != exp_fb:
                fails.append(f"{label}: feedback truthy got {got_fb} want {exp_fb}")
        else:
            # When we did not assert feedback, only insist it is None-ish on
            # the canonical accept cases where the text carries no trigger and
            # score>=8 (already encoded as exp_fb False above for those rows).
            pass
    assert not fails, "\n  " + "\n  ".join(fails)


def test_parse_feedback_is_structured_when_keyword_present():
    text = "评分：4/10\n具体来说：\n1. 脸型偏宽\n2. 眼睛不像"
    _score, feedback = GeminiWorker._parse_judge_response(text)
    assert feedback is not None
    assert "脸型" in feedback and "眼睛" in feedback


def test_parse_feedback_fulltext_fallback_when_low_score_no_keyword():
    # Score < 8 but no trigger keyword -> whole text becomes the feedback so the
    # revision prompt still has something concrete to send.
    text = "这张照片和本人差距较大，6/10"
    score, feedback = GeminiWorker._parse_judge_response(text)
    assert score == 6
    assert feedback is not None and "差距" in feedback


# ──────────────────────────────────────────────────────────────────────
# Layer 2: the loop control flow (real method, mock client)
# ──────────────────────────────────────────────────────────────────────

class FakeClient:
    """Stand-in for an ImageProvider. Records every call.

    Implements the ImageProvider surface so the loop control flow is exercised
    against the real worker with NO network and NO API key.
    """

    def __init__(self, judge_texts: list[str], judge_exc: Exception | None = None):
        self._judge_texts = list(judge_texts)
        self.judge_exc = judge_exc
        self.start_calls = 0
        self.converse_calls = 0
        self.judge_calls = 0

    def create_from_references(self, prompt, reference_paths, template_path, title, editing_mode=True):
        self.start_calls += 1
        return "/tmp/fake_initial.png"

    def local_edit(self, current_image_path, reference_paths, edit_prompt, title):
        self.converse_calls += 1
        return f"/tmp/fake_iter_{self.converse_calls}.png"

    def judge(self, current_image_path, reference_paths, judge_prompt, timeout=None):
        self.judge_calls += 1
        if self.judge_exc is not None:
            raise self.judge_exc
        return self._judge_texts.pop(0)

    def upscale(self, image_path):
        return image_path

    def end_session(self):
        pass


def _make_worker(judge_texts, judge_exc=None):
    """Build a GeminiWorker WITHOUT connecting to Chrome (skip __init__)."""
    w = GeminiWorker.__new__(GeminiWorker)
    w.active_session_id = None
    w._turn_counts = {}
    w._eval_service = EvaluationService()
    w._agent_router = AgentRouter(identity_threshold_profile)
    w._face_swap_repair = FaceSwapRepair()
    w._face_swap_repair._load_failed = True  # tests have no model; skip face-swap
    # Build a minimal gateway that routes to our FakeClient
    from server.generation.gateway import ImageGateway
    gw = ImageGateway.__new__(ImageGateway)
    fake_provider = FakeClient(judge_texts, judge_exc=judge_exc)
    gw._openrouter = fake_provider
    gw._chrome = None
    w._gateway = gw
    w._ensure_session = lambda *a, **k: None  # type: ignore[assignment]
    return w


def _make_loader_worker():
    w = GeminiWorker.__new__(GeminiWorker)
    w._face_swap_repair = FaceSwapRepair()
    w._face_swap_repair._load_failed = False
    w._identity_app = None
    w._identity_app_load_failed = False
    w._eval_service = EvaluationService()
    return w


def test_face_swapper_lazy_loader_loads_and_caches_model(tmp_path, monkeypatch):
    model_path = tmp_path / "inswapper_128.onnx"
    model_path.write_bytes(b"model")
    loaded_paths = []

    class FakeFaceSwapper:
        def __init__(self, path):
            loaded_paths.append(path)

    import server.repair.identity_repair as repair_module

    monkeypatch.setattr(repair_module.settings, "face_swap_enabled", True)
    monkeypatch.setattr(repair_module.settings, "face_swap_model_path", model_path)
    monkeypatch.setattr(repair_module, "FaceSwapper", FakeFaceSwapper)

    from server.repair import FaceSwapRepair

    repair = FaceSwapRepair()
    first = repair._get_swapper()
    second = repair._get_swapper()

    assert isinstance(first, FakeFaceSwapper)
    assert second is first
    assert loaded_paths == [model_path]
    assert repair._load_failed is False


def test_face_swapper_lazy_loader_marks_missing_model_unavailable(tmp_path, monkeypatch):
    import server.repair.identity_repair as repair_module

    monkeypatch.setattr(repair_module.settings, "face_swap_enabled", True)
    monkeypatch.setattr(
        repair_module.settings,
        "face_swap_model_path",
        tmp_path / "missing.onnx",
    )
    from server.repair import FaceSwapRepair

    repair = FaceSwapRepair()
    assert repair._get_swapper() is None
    assert repair._load_failed is True


def test_loop_accepts_fast_when_score_meets_threshold():
    w = _make_worker(["评分：9/10\n非常相似。"])
    fp, meta = w.execute_generate_with_resemblance_loop(
        "s1", "prompt", ["a.jpg", "b.jpg"], "title", template_path=None
    )
    assert meta["iterations"] == 1
    assert meta["final_score"] == 9
    assert meta["history"][0]["accepted"] is True
    # Accepted on first judge → NO revision turn should be requested.
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").converse_calls == 0
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 1


def test_loop_revises_then_accepts():
    w = _make_worker(["评分：5/10\n需要调整：眼睛偏圆", "评分：8/10"])
    _fp, meta = w.execute_generate_with_resemblance_loop(
        "s2", "prompt", ["a.jpg"], "title"
    )
    assert meta["iterations"] == 2
    assert meta["final_score"] == 8
    assert meta["history"][1]["accepted"] is True
    assert meta["history"][0]["accepted"] is False   # iter 1 was a revise
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").converse_calls == 1              # exactly one revision
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 2


def test_loop_reaches_max_iterations_and_accepts_last():
    # Three sub-threshold scores → 3 iterations, accept the last image.
    w = _make_worker([
        "评分：5/10\n需要调整：脸型",
        "评分：6/10\n差异：鼻子",
        "评分：6/10\n不像：眉毛",
    ])
    _fp, meta = w.execute_generate_with_resemblance_loop(
        "s3", "prompt", ["a.jpg"], "title"
    )
    assert meta["iterations"] == MAX_RESEMBLANCE_ITERATIONS  # 3
    assert meta["final_score"] == 6
    # The final iteration must be accepted (max_reached), and it must NOT have
    # requested another revision (i>=MAX breaks before the revise branch).
    assert meta["history"][-1]["accepted"] is True
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").converse_calls == MAX_RESEMBLANCE_ITERATIONS - 1  # 2
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == MAX_RESEMBLANCE_ITERATIONS         # 3


def test_loop_retries_fast_judge_failure_then_accepts_current():
    # converse_text throws (e.g. Gemini returned an image instead of text, or a
    # DOM glitch). FakeClient raises INSTANTLY → a "fast" failure → the loop
    # retries once (judge_calls == 2) before falling back to accept-current.
    # The loop must still swallow it and accept the current image rather than
    # crash the whole job. (A SLOW timeout — one that burns the full budget —
    # is NOT retried, to avoid doubling worst-case latency; that path shares
    # this accept-current fallback but would show judge_calls == 1.)
    w = _make_worker([], judge_exc=RuntimeError("text-response timeout"))
    _fp, meta = w.execute_generate_with_resemblance_loop(
        "s4", "prompt", ["a.jpg"], "title"
    )
    assert meta["iterations"] == 1
    assert meta["final_score"] is None
    assert meta["history"][0]["accepted"] is True
    assert meta["history"][0]["error"] is not None
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").converse_calls == 0
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 2   # fast failure retried once


def test_threshold_constant_is_8():
    # Guard the magic number: the whole accept criterion depends on it.
    assert RESEMBLANCE_THRESHOLD == 8
    assert IDENTITY_PASS_THRESHOLD == 8
    assert IDENTITY_REPAIR_THRESHOLD == 7


# ──────────────────────────────────────────────────────────────────────
# Layer 3: controlled candidate pipeline helpers (pure)
# ──────────────────────────────────────────────────────────────────────

def test_quality_judge_json_parse_normalizes_scores_and_action():
    text = """
    {
      "scores": {
        "identity": 8.6,
        "face_quality": "9",
        "style_match": 7,
        "artifact": 10,
        "commercial_readiness": 8
      },
      "hard_failures": [],
      "recommended_action": "accept",
      "notes": "Strong candidate"
    }
    """
    out = EvaluationService._parse_quality_judge_response(text)
    assert out["scores"]["identity"] == 9
    assert out["scores"]["face_quality"] == 9
    assert out["recommended_action"] == "accept"
    assert out["hard_failures"] == []


def test_quality_judge_falls_back_to_old_score_parser():
    out = EvaluationService._parse_quality_judge_response("评分：7/10\n需要调整：脸型")
    assert out["scores"]["identity"] == 7
    assert out["scores"]["commercial_readiness"] == 7
    assert out["recommended_action"] == "retry"


def test_aggregate_quality_penalizes_hard_failures():
    clean = {
        "scores": {
            "identity": 8,
            "face_quality": 8,
            "style_match": 8,
            "artifact": 8,
            "commercial_readiness": 8,
        },
        "hard_failures": [],
        "recommended_action": "accept",
    }
    failed = {
        **clean,
        "hard_failures": ["face_distorted"],
        "recommended_action": "retry",
    }
    assert EvaluationService._aggregate_quality_score(clean) > EvaluationService._aggregate_quality_score(failed)


def test_aggregate_quality_is_clamped_to_ten():
    perfect = {
        "scores": {
            "identity": 10,
            "face_quality": 10,
            "style_match": 10,
            "artifact": 10,
            "commercial_readiness": 10,
        },
        "hard_failures": [],
        "recommended_action": "accept",
    }
    assert EvaluationService._aggregate_quality_score(perfect) == 10.0


def test_merge_local_quality_caps_vlm_scores_and_discards_severe_failures():
    judgement = {
        "scores": {
            "identity": 9,
            "face_quality": 10,
            "style_match": 9,
            "artifact": 10,
            "commercial_readiness": 10,
        },
        "hard_failures": [],
        "recommended_action": "accept",
        "notes": "VLM liked it",
    }
    local = {
        "scores": {
            "face_quality": 3,
            "artifact": 6,
            "commercial_readiness": 3,
        },
        "hard_failures": ["no_face"],
        "measurements": {"face_count": 0},
        "notes": "no_face",
    }
    merged = EvaluationService._merge_local_quality(judgement, local)
    assert merged["scores"]["identity"] == 9
    assert merged["scores"]["face_quality"] == 3
    assert merged["scores"]["artifact"] == 6
    assert merged["recommended_action"] == "discard"
    assert "no_face" in merged["hard_failures"]
    assert merged["local_quality"]["measurements"]["face_count"] == 0
    qa = merged["quality_evaluation"]
    assert qa["face_quality"]["status"] == "fail"
    assert qa["face_quality"]["issues"] == ["no_face"]
    assert qa["face_quality"]["measurements"]["face_count"] == 0
    assert qa["artifacts"]["score"] == 6
    assert qa["safety"]["status"] == "pass"


def test_identity_cosine_mapping_has_acceptance_gate():
    assert EvaluationService._identity_cosine_to_score(0.60) == 10
    assert EvaluationService._identity_cosine_to_score(0.53) == 9
    assert EvaluationService._identity_cosine_to_score(0.46) == 8
    assert EvaluationService._identity_cosine_to_score(0.41) == 7
    assert EvaluationService._identity_cosine_to_score(0.20) == 3


def test_local_identity_similarity_uses_six_identity_pack_references(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    seen_images = []

    def fake_imread(path):
        return str(path)

    class FakeIdentityApp:
        def get(self, img):
            seen_images.append(img)
            return [
                SimpleNamespace(
                    bbox=[0, 0, 10, 10],
                    normed_embedding=np.array([1.0, 0.0, 0.0]),
                )
            ]

    monkeypatch.setattr(cv2, "imread", fake_imread)
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: FakeIdentityApp()  # type: ignore[assignment]

    result = eval_svc._local_identity_similarity_check(
        "/tmp/generated.png",
        [f"/tmp/ref_{idx}.png" for idx in range(1, 8)],
    )

    assert result["measurements"]["reference_count"] == 6
    assert result["measurements"]["reference_faces_detected"] == 6
    assert seen_images[:6] == [f"/tmp/ref_{idx}.png" for idx in range(1, 7)]
    assert seen_images[6] == "/tmp/generated.png"


def test_merge_identity_quality_caps_vlm_identity_and_requests_repair():
    judgement = {
        "scores": {
            "identity": 10,
            "face_quality": 9,
            "style_match": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": [],
        "recommended_action": "accept",
        "notes": "VLM perfect",
    }
    local_identity = {
        "score": 7,
        "cosine_similarity": 0.41,
        "reference_consistency": 0.55,
        "hard_failures": ["identity_too_low"],
        "measurements": {},
        "notes": "identity_too_low",
    }
    merged = EvaluationService._merge_identity_quality(judgement, local_identity)
    assert merged["scores"]["identity"] == 7
    assert merged["recommended_action"] == "face_swap"
    assert "identity_too_low" in merged["hard_failures"]
    assert merged["identity_quality"]["cosine_similarity"] == 0.41
    qa = merged["quality_evaluation"]
    assert qa["identity"]["score"] == 7
    assert qa["identity"]["status"] == "fail"
    assert qa["identity"]["cosine_similarity"] == 0.41
    assert qa["identity"]["reference_consistency"] == 0.55
    assert qa["identity"]["issues"] == ["identity_too_low"]
    assert qa["prompt_adherence"]["score"] == 0.9
    assert qa["aesthetic"]["score"] == 0.9


def test_local_image_quality_rejects_blank_image(tmp_path):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), blank)

    out = EvaluationService._local_image_quality_check(str(path))
    assert "no_face" in out["hard_failures"]
    assert out["scores"]["face_quality"] <= 3


def test_select_candidate_prefers_identity_hard_gate_over_aggregate_score():
    pretty_but_wrong_person = {
        "aggregate_score": 9.5,
        "gate_status": {"hard_gates_pass": False},
        "agent_action": {"action": "REGENERATE_FROM_ORIGINAL"},
    }
    identity_safe = {
        "aggregate_score": 7.0,
        "gate_status": {"hard_gates_pass": True},
        "agent_action": {"action": "ACCEPT"},
    }
    assert (
        GeminiWorker._select_candidate([pretty_but_wrong_person, identity_safe])
        is identity_safe
    )


def test_candidate_action_uses_bounded_identity_state_machine():
    assert GeminiWorker._decide_candidate_action({
        "scores": {"identity": 9, "face_quality": 9, "artifact": 9, "commercial_readiness": 9},
        "hard_failures": [],
        "recommended_action": "accept",
    })["action"] == "ACCEPT"
    assert GeminiWorker._decide_candidate_action({
        "scores": {"identity": 7, "style_match": 9},
        "hard_failures": [],
        "recommended_action": "retry",
    })["action"] == "IDENTITY_REPAIR"
    assert GeminiWorker._decide_candidate_action({
        "scores": {"identity": 5, "style_match": 9},
        "hard_failures": [],
        "recommended_action": "retry",
    })["action"] == "REGENERATE_FROM_ORIGINAL"


def test_candidate_gate_treats_safety_as_hard_delivery_gate():
    gate = EvaluationService._candidate_gate_status({
        "scores": {
            "identity": 10,
            "face_quality": 10,
            "artifact": 10,
            "commercial_readiness": 10,
        },
        "hard_failures": ["unsafe_content"],
        "recommended_action": "accept",
    })

    assert gate["safety_pass"] is False
    assert gate["hard_gates_pass"] is False
    assert gate["hard_gate_failures"] == ["unsafe_content"]


def test_candidate_gate_lists_all_hard_gate_failures():
    gate = EvaluationService._candidate_gate_status({
        "scores": {
            "identity": 6,
            "face_quality": 6,
            "artifact": 6,
            "commercial_readiness": 6,
        },
        "hard_failures": ["no_face", "bad_artifacts"],
        "recommended_action": "retry",
    })

    assert gate["hard_gates_pass"] is False
    assert gate["hard_gate_failures"] == [
        "no_usable_face_detected",
        "identity_fail",
        "quality_below_threshold",
        "severe_quality_failure",
    ]


def test_identity_threshold_profile_varies_by_shot_geometry():
    closeup = identity_threshold_profile({
        "shot_id": "closeup",
        "framing": "close-up portrait",
    })
    medium = identity_threshold_profile({
        "shot_id": "half_body",
        "framing": "medium shot",
    })
    small_face = identity_threshold_profile({
        "shot_id": "environmental",
        "framing": "full body",
    })
    side = identity_threshold_profile({
        "shot_id": "profile",
        "pose": "left 45 side profile",
    })

    assert closeup["profile"] == "closeup"
    assert closeup["identity_pass_threshold"] == 8
    assert closeup["identity_repair_threshold"] == 7
    assert medium["profile"] == "medium"
    assert medium["identity_pass_threshold"] == 7.5
    assert medium["identity_repair_threshold"] == 6.5
    assert small_face["profile"] == "small_face"
    assert small_face["identity_pass_threshold"] == 7
    assert small_face["identity_repair_threshold"] == 6
    assert side["profile"] == "side_profile"


def test_candidate_gate_uses_shot_specific_identity_thresholds():
    judgement = {
        "scores": {
            "identity": 7,
            "face_quality": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": [],
        "recommended_action": "accept",
    }

    closeup_gate = EvaluationService._candidate_gate_status(
        judgement,
        identity_threshold_profile({"shot_id": "closeup"}),
    )
    environmental_gate = EvaluationService._candidate_gate_status(
        judgement,
        identity_threshold_profile({
            "shot_id": "environmental",
            "framing": "full body",
        }),
    )

    assert closeup_gate["identity_pass"] is False
    assert closeup_gate["hard_gates_pass"] is False
    assert closeup_gate["identity_pass_threshold"] == 8
    assert environmental_gate["identity_pass"] is True
    assert environmental_gate["hard_gates_pass"] is True
    assert environmental_gate["identity_pass_threshold"] == 7
    assert environmental_gate["identity_threshold_profile"] == "small_face"


def test_candidate_action_uses_shot_specific_gray_zone():
    judgement = {
        "scores": {"identity": 7, "style_match": 9},
        "hard_failures": [],
        "recommended_action": "retry",
    }

    assert GeminiWorker._decide_candidate_action(
        judgement,
        identity_thresholds=identity_threshold_profile({"shot_id": "closeup"}),
    )["action"] == "IDENTITY_REPAIR"
    assert GeminiWorker._decide_candidate_action(
        judgement,
        identity_thresholds=identity_threshold_profile({
            "shot_id": "environmental",
            "framing": "full body",
        }),
    )["action"] == "ACCEPT"


def test_candidate_shortlist_records_top_two_without_paths():
    candidates = [
        {
            "index": 1,
            "candidate_id": "cand_1",
            "filename": "a.png",
            "path": "/tmp/a.png",
            "aggregate_score": 6.0,
            "gate_status": {
                "hard_gates_pass": False,
                "hard_gate_failures": ["identity_fail"],
            },
            "agent_action": {
                "action": "REGENERATE_FROM_ORIGINAL",
                "reason": "identity_below_repair_threshold",
            },
        },
        {
            "index": 2,
            "candidate_id": "cand_2",
            "filename": "b.png",
            "path": "/tmp/b.png",
            "aggregate_score": 8.0,
            "gate_status": {"hard_gates_pass": True, "hard_gate_failures": []},
            "agent_action": {"action": "ACCEPT", "reason": "all_hard_gates_pass"},
            "selected": True,
        },
        {
            "index": 3,
            "candidate_id": "cand_3",
            "filename": "c.png",
            "path": "/tmp/c.png",
            "aggregate_score": 7.0,
            "gate_status": {"hard_gates_pass": True, "hard_gate_failures": []},
            "agent_action": {"action": "ACCEPT", "reason": "all_hard_gates_pass"},
        },
    ]

    shortlist = GeminiWorker._candidate_shortlist(candidates)

    assert [item["candidate_id"] for item in shortlist] == ["cand_2", "cand_3"]
    assert shortlist[0]["rank"] == 1
    assert shortlist[0]["selected"] is True
    assert "path" not in shortlist[0]


def test_candidate_action_drops_unsafe_content_before_repair_or_edit():
    action = GeminiWorker._decide_candidate_action({
        "scores": {
            "identity": 10,
            "face_quality": 10,
            "artifact": 10,
            "commercial_readiness": 10,
        },
        "hard_failures": ["unsafe_content"],
        "recommended_action": "accept",
    })

    assert action == {"action": "DROP_CANDIDATE", "reason": "unsafe_content"}


def test_identity_repair_skips_when_candidate_passes_identity_gate():
    judgement = {
        "scores": {"identity": 9},
        "hard_failures": [],
        "recommended_action": "accept",
    }
    assert GeminiWorker._should_apply_identity_repair(judgement) is False


def test_identity_repair_runs_only_for_identity_gray_zone():
    gray_zone_identity = {
        "scores": {"identity": 7},
        "hard_failures": [],
        "recommended_action": "retry",
    }
    below_repair_threshold = {
        "scores": {"identity": 6},
        "hard_failures": [],
        "recommended_action": "face_swap",
    }
    passing_identity_with_explicit_action = {
        "scores": {"identity": 9},
        "hard_failures": [],
        "recommended_action": "face_swap",
    }
    unverified_identity = {
        "scores": {"identity": None},
        "hard_failures": [],
        "recommended_action": "retry",
    }
    assert GeminiWorker._should_apply_identity_repair(gray_zone_identity) is True
    assert GeminiWorker._should_apply_identity_repair(below_repair_threshold) is False
    assert (
        GeminiWorker._should_apply_identity_repair(
            passing_identity_with_explicit_action
        )
        is False
    )
    assert GeminiWorker._should_apply_identity_repair(unverified_identity) is True


def test_identity_pack_assigns_reference_roles_without_storing_embeddings():
    pack = build_identity_pack_metadata([
        "/tmp/front-neutral.jpg",
        "/tmp/front-smile.jpg",
        "/tmp/left.jpg",
        "/tmp/right.jpg",
        "/tmp/lifestyle.jpg",
        "/tmp/profile.jpg",
    ])

    roles = [ref["role"] for ref in pack["reference_images"]]
    assert roles == [
        "front_neutral",
        "front_smile",
        "left_45",
        "right_45",
        "lifestyle",
        "side_profile",
    ]
    assert pack["primary_reference_ids"] == ["ref_1", "ref_2", "ref_3", "ref_4"]
    assert pack["temporary_face_template"]["storage"] == "in_memory_task_scope"
    assert pack["temporary_face_template"]["stores_embedding_in_metadata"] is False
    assert pack["temporary_face_template"]["built_from_reference_ids"] == [
        "ref_1",
        "ref_2",
        "ref_3",
        "ref_4",
    ]
    assert pack["cross_user_search"] is False
    assert pack["persistent_face_library"] is False
    assert pack["version"] == "identity_pack_v2"


def test_pipeline_contract_models_validate_worker_metadata():
    pack = IdentityPack(**build_identity_pack_metadata([
        "/tmp/front-neutral.jpg",
        "/tmp/front-smile.jpg",
        "/tmp/left.jpg",
        "/tmp/right.jpg",
    ]))
    shot = ShotSpec(**build_shot_spec_metadata(
        "studio portrait prompt",
        "business_closeup",
        template_path="/tmp/template.png",
    ))
    style = StyleTemplate(
        style_id="business",
        template_id="template",
        prompt_version="controlled_candidate_v2",
    )
    invocation = ProviderInvocation(
        invocation_id="create_1",
        provider="openrouter",
        model="gemini",
        operation="CREATE_FROM_REFERENCES",
        prompt_version=style.prompt_version,
        reference_ids=pack.primary_reference_ids,
        latency_ms=17830,
        cost=0.14,
        estimated_cost=0.14,
        result_status="success",
    )
    judgement = EvaluationResult(
        scores={"identity": 9, "face_quality": 8, "style_match": 8},
        recommended_action="ACCEPT",
        hard_failures=[],
    )
    action = AgentAction(
        action="ACCEPT",
        reason="hard_gates_passed",
        candidate_id="cand_1",
        candidate_index=1,
        state="evaluated",
        executed=True,
    )
    candidate = Candidate(
        index=1,
        candidate_id="cand_1",
        filename="candidate.png",
        judgement=judgement,
        gate_status={
            "safety_pass": True,
            "face_detected": True,
            "identity_pass": True,
            "quality_pass": True,
            "hard_gates_pass": True,
        },
        agent_action=action,
        provider_invocation_id=invocation.invocation_id,
        selected=True,
    )
    final = FinalAsset(
        image_id="img_1",
        candidate_id=candidate.candidate_id,
        filename="final.png",
        deliverable=True,
        visible_ai_label=True,
        metadata_ai_label=True,
    )
    feedback = UserFeedbackRecord(
        session_id="s_contract",
        image_id=final.image_id,
        event="downloaded",
        score=2,
    )

    job = PhotoJob(
        job_id="job_1",
        session_id="s_contract",
        identity_pack=pack,
        shot_specs=[shot],
        candidates=[candidate],
        final_assets=[final],
        user_feedback=[feedback],
        provider_invocations=[invocation],
    )

    assert job.identity_pack.reference_images[0].role == "front_neutral"
    assert job.identity_pack.temporary_face_template.stores_embedding_in_metadata is False
    assert job.shot_specs[0].prompt_blocks.identity_block == "derived_from_identity_pack"
    assert job.provider_invocations[0].operation == "CREATE_FROM_REFERENCES"
    assert job.provider_invocations[0].cost == 0.14
    assert job.candidates[0].judgement.scores.identity == 9
    assert job.candidates[0].agent_action.action == "ACCEPT"
    assert job.final_assets[0].operation == "FINAL_RENDER"
    assert job.user_feedback[0].event == "downloaded"


class FakePipelineClient:
    """ImageProvider-compatible fake for pipeline tests."""

    def __init__(self, judge_texts: list[str]):
        self._judge_texts = list(judge_texts)
        self.start_calls = 0
        self.judge_calls = 0
        self.converse_calls = 0
        self._last_image_path = None
        self.photo_counts = []

    def create_from_references(self, prompt, reference_paths, template_path, title, editing_mode=True):
        self.start_calls += 1
        self.photo_counts.append(len(reference_paths))
        self._last_image_path = f"/tmp/{title}.png"
        return self._last_image_path

    def local_edit(self, current_image_path, reference_paths, edit_prompt, title):
        self.converse_calls += 1
        self._last_image_path = f"/tmp/{title}.png"
        return self._last_image_path

    def judge(self, current_image_path, reference_paths, judge_prompt, timeout=None):
        self.judge_calls += 1
        return self._judge_texts.pop(0)

    def upscale(self, image_path):
        return image_path

    def end_session(self):
        pass


def _make_pipeline_worker(judge_texts, swap_result=None):
    w = GeminiWorker.__new__(GeminiWorker)
    w.active_session_id = None
    w._turn_counts = {}
    w._face_swap_repair = FaceSwapRepair()
    w._face_swap_repair._load_failed = True
    w._ensure_session = lambda *a, **k: None  # type: ignore[assignment]
    w._eval_service = EvaluationService()
    w._agent_router = AgentRouter(identity_threshold_profile)
    w.swap_calls = 0

    # Build gateway with FakePipelineClient
    from server.generation.gateway import ImageGateway
    gw = ImageGateway.__new__(ImageGateway)
    fake_provider = FakePipelineClient(judge_texts)
    gw._openrouter = fake_provider
    gw._chrome = None
    w._gateway = gw

    def fake_swap(_generated_path, _photo_paths, _title):
        w.swap_calls += 1
        if swap_result is not None:
            return swap_result
        from server.repair.identity_repair import FaceSwapResult
        return FaceSwapResult(
            output_path=Path("/tmp/not_swapped.png"),
            swapped=False,
            message="not needed",
            source_face_count=1,
            target_face_count=1,
        )

    w._apply_face_swap = fake_swap  # type: ignore[assignment]
    return w


def _judge_json(identity: int, action: str = "accept") -> str:
    return (
        '{"scores":{"identity":%d,"face_quality":8,"style_match":8,'
        '"artifact":8,"commercial_readiness":8},'
        '"hard_failures":[],"recommended_action":"%s","notes":"ok"}'
    ) % (identity, action)


def _judge_json_scores(
    *,
    identity: int = 9,
    face_quality: int = 8,
    style_match: int = 8,
    artifact: int = 8,
    commercial_readiness: int = 8,
    action: str = "accept",
) -> str:
    return (
        '{"scores":{"identity":%d,"face_quality":%d,"style_match":%d,'
        '"artifact":%d,"commercial_readiness":%d},'
        '"hard_failures":[],"recommended_action":"%s","notes":"local issue"}'
    ) % (
        identity,
        face_quality,
        style_match,
        artifact,
        commercial_readiness,
        action,
    )


def test_quality_pipeline_skips_face_swap_for_high_identity_candidate():
    w = _make_pipeline_worker([_judge_json(9), _judge_json(8), _judge_json(8)])
    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s1", "prompt", ["a.jpg"], "title", template_path=None
    )
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").start_calls == 3
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 3
    assert w.swap_calls == 0
    assert meta["pipeline"] == "controlled_candidate_v2"
    assert meta["budget"]["initial_candidates"] == 3
    assert meta["budget"]["max_total_api_cost"] == 1.0
    assert meta["budget"]["estimated_cost_used"] == 0.36
    assert meta["identity_pack"]["persistent_face_library"] is False
    assert [
        ref["role"]
        for ref in meta["identity_pack"]["reference_images"]
    ] == ["front_neutral"]
    assert meta["provider_invocations"][0]["operation"] == "CREATE_FROM_REFERENCES"
    assert meta["provider_invocations"][0]["provider"] == "openrouter"
    assert meta["provider_invocations"][0]["reference_ids"] == ["ref_1"]
    assert meta["provider_invocations"][0]["cost"] == 0.12
    assert meta["provider_invocations"][0]["estimated_cost"] == 0.12
    assert meta["provider_invocations"][0]["reference_roles"][0]["role"] == "front_neutral"
    assert meta["provider_invocations"][0]["reference_roles"][0]["usage"] == "primary_identity"
    assert meta["provider_invocations"][0]["provider_capabilities"][
        "supports_multiple_references"
    ] is True
    assert meta["agent_actions"][0]["action"] == "ACCEPT"
    assert meta["shortlist"][0]["candidate_id"] == "cand_1"
    assert meta["shortlist"][0]["hard_gates_pass"] is True
    assert meta["shortlist"][0]["selected"] is True
    assert "path" not in meta["shortlist"][0]
    assert meta["face_swap"]["action"] == "none"
    assert "path" not in meta["candidates"][0]
    assert "path" not in meta["candidates"][1]
    assert "path" not in meta["candidates"][2]


def test_quality_pipeline_budget_stops_initial_candidate_generation(monkeypatch):
    monkeypatch.setattr(gemini_worker_module, "MAX_PIPELINE_TOTAL_API_COST", 0.13)
    w = _make_pipeline_worker([_judge_json(9)])

    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_budget", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").start_calls == 1
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 1
    assert len(meta["candidates"]) == 1
    assert len(meta["provider_invocations"]) == 1
    assert meta["budget"]["initial_candidates"] == 3
    assert meta["budget"]["initial_candidates_generated"] == 1
    assert meta["budget"]["estimated_cost_used"] == 0.12
    assert any(
        action["action"] == "DROP_CANDIDATE"
        and action["reason"] == "max_total_api_cost_reached"
        and action["state"] == "BUDGET_CHECK"
        and action["candidate_index"] == 2
        and action["executed"] is True
        for action in meta["agent_actions"]
    )


def test_quality_pipeline_keeps_six_reference_identity_pack_but_generates_with_four():
    w = _make_pipeline_worker([_judge_json(9), _judge_json(8), _judge_json(8)])
    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_refs",
        "prompt",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"],
        "title",
        template_path=None,
    )

    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").photo_counts == [4, 4, 4]
    assert [
        ref["role"]
        for ref in meta["identity_pack"]["reference_images"]
    ] == [
        "front_neutral",
        "front_smile",
        "left_45",
        "right_45",
        "lifestyle",
        "side_profile",
    ]
    assert meta["provider_invocations"][0]["reference_ids"] == [
        "ref_1",
        "ref_2",
        "ref_3",
        "ref_4",
    ]
    assert [
        ref["role"]
        for ref in meta["provider_invocations"][0]["reference_roles"]
    ] == ["front_neutral", "front_smile", "left_45", "right_45"]


def test_quality_pipeline_records_shot_specific_identity_thresholds():
    w = _make_pipeline_worker([_judge_json(8), _judge_json(8), _judge_json(8)])
    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_thresholds",
        "prompt",
        ["a.jpg"],
        "title",
        template_path=None,
        shot_spec_metadata={
            "shot_id": "half_body",
            "framing": "medium shot",
            "pose": "standing naturally",
        },
    )

    profile = meta["strategy"]["identity_threshold_profile"]
    assert profile["profile"] == "medium"
    assert profile["identity_pass_threshold"] == 7.5
    assert profile["identity_repair_threshold"] == 6.5
    assert meta["strategy"]["identity_pass_threshold"] == 7.5
    assert meta["candidates"][0]["gate_status"]["identity_threshold_profile"] == "medium"
    assert meta["candidates"][0]["gate_status"]["identity_pass_threshold"] == 7.5


def test_quality_pipeline_executes_regenerate_from_original_once():
    w = _make_pipeline_worker([
        _judge_json(5, "retry"),
        _judge_json(5, "retry"),
        _judge_json(5, "retry"),
        _judge_json(9, "accept"),
    ])

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s_regen", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert fp == "/tmp/title_regen1.png"
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").start_calls == 4
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 4
    assert meta["budget"]["regenerations_used"] == 1
    assert meta["budget"]["estimated_cost_used"] == 0.48
    assert meta["selected_candidate"]["candidate_id"] == "cand_4"
    assert meta["selected_candidate"]["deliverable"] is True
    assert any(
        inv["invocation_id"] == "regenerate_1"
        and inv["operation"] == "CREATE_FROM_REFERENCES"
        and inv["parent_candidate_id"] == "cand_1"
        and inv["cost"] == 0.12
        for inv in meta["provider_invocations"]
    )
    assert any(
        action["action"] == "REGENERATE_FROM_ORIGINAL"
        and action["executed"] is True
        for action in meta["agent_actions"]
    )
    assert meta["history"][-1]["regenerated_from_candidate_id"] == "cand_1"


def test_quality_pipeline_does_not_identity_repair_below_repair_threshold():
    w = _make_pipeline_worker([
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
    ])

    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_regen_fail", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").start_calls == 5
    assert w.swap_calls == 0
    assert meta["budget"]["regenerations_used"] == 2
    assert meta["budget"]["identity_repairs_used"] == 0
    assert meta["face_swap"]["action"] == "none"
    assert meta["selected_candidate"]["deliverable"] is False
    assert meta["selected_candidate"]["gate_status"]["hard_gate_failures"] == [
        "identity_fail"
    ]
    assert any(
        action["action"] == "DROP_CANDIDATE"
        and action["reason"] == "max_regenerations_reached"
        and action["executed"] is True
        for action in meta["agent_actions"]
    )


def test_quality_pipeline_repairs_gray_zone_identity_candidate_and_rescores():
    swapped = SimpleNamespace(
        output_path=Path("/tmp/swapped.png"),
        swapped=True,
        message="swapped",
        source_face_count=1,
        target_face_count=1,
    )
    w = _make_pipeline_worker(
        [
            _judge_json(7, "face_swap"),
            _judge_json(6),
            _judge_json(6),
            _judge_json(9),
        ],
        swap_result=swapped,
    )
    fp, meta = w.execute_generate_with_quality_pipeline(
        "s2", "prompt", ["a.jpg"], "title", template_path=None
    )
    assert fp == "/tmp/swapped.png"
    assert w.swap_calls == 1
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 4
    assert meta["budget"]["identity_repairs_used"] == 1
    assert any(
        action["action"] == "IDENTITY_REPAIR"
        and action["selected_for_execution"] is True
        for action in meta["agent_actions"]
    )
    assert meta["face_swap"]["applied"] is True
    assert meta["face_swap"]["output_filename"] == "swapped.png"
    assert "output_path" not in meta["face_swap"]
    assert meta["final_score"] == 9


def test_quality_pipeline_local_edits_artifact_candidate_and_rescores():
    w = _make_pipeline_worker(
        [
            _judge_json_scores(identity=9, artifact=7, commercial_readiness=8),
            _judge_json_scores(identity=6),
            _judge_json_scores(identity=6),
            _judge_json_scores(identity=9, artifact=9, commercial_readiness=9),
        ]
    )

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s3", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert fp == "/tmp/title_cand1_local_edit.png"
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").start_calls == 3
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").converse_calls == 1
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").judge_calls == 4
    assert meta["budget"]["local_edits_used"] == 1
    assert meta["local_edit"]["applied"] is True
    assert meta["local_edit"]["output_filename"] == "title_cand1_local_edit.png"
    assert any(
        inv["operation"] == "LOCAL_EDIT"
        and inv["cost"] == 0.12
        and inv["estimated_cost"] == 0.12
        for inv in meta["provider_invocations"]
    )
    assert any(
        action["action"] == "LOCAL_EDIT" and action["executed"] is True
        for action in meta["agent_actions"]
    )


def test_quality_pipeline_drops_candidate_when_local_edit_still_fails_gate():
    w = _make_pipeline_worker(
        [
            _judge_json_scores(identity=9, artifact=7, commercial_readiness=8),
            _judge_json_scores(identity=6),
            _judge_json_scores(identity=6),
            _judge_json_scores(identity=9, artifact=7, commercial_readiness=8),
        ]
    )

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s3_fail", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert fp == "/tmp/title_cand1_local_edit.png"
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").converse_calls == 1
    assert meta["budget"]["local_edits_used"] == 1
    assert meta["local_edit"]["applied"] is True
    assert meta["selected_candidate"]["deliverable"] is False
    assert meta["selected_candidate"]["gate_status"]["hard_gate_failures"] == [
        "quality_below_threshold"
    ]
    assert any(
        action["action"] == "LOCAL_EDIT" and action["executed"] is True
        for action in meta["agent_actions"]
    )
    assert any(
        action["action"] == "DROP_CANDIDATE"
        and action["reason"] == "local_edit_failed_delivery_gate"
        and action["state"] == "FINAL_EVALUATE"
        and action["executed"] is True
        and action["selected_for_execution"] is True
        and action["hard_gate_failures"] == ["quality_below_threshold"]
        for action in meta["agent_actions"]
    )


# Allow running without pytest: `python test_resemblance_loop.py`.
if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"✅ PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"❌ FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
