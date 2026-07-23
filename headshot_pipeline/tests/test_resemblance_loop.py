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
from server.config import settings  # noqa: E402
from server.evaluation import EvaluationService, AgentRouter, PolicyEngine  # noqa: E402
from server.gemini_worker import (  # noqa: E402
    GeminiWorker,
    IDENTITY_PASS_THRESHOLD,
    IDENTITY_REPAIR_THRESHOLD,
    MAX_RESEMBLANCE_ITERATIONS,
    RESEMBLANCE_THRESHOLD,
    build_composition_scaffold_prompt,
    build_editing_prompt,
    build_identity_pack_metadata,
    build_shot_spec_metadata,
    identity_threshold_profile,
    should_use_composition_first,
    should_prefer_local_reframe,
    should_prefer_local_sharpness_edit,
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


@pytest.fixture(autouse=True)
def _stable_gateway_cost_baseline(monkeypatch):
    """Keep legacy loop-budget assertions independent of a developer's .env."""
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")


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
    # gemini-3.1-flash-image returns decimals despite the integer prompt.
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


def test_editing_prompt_uses_semantic_roles_and_shot_spec_composition():
    prompt = build_editing_prompt(
        "ShotSpec: medium half-body portrait, three-quarter angle.",
        num_user_photos=4,
    )

    assert "identity reference images all show the user" in prompt
    assert "written ShotSpec for framing, camera angle, and pose" in prompt
    assert "Do not copy its person's identity, framing, or pose" in prompt
    assert "Image 5" not in prompt


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
    w._agent_router = PolicyEngine(
        agent_router=AgentRouter(identity_threshold_profile),
        learning_layer=None,
    )
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
        def __init__(self, path, **kwargs):
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


def test_face_swapper_reuses_evaluator_analysis_app(tmp_path, monkeypatch):
    model_path = tmp_path / "inswapper_128.onnx"
    model_path.write_bytes(b"model")
    shared_app = object()
    received_apps = []

    class FakeFaceSwapper:
        def __init__(
            self, _path, *, analysis_app=None, providers=None,
            analysis_release=None,
        ):
            received_apps.append((analysis_app, providers, analysis_release))

    import server.repair.identity_repair as repair_module

    monkeypatch.setattr(repair_module.settings, "face_swap_enabled", True)
    monkeypatch.setattr(repair_module.settings, "face_swap_model_path", model_path)
    monkeypatch.setattr(repair_module, "FaceSwapper", FakeFaceSwapper)

    release = lambda: None
    repair = FaceSwapRepair(
        analysis_app_factory=lambda: shared_app,
        analysis_app_release=release,
    )

    assert isinstance(repair._get_swapper(), FakeFaceSwapper)
    assert received_apps == [(shared_app, ["CPUExecutionProvider"], release)]


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


def test_calibrated_cosine_threshold_is_a_real_hard_failure(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    class FakeLearningLayer:
        def get_calibration(self):
            return SimpleNamespace(
                sample_count=20,
                identity_pass_threshold=8.0,
                identity_repair_threshold=7.0,
                identity_cosine_accept=0.47,
            )

    class FakeIdentityApp:
        def get(self, img):
            if "generated" in img:
                embedding = np.array([0.46, np.sqrt(1.0 - 0.46**2), 0.0])
            else:
                embedding = np.array([1.0, 0.0, 0.0])
            return [SimpleNamespace(bbox=[0, 0, 10, 10], normed_embedding=embedding)]

    monkeypatch.setattr(cv2, "imread", lambda path: str(path))
    eval_svc = EvaluationService(learning_layer=FakeLearningLayer())
    eval_svc._get_identity_app = lambda: FakeIdentityApp()  # type: ignore[assignment]

    result = eval_svc._local_identity_similarity_check(
        "/tmp/generated.png",
        ["/tmp/reference.png"],
    )

    assert result["score"] == 7
    assert result["measurements"]["identity_accept_cosine"] == 0.47
    assert "identity_too_low" in result["hard_failures"]
    gate = EvaluationService._candidate_gate_status({
        "scores": {
            "identity": result["score"],
            "face_quality": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": result["hard_failures"],
    })
    assert gate["identity_pass"] is False


def test_hero_cosine_threshold_overrides_looser_learning_threshold(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    class FakeIdentityApp:
        def get(self, img):
            if "generated" in img:
                embedding = np.array([0.74, np.sqrt(1.0 - 0.74**2), 0.0])
            else:
                embedding = np.array([1.0, 0.0, 0.0])
            return [SimpleNamespace(bbox=[0, 0, 10, 10], normed_embedding=embedding)]

    monkeypatch.setattr(cv2, "imread", lambda path: str(path))
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: FakeIdentityApp()  # type: ignore[assignment]

    result = eval_svc._local_identity_similarity_check(
        "/tmp/generated.png",
        ["/tmp/reference.png"],
        cosine_accept_threshold=0.78,
    )

    assert result["cosine_similarity"] == 0.74
    assert result["score"] == 7
    assert result["measurements"]["identity_accept_cosine"] == 0.78
    assert "identity_too_low" in result["hard_failures"]


def test_hero_threshold_calibrates_to_multi_view_reference_consistency(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    reference_cosine = 0.77
    target_cosine = 0.775
    reference_1 = np.array([1.0, 0.0, 0.0])
    reference_2 = np.array([
        reference_cosine,
        np.sqrt(1.0 - reference_cosine**2),
        0.0,
    ])
    reference_mean = reference_1 + reference_2
    reference_mean = reference_mean / np.linalg.norm(reference_mean)
    perpendicular = np.array([-reference_mean[1], reference_mean[0], 0.0])
    generated = (
        target_cosine * reference_mean
        + np.sqrt(1.0 - target_cosine**2) * perpendicular
    )

    class FakeIdentityApp:
        def get(self, img):
            if "generated" in img:
                embedding = generated
            elif "ref_1" in img:
                embedding = reference_1
            else:
                embedding = reference_2
            return [SimpleNamespace(bbox=[0, 0, 10, 10], normed_embedding=embedding)]

    monkeypatch.setattr(cv2, "imread", lambda path: str(path))
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: FakeIdentityApp()  # type: ignore[assignment]

    result = eval_svc._local_identity_similarity_check(
        "/tmp/generated.png",
        ["/tmp/ref_1.png", "/tmp/ref_2.png"],
        cosine_accept_threshold=0.78,
        reference_calibration_floor=0.76,
    )

    assert result["cosine_similarity"] == target_cosine
    assert result["reference_consistency"] == reference_cosine
    assert result["measurements"]["identity_accept_cosine"] == reference_cosine
    assert result["measurements"]["identity_accept_cosine_base"] == 0.78
    assert result["measurements"]["identity_threshold_reference_calibrated"] is True
    assert "identity_too_low" not in result["hard_failures"]


def test_hero_reference_calibration_never_drops_below_floor(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    reference_cosine = 0.40
    target_cosine = 0.75
    reference_1 = np.array([1.0, 0.0, 0.0])
    reference_2 = np.array([
        reference_cosine,
        np.sqrt(1.0 - reference_cosine**2),
        0.0,
    ])
    reference_mean = reference_1 + reference_2
    reference_mean = reference_mean / np.linalg.norm(reference_mean)
    perpendicular = np.array([-reference_mean[1], reference_mean[0], 0.0])
    generated = (
        target_cosine * reference_mean
        + np.sqrt(1.0 - target_cosine**2) * perpendicular
    )

    class FakeIdentityApp:
        def get(self, img):
            if "generated" in img:
                embedding = generated
            elif "ref_1" in img:
                embedding = reference_1
            else:
                embedding = reference_2
            return [SimpleNamespace(bbox=[0, 0, 10, 10], normed_embedding=embedding)]

    monkeypatch.setattr(cv2, "imread", lambda path: str(path))
    eval_svc = EvaluationService()
    eval_svc._get_identity_app = lambda: FakeIdentityApp()  # type: ignore[assignment]

    result = eval_svc._local_identity_similarity_check(
        "/tmp/generated.png",
        ["/tmp/ref_1.png", "/tmp/ref_2.png"],
        cosine_accept_threshold=0.78,
        reference_calibration_floor=0.76,
    )

    assert result["reference_consistency"] == reference_cosine
    assert result["measurements"]["identity_accept_cosine"] == 0.76
    assert result["measurements"]["identity_reference_calibration_floor"] == 0.76
    assert "identity_too_low" in result["hard_failures"]


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


def test_local_image_quality_uses_face_crop_for_soft_background(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    image = np.full((1024, 768, 3), 180, dtype=np.uint8)
    x, y, width, height = 250, 260, 240, 300
    rng = np.random.default_rng(11)
    textured = image[y:y + height, x:x + width].astype(np.int16)
    textured += rng.integers(-28, 29, textured.shape, dtype=np.int16)
    image[y:y + height, x:x + width] = np.clip(textured, 0, 255).astype(np.uint8)
    cv2.ellipse(
        image,
        (x + width // 2, y + height // 2),
        (width // 3, height // 2 - 10),
        0, 0, 360, (170, 170, 170), 2,
    )
    cv2.circle(image, (x + 80, y + 115), 7, (40, 40, 40), -1)
    cv2.circle(image, (x + 160, y + 115), 7, (40, 40, 40), -1)
    cv2.line(image, (x + 95, y + 205), (x + 145, y + 205), (60, 60, 60), 2)
    for offset in range(0, 210, 10):
        cv2.line(
            image,
            (x + 15 + offset, y + 35),
            (x + 22 + offset, y + 48),
            (95, 95, 95),
            1,
        )
    path = tmp_path / "sharp-face-soft-background.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[x, y, width, height]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    out = EvaluationService._local_image_quality_check(str(path))

    assert (
        out["measurements"]["blur_variance"]
        < out["measurements"]["face_blur_variance"]
    )
    assert out["measurements"]["face_blur_variance"] >= 60
    assert out["measurements"]["sharpness_metric_source"] == "face_crop_256"
    assert "too_blurry" not in out["hard_failures"]
    assert out["scores"]["face_quality"] >= 8


def test_local_image_quality_still_rejects_blurry_detected_face(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    image = np.full((1024, 768, 3), 180, dtype=np.uint8)
    x, y, width, height = 250, 260, 240, 300
    path = tmp_path / "blurry-face.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[x, y, width, height]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    out = EvaluationService._local_image_quality_check(str(path))

    assert out["measurements"]["sharpness_metric_source"] == "face_crop_256"
    assert out["measurements"]["face_blur_variance"] < 20
    assert "too_blurry" in out["hard_failures"]
    assert out["scores"]["face_quality"] <= 2


def test_local_image_quality_filters_small_haar_duplicate(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, (1024, 768, 3), dtype=np.uint8)
    path = tmp_path / "one-face-with-ear-false-positive.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([
                [250, 220, 180, 180],
                [205, 275, 145, 145],
                [490, 290, 90, 90],
            ])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    out = EvaluationService._local_image_quality_check(str(path))

    assert out["measurements"]["raw_face_count"] == 3
    assert out["measurements"]["face_count"] == 1
    assert "multiple_faces" not in out["hard_failures"]


def test_local_image_quality_keeps_separate_substantial_second_face(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    rng = np.random.default_rng(23)
    image = rng.integers(0, 256, (1024, 768, 3), dtype=np.uint8)
    path = tmp_path / "two-people.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([
                [90, 220, 180, 180],
                [500, 260, 150, 150],
            ])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    out = EvaluationService._local_image_quality_check(str(path))

    assert out["measurements"]["raw_face_count"] == 2
    assert out["measurements"]["face_count"] == 2
    assert "multiple_faces" in out["hard_failures"]


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


def test_hero_selection_prefers_stronger_cosine_after_both_pass_gates():
    prettier = {
        "aggregate_score": 9.7,
        "selection_profile": "hero_identity",
        "judgement": {"identity_quality": {"cosine_similarity": 0.79}},
        "gate_status": {"hard_gates_pass": True},
    }
    more_recognizable = {
        "aggregate_score": 9.1,
        "selection_profile": "hero_identity",
        "judgement": {"identity_quality": {"cosine_similarity": 0.87}},
        "gate_status": {"hard_gates_pass": True},
    }

    assert AgentRouter().select_candidate([prettier, more_recognizable]) is more_recognizable


def test_candidate_action_uses_bounded_identity_state_machine():
    assert GeminiWorker._decide_candidate_action({
        "scores": {"identity": 9, "face_quality": 9, "realism": 9, "artifact": 9, "commercial_readiness": 9},
        "hard_failures": [],
        "recommended_action": "accept",
    })["action"] == "ACCEPT"
    assert GeminiWorker._decide_candidate_action({
        "scores": {"identity": 7, "style_match": 9},
        "hard_failures": [],
        "recommended_action": "retry",
    })["action"] == "IDENTITY_REPAIR"
    assert GeminiWorker._decide_candidate_action({
        "scores": {
            "identity": 5,
            "face_quality": 9,
            "style_match": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": ["identity_too_low"],
        "recommended_action": "face_swap",
    })["action"] == "IDENTITY_REPAIR"


def test_candidate_gate_treats_safety_as_hard_delivery_gate():
    gate = EvaluationService._candidate_gate_status({
        "scores": {
            "identity": 10,
            "face_quality": 10,
            "realism": 10,
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
            "realism": 9,
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
        "scores": {"identity": 7, "style_match": 9, "realism": 9},
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


def test_identity_repair_runs_for_gray_zone_or_good_low_identity_frame():
    gray_zone_identity = {
        "scores": {"identity": 7},
        "hard_failures": [],
        "recommended_action": "retry",
    }
    below_repair_threshold = {
        "scores": {
            "identity": 6,
            "face_quality": 9,
            "style_match": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": ["identity_too_low"],
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
    assert GeminiWorker._should_apply_identity_repair(below_repair_threshold) is True
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
        self.generation_prompts = []
        self.template_paths = []
        self.judge_prompts = []

    def create_from_references(self, prompt, reference_paths, template_path, title, editing_mode=True):
        self.start_calls += 1
        self.photo_counts.append(len(reference_paths))
        self.generation_prompts.append(prompt)
        self.template_paths.append(template_path)
        self._last_image_path = f"/tmp/{title}.png"
        return self._last_image_path

    def local_edit(self, current_image_path, reference_paths, edit_prompt, title):
        self.converse_calls += 1
        self._last_image_path = f"/tmp/{title}.png"
        return self._last_image_path

    def judge(self, current_image_path, reference_paths, judge_prompt, timeout=None):
        self.judge_calls += 1
        self.judge_prompts.append(judge_prompt)
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
    w._agent_router = PolicyEngine(
        agent_router=AgentRouter(identity_threshold_profile),
        learning_layer=None,
    )
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
        '"realism":8,"artifact":8,"commercial_readiness":8},'
        '"hard_failures":[],"recommended_action":"%s","notes":"ok"}'
    ) % (identity, action)


def _judge_json_scores(
    *,
    identity: int = 9,
    face_quality: int = 8,
    style_match: int = 8,
    realism: int = 8,
    artifact: int = 8,
    commercial_readiness: int = 8,
    action: str = "accept",
) -> str:
    return (
        '{"scores":{"identity":%d,"face_quality":%d,"style_match":%d,'
        '"realism":%d,"artifact":%d,"commercial_readiness":%d},'
        '"hard_failures":[],"recommended_action":"%s","notes":"local issue"}'
    ) % (
        identity,
        face_quality,
        style_match,
        realism,
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
    assert meta["pipeline"] == "controlled_candidate_v3"
    assert meta["budget"]["initial_candidates"] == 3
    assert meta["budget"]["max_total_api_cost"] == 0.4
    assert meta["budget"]["estimated_cost_used"] == 0.12
    assert meta["identity_pack"]["persistent_face_library"] is False
    assert [
        ref["role"]
        for ref in meta["identity_pack"]["reference_images"]
    ] == ["front_neutral"]
    assert meta["provider_invocations"][0]["operation"] == "CREATE_FROM_REFERENCES"
    assert meta["provider_invocations"][0]["provider"] == "openrouter"
    assert meta["strategy"]["generation_task_type"] == "half_body"
    assert meta["strategy"]["generation_routing"]["provider"] == "openrouter"
    assert meta["provider_invocations"][0]["routing_decision"] == (
        meta["strategy"]["generation_routing"]
    )
    assert meta["provider_invocations"][0]["reference_ids"] == ["ref_1"]
    assert meta["provider_invocations"][0]["cost"] == 0.04
    assert meta["provider_invocations"][0]["estimated_cost"] == 0.04
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


def test_openrouter_full_set_uses_photoreal_quality_provider():
    w = _make_pipeline_worker([_judge_json(9), _judge_json(9), _judge_json(9)])
    regular_provider = w._gateway._openrouter
    quality_provider = FakePipelineClient([])
    quality_provider.model = "black-forest-labs/flux.2-pro"
    quality_provider.image_provider = "black-forest-labs"
    quality_provider.estimated_image_cost = 0.12
    w._gateway._openrouter_hero = quality_provider

    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_quality_route", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert quality_provider.start_calls == 3
    assert regular_provider.start_calls == 0
    assert regular_provider.judge_calls == 3
    assert meta["strategy"]["generation_routing"] == {
        "provider": "openrouter",
        "model": "black-forest-labs/flux.2-pro",
        "provider_tag": "black-forest-labs",
        "reason": "photoreal_identity_quality_route",
        "estimated_cost": 0.12,
        "estimated_latency_ms": 55_000,
        "confidence": 0.9,
    }
    assert meta["budget"]["estimated_cost_used"] == 0.36
    assert {
        invocation["model"] for invocation in meta["provider_invocations"]
    } == {"black-forest-labs/flux.2-pro"}


def test_quality_pipeline_budget_stops_initial_candidate_generation(monkeypatch):
    monkeypatch.setattr(gemini_worker_module, "MAX_PIPELINE_TOTAL_API_COST", 0.05)
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
    assert meta["budget"]["estimated_cost_used"] == 0.04
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


def test_siliconflow_invocation_reserves_template_slot_and_records_two_refs(
    monkeypatch,
):
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    w = _make_pipeline_worker([_judge_json(9), _judge_json(8), _judge_json(8)])

    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_siliconflow_refs",
        "prompt",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"],
        "title",
        template_path="style.png",
    )

    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").photo_counts == [2, 2, 2]
    assert len(meta["identity_pack"]["reference_images"]) == 6
    assert meta["provider_invocations"][0]["provider"] == "siliconflow"
    assert meta["provider_invocations"][0]["reference_ids"] == ["ref_1", "ref_2"]
    assert [
        ref["role"]
        for ref in meta["provider_invocations"][0]["reference_roles"]
    ] == ["front_neutral", "front_smile"]


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


def test_composition_first_never_applies_to_identity_critical_hero():
    half_body = {"shot_id": "half_body"}
    assert should_use_composition_first(half_body, backend="siliconflow") is True
    assert should_use_composition_first(half_body, backend="openrouter") is False
    assert should_use_composition_first(
        half_body, force_closeup=True, backend="siliconflow"
    ) is False
    assert should_use_composition_first(
        half_body, force_closeup=True, backend="openrouter"
    ) is False
    assert should_use_composition_first(
        half_body, force_closeup=True, backend="chrome"
    ) is False
    assert should_use_composition_first(
        {"shot_id": "profile"}, backend="siliconflow"
    ) is True


def test_composition_scaffold_prompt_makes_half_body_geometry_non_negotiable():
    prompt = build_composition_scaffold_prompt(
        "Polished Korean studio wardrobe and soft window light.",
        {
            "shot_id": "half_body",
            "framing": "medium half-body portrait",
            "pose": "slight three-quarter angle",
            "lighting": "soft window light",
            "lens": "50mm",
        },
        1,
        3,
    )

    assert "WIDE THREE-QUARTER FASHION PORTRAIT" in prompt
    assert "down to mid-thigh" in prompt
    assert "12% clear margin below the fingertips" in prompt
    assert "must not be a headshot or chest crop" in prompt
    assert "temporary face" not in prompt
    assert "Written ShotSpec" not in prompt
    assert len(prompt) < 2000


def test_environmental_scaffold_ends_with_executable_composition_lock():
    prompt = build_composition_scaffold_prompt(
        "Korean beauty close-up, shot on 85mm with sharp eyes.",
        {
            "shot_id": "environmental",
            "framing": "wider environmental portrait",
            "pose": "standing naturally",
            "lighting": "window light",
            "lens": "35mm environmental portrait lens",
        },
        1,
        2,
    )

    assert "complete head through both feet" in prompt
    assert "45% to 65%" in prompt
    assert "Never crop the head, hands, legs or feet" in prompt
    assert "full-length lower garment and both shoes" in prompt
    assert "shot on 85mm" not in prompt
    assert "shot on 85mm" not in prompt
    assert prompt.rstrip().endswith(
        "this must not be a close-up or chest portrait."
    )


def test_scaffold_style_direction_keeps_wardrobe_and_scene_not_face_bias():
    prompt = build_composition_scaffold_prompt(
        "unused composed prompt",
        {
            "style_label": "Japanese & Korean Portrait",
            "template_label": "Korean silk",
            "shot_id": "environmental",
            "framing": "wider environmental portrait",
            "pose": "standing naturally",
            "lighting": "window light",
            "lens": "35mm",
            "prompt_blocks": {
                "style_block": (
                    "Generate an elegant portrait of a woman. "
                    "She wears a cream silk blouse. "
                    "Makeup is dewy with soft pink lips. "
                    "The background is warm beige. "
                    "Lighting comes from a large window. "
                    "Shot on 85mm f/2, sharp focus on eyes."
                )
            },
        },
        1,
        2,
    )

    style_section = prompt.split("Style:", 1)[1].split("Pose:", 1)[0]
    assert "cream silk blouse" in style_section
    assert "background is warm beige" in style_section
    assert "large window" in style_section
    assert "Makeup" not in style_section
    assert "85mm" not in style_section


def test_profile_scaffold_requires_visible_head_turn():
    prompt = build_composition_scaffold_prompt(
        "Soft Korean beauty portrait.",
        {
            "shot_id": "profile",
            "framing": "shoulder-up three-quarter profile portrait",
            "pose": "looking gently away",
            "lighting": "soft rim light",
            "lens": "85mm portrait lens",
        },
        1,
        2,
    )

    assert "head turned 45 to 70 degrees" in prompt
    assert "nose must sit visibly off" in prompt
    assert prompt.rstrip().endswith(
        "This must not be a frontal or near-frontal face."
    )


def test_sharpness_only_failure_preserves_composition_for_local_edit():
    judgement = {
        "scores": {"identity": 10, "face_quality": 4, "style_match": 10},
        "hard_failures": ["too_blurry"],
        "local_quality": {
            "scores": {"face_quality": 4},
            "hard_failures": ["too_blurry"],
            "measurements": {
                "sharpness_metric_source": "face_crop",
                "sharpness_value": 18.0,
            },
        },
    }
    thresholds = {"identity_pass_threshold": 7.5}

    assert should_prefer_local_sharpness_edit(judgement, thresholds) is True
    judgement["hard_failures"].append("wrong_composition")
    assert should_prefer_local_sharpness_edit(judgement, thresholds) is False


def test_soft_post_identity_repair_routes_to_local_sharpness_edit():
    judgement = {
        "scores": {"identity": 10, "face_quality": 6, "style_match": 10},
        "hard_failures": [],
        "local_quality": {
            "scores": {"face_quality": 6},
            "hard_failures": [],
            "measurements": {
                "sharpness_metric_source": "face_crop",
                "sharpness_value": 38.7,
            },
        },
    }

    assert should_prefer_local_sharpness_edit(
        judgement, {"identity_pass_threshold": 8.0}
    ) is True


def test_sharpness_recovery_uses_the_active_delivery_threshold():
    judgement = {
        "scores": {"identity": 10, "face_quality": 8, "style_match": 9},
        "hard_failures": [],
        "local_quality": {
            "scores": {
                "face_quality": 8,
                "artifact": 8,
                "commercial_readiness": 8,
            },
            "hard_failures": [],
            "measurements": {
                "sharpness_metric_source": "face_crop_256",
                "sharpness_value": 77.0,
            },
        },
    }

    assert should_prefer_local_sharpness_edit(
        judgement,
        {"identity_pass_threshold": 8, "quality_accept_threshold": 9},
    ) is True
    assert should_prefer_local_sharpness_edit(
        judgement,
        {"identity_pass_threshold": 8, "quality_accept_threshold": 8},
    ) is False


def test_small_profile_face_routes_to_deterministic_reframe():
    judgement = {
        "scores": {"identity": 10, "face_quality": 7, "style_match": 10},
        "hard_failures": ["face_scale_unusual"],
        "local_quality": {
            "scores": {"face_quality": 7},
            "hard_failures": ["face_scale_unusual"],
            "measurements": {
                "face_area_ratio": 0.0132,
                "face_area_range": [0.02, 0.38],
                "face_count": 1,
            },
        },
    }

    assert should_prefer_local_reframe(
        judgement,
        {"identity_pass_threshold": 7.5},
        {"shot_id": "profile"},
    ) is True
    judgement["hard_failures"].append("wrong_composition")
    assert should_prefer_local_reframe(
        judgement,
        {"identity_pass_threshold": 7.5},
        {"shot_id": "profile"},
    ) is False


def test_profile_pipeline_reframes_then_sharpens_before_delivery(monkeypatch):
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    swapped = SimpleNamespace(
        output_path=Path("/tmp/composition_identity.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    w = _make_pipeline_worker([], swap_result=swapped)
    judgements = iter([
        # composition_scaffold stage judgement (recorded, not routed on)
        {
            "scores": {
                "identity": 3,
                "face_quality": 9,
                "style_match": 10,
                "realism": 7,
                "artifact": 9,
                "commercial_readiness": 9,
            },
            "hard_failures": [],
            "recommended_action": "accept",
        },
        # composition_face_swap stage judgement (recorded, not routed on)
        {
            "scores": {
                "identity": 10,
                "face_quality": 9,
                "style_match": 10,
                "realism": 7,
                "artifact": 9,
                "commercial_readiness": 9,
            },
            "hard_failures": [],
            "recommended_action": "accept",
        },
        # composition_identity_blend stage judgement → reframe local edit
        {
            "scores": {
                "identity": 10,
                "face_quality": 7,
                "style_match": 10,
                "realism": 7,
                "artifact": 9,
                "commercial_readiness": 9,
            },
            "hard_failures": ["face_scale_unusual"],
            "recommended_action": "retry",
            "local_quality": {
                "scores": {"face_quality": 7},
                "hard_failures": ["face_scale_unusual"],
                "measurements": {
                    "face_area_ratio": 0.0132,
                    "face_area_range": [0.02, 0.38],
                    "face_count": 1,
                },
            },
        },
        # post-reframe judgement → follow-up sharpen
        {
            "scores": {
                "identity": 10,
                "face_quality": 4,
                "style_match": 10,
                "realism": 7,
                "artifact": 4,
                "commercial_readiness": 4,
            },
            "hard_failures": ["too_blurry"],
            "recommended_action": "retry",
            "local_quality": {
                "scores": {"face_quality": 4},
                "hard_failures": ["too_blurry"],
                "measurements": {
                    "sharpness_metric_source": "face_crop",
                    "sharpness_value": 23.9,
                },
            },
        },
        # post-sharpen judgement → the only hard-gate-passing variant
        {
            "scores": {
                "identity": 10,
                "face_quality": 9,
                "style_match": 10,
                "realism": 9,
                "artifact": 9,
                "commercial_readiness": 9,
            },
            "hard_failures": [],
            "recommended_action": "accept",
            "local_quality": {
                "scores": {"face_quality": 10},
                "hard_failures": [],
                "measurements": {
                    "sharpness_metric_source": "face_crop",
                    "sharpness_value": 102.5,
                },
            },
        },
    ])
    w._eval_service.judge_current_candidate = (  # type: ignore[assignment]
        lambda *_args, **_kwargs: next(judgements)
    )
    monkeypatch.setattr(
        gemini_worker_module,
        "reframe_small_face_region",
        lambda _source, output: Path(output),
    )
    monkeypatch.setattr(
        gemini_worker_module,
        "sharpen_face_region",
        lambda _source, output: Path(output),
    )

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s_profile_reframe",
        "Korean studio portrait.",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "title",
        template_path=None,
        shot_spec_metadata={
            "shot_id": "profile",
            "framing": "shoulder-up three-quarter profile portrait",
            "pose": "head turned 60 degrees away",
            "lighting": "soft rim light",
            "lens": "85mm portrait lens",
        },
    )

    assert fp.endswith("_reframed_face_sharp.png")
    assert meta["budget"]["local_edits_used"] == 2
    assert meta["selected_candidate"]["deliverable"] is True
    assert [
        invocation["model"]
        for invocation in meta["provider_invocations"]
        if invocation["operation"] == "LOCAL_EDIT"
    ] == ["opencv_face_reframe_v1", "opencv_face_unsharp_v1"]
    assert meta["local_edit"]["followup_sharpness"]["applied"] is True
    assert meta["variant_selection"] == {
        "selected_stage": "post_reframe_sharpness",
        "reason": "last_stage_most_real",
        "total_variants": 5,
    }


def test_siliconflow_half_body_pipeline_builds_scaffold_then_writes_identity(
    monkeypatch,
):
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    swapped = SimpleNamespace(
        output_path=Path("/tmp/composition_identity.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    w = _make_pipeline_worker(
        [_judge_json(8), _judge_json(8), _judge_json(8)],
        swap_result=swapped,
    )

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s_composition",
        "Korean studio portrait.",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "title",
        template_path=None,
        shot_spec_metadata={
            "shot_id": "half_body",
            "framing": "medium half-body portrait",
            "pose": "standing at a three-quarter angle",
            "lighting": "soft window light",
            "lens": "50mm portrait lens",
        },
    )

    provider = w._gateway._provider_for("COMPOSITION_SCAFFOLD")
    assert fp == "/tmp/title_cand1_identity_blend.png"
    assert provider.photo_counts == [0]
    assert w.swap_calls == 1
    assert meta["strategy"]["generation_mode"] == "composition_first"
    assert meta["strategy"]["generation_operation"] == "COMPOSITION_SCAFFOLD"
    assert meta["budget"]["initial_candidates"] == 1
    assert meta["budget"]["max_regenerations"] == 1
    assert meta["budget"]["composition_identity_writes_used"] == 1
    assert meta["budget"]["identity_blends_used"] == 1
    assert meta["provider_invocations"][0]["operation"] == "COMPOSITION_SCAFFOLD"
    assert meta["provider_invocations"][0]["reference_ids"] == []
    assert meta["provider_invocations"][0]["model"] == "Qwen/Qwen-Image"
    assert meta["provider_invocations"][1]["operation"] == "IDENTITY_REPAIR"
    assert meta["provider_invocations"][1]["reference_ids"] == [
        "ref_1", "ref_2", "ref_3", "ref_4",
    ]
    assert meta["provider_invocations"][2]["operation"] == "IDENTITY_BLEND"
    assert meta["provider_invocations"][2]["reference_ids"] == ["ref_1", "ref_3"]
    assert meta["face_swap"]["action"] == "composition_identity_write"
    assert meta["face_swap"]["identity_blend"]["applied"] is True
    assert "Composition target (hard requirement)" in provider.judge_prompts[0]
    assert '"framing": "medium half-body portrait"' in provider.judge_prompts[0]
    assert [
        variant["stage"]
        for variant in meta["selected_candidate"]["variants"]
    ] == [
        "composition_scaffold",
        "composition_face_swap",
        "composition_identity_blend",
    ]


def test_wrong_composition_is_a_delivery_hard_failure():
    judgement = {
        "scores": {
            "identity": 9,
            "face_quality": 9,
            "style_match": 5,
            "realism": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": ["wrong_composition"],
        "recommended_action": "retry",
    }
    gate = EvaluationService._candidate_gate_status(
        judgement,
        identity_threshold_profile({"shot_id": "half_body"}),
    )
    assert gate["hard_gates_pass"] is False
    assert gate["severe_quality_fail"] is True
    assert "severe_quality_failure" in gate["hard_gate_failures"]


def test_half_body_local_geometry_allows_face_in_upper_third(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    image = np.full((1160, 896, 3), 160, dtype=np.uint8)
    path = tmp_path / "half-body.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[350, 110, 190, 230]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    result = EvaluationService._local_image_quality_check(
        str(path),
        {"shot_id": "half_body", "framing": "medium half-body portrait"},
    )

    assert result["measurements"]["geometry_profile"] == "medium"
    assert "face_off_center" not in result["hard_failures"]
    assert "face_scale_unusual" not in result["hard_failures"]


def test_profile_local_geometry_allows_face_in_upper_third(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    image = np.full((1160, 896, 3), 160, dtype=np.uint8)
    path = tmp_path / "profile.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[350, 100, 190, 220]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    result = EvaluationService._local_image_quality_check(
        str(path),
        {"shot_id": "profile", "framing": "turned profile portrait"},
    )

    assert result["measurements"]["geometry_profile"] == "profile_editorial"
    assert "face_off_center" not in result["hard_failures"]
    assert "face_scale_unusual" not in result["hard_failures"]


def test_hero_local_geometry_rejects_arm_length_selfie_crop(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    image = np.random.default_rng(23).integers(
        0, 255, size=(1160, 896, 3), dtype=np.uint8
    )
    path = tmp_path / "selfie-like-hero.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[190, 80, 520, 650]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    result = EvaluationService._local_image_quality_check(
        str(path),
        {
            "shot_id": "closeup",
            "framing": "natural chest-up editorial portrait with breathing room",
            "hero_preview": True,
        },
    )

    assert result["measurements"]["geometry_profile"] == "hero_editorial"
    assert result["measurements"]["face_area_ratio"] > 0.24
    assert "anti_selfie_composition" in result["hard_failures"]
    gate = EvaluationService._candidate_gate_status({
        "scores": {
            "identity": 9,
            "face_quality": 9,
            "style_match": 9,
            "realism": 9,
            "artifact": 9,
            "commercial_readiness": 9,
        },
        "hard_failures": ["anti_selfie_composition"],
    })
    assert gate["hard_gates_pass"] is False


def test_hero_local_geometry_accepts_editorial_chest_up_scale(
    tmp_path, monkeypatch,
):
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    import cv2

    image = np.random.default_rng(24).integers(
        0, 255, size=(1160, 896, 3), dtype=np.uint8
    )
    path = tmp_path / "editorial-hero.png"
    cv2.imwrite(str(path), image)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[285, 120, 280, 340]])

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: FaceCascade())
    result = EvaluationService._local_image_quality_check(
        str(path),
        {
            "shot_id": "closeup",
            "framing": "natural chest-up editorial portrait with breathing room",
            "hero_preview": True,
        },
    )

    assert result["measurements"]["geometry_profile"] == "hero_editorial"
    assert 0.045 <= result["measurements"]["face_area_ratio"] <= 0.24
    assert "anti_selfie_composition" not in result["hard_failures"]
    assert "face_scale_unusual" not in result["hard_failures"]


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
    assert meta["budget"]["estimated_cost_used"] == 0.16
    assert meta["selected_candidate"]["candidate_id"] == "cand_4"
    assert meta["selected_candidate"]["deliverable"] is True
    assert any(
        inv["invocation_id"] == "regenerate_1"
        and inv["operation"] == "CREATE_FROM_REFERENCES"
        and inv["parent_candidate_id"] == "cand_1"
        and inv["cost"] == 0.04
        for inv in meta["provider_invocations"]
    )
    assert any(
        action["action"] == "REGENERATE_FROM_ORIGINAL"
        and action["executed"] is True
        for action in meta["agent_actions"]
    )
    assert meta["history"][-1]["regenerated_from_candidate_id"] == "cand_1"


def test_reselected_candidate_cannot_repeat_an_executed_recovery_strategy():
    w = _make_pipeline_worker([])
    selected = {
        "candidate_id": "hero_cand_1",
        "index": 1,
        "agent_action": {
            "action": "REGENERATE_FROM_ORIGINAL",
            "failure_class": "synthetic_texture",
            "route_mode": "primary",
            "recovery_strategy": "photoreal_regeneration",
        },
    }
    actions = [{
        **selected["agent_action"],
        "candidate_id": "hero_cand_1",
        "executed": True,
    }]

    planned = w._replan_recovery_before_execution(
        selected,
        actions,
        state="REPLAN_BEFORE_REGENERATION",
    )

    assert planned["action"] == "DROP_CANDIDATE"
    assert planned["recovery_strategy"] == "texture_routes_exhausted"
    assert selected["agent_action"] == planned
    assert actions[-1]["state"] == "REPLAN_BEFORE_REGENERATION"
    assert actions[-1]["executed"] is True
    assert actions[-1]["selected_for_execution"] is True


def test_pipeline_persists_executed_recovery_outcome_for_future_policy():
    class StrategyRecorder:
        def __init__(self):
            self.records = []

        def record_pipeline_outcome(self, **payload):
            self.records.append(payload)

    w = _make_pipeline_worker([
        _judge_json(5, "retry"),
        _judge_json(5, "retry"),
        _judge_json(5, "retry"),
        _judge_json(9, "accept"),
    ])
    recorder = StrategyRecorder()
    w._learning_layer = recorder

    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_strategy_memory", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert meta["learning"]["strategy_outcomes_recorded"] == 1
    assert len(recorder.records) == 1
    assert recorder.records[0]["action"] == "REGENERATE_FROM_ORIGINAL"
    assert recorder.records[0]["failure_class"] == "identity_similarity"
    assert recorder.records[0]["passed"] is True


def test_quality_pipeline_repairs_good_candidate_below_repair_threshold():
    swapped = SimpleNamespace(
        output_path=Path("/tmp/low_identity_swapped.png"),
        swapped=True,
        message="swapped",
        source_face_count=1,
        target_face_count=1,
    )
    w = _make_pipeline_worker([
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(9, "accept"),
    ], swap_result=swapped)

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s_low_identity_repair", "prompt", ["a.jpg"], "title", template_path=None
    )

    assert fp == "/tmp/low_identity_swapped.png"
    assert w._gateway._provider_for("CREATE_FROM_REFERENCES").start_calls == 3
    assert w.swap_calls == 1
    assert meta["budget"]["regenerations_used"] == 0
    assert meta["budget"]["identity_repairs_used"] == 1
    assert meta["face_swap"]["action"] == "face_swap"
    assert meta["selected_candidate"]["deliverable"] is True
    assert any(
        action["action"] == "IDENTITY_REPAIR"
        and action["base_action"] == "IDENTITY_REPAIR"
        and action["selected_for_execution"] is True
        for action in meta["agent_actions"]
    )


def test_quality_pipeline_reports_failed_gate_instead_of_accepting_progress():
    w = _make_pipeline_worker([
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
        _judge_json(6, "face_swap"),
    ])
    progress = []

    _fp, meta = w.execute_generate_with_quality_pipeline(
        "s_progress_fail",
        "prompt",
        ["a.jpg"],
        "title",
        template_path=None,
        progress_callback=lambda *args: progress.append(args),
    )

    assert meta["selected_candidate"]["deliverable"] is False
    assert meta["history"][-1]["accepted"] is False
    assert progress[-1][1] == 3
    assert progress[-1][2] == "failed_gate"
    assert "1/3" in progress[-1][3]


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
        and inv["cost"] == 0.04
        and inv["estimated_cost"] == 0.04
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
    # No variant passes the hard gates → keep the last-stage output and let
    # the final delivery gate intercept it (no fabricated pass).
    assert meta["variant_selection"] == {
        "selected_stage": "local_edit",
        "reason": "no_hard_gate_passing_variant",
        "total_variants": 2,
    }
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


def test_hero_preview_pipeline_uses_hero_profile_through_shared_skeleton():
    """Hero preview must flow through the shared skeleton with hero-only labelling.

    Locks the Task 3 refactor: execute_hero_preview is now a thin wrapper over
    _execute_candidate_pipeline. The hero profile (forced closeup, hero_cand_
    prefix, strict Hero thresholds, and direct identity-first route) must reach
    the metadata without invoking an unnecessary local identity repair.
    """
    swapped = SimpleNamespace(
        output_path=Path("/tmp/hero_identity.png"),
        swapped=True,
        message="swapped",
        source_face_count=1,
        target_face_count=1,
    )
    judgement = (
        '{"scores":{"identity":9,"face_quality":9,"style_match":9,'
        '"realism":9,"artifact":9,"commercial_readiness":9},'
        '"hard_failures":[],"recommended_action":"accept","notes":"ok"}'
    )
    w = _make_pipeline_worker([judgement], swap_result=swapped)
    fp, meta = w.execute_hero_preview(
        "s_hero", "prompt", ["a.jpg"], "title", template_path="style-person.png"
    )

    provider = w._gateway._provider_for("CREATE_FROM_REFERENCES")
    assert fp == "/tmp/title_hero_cand1.png"
    assert provider.start_calls == 1
    assert provider.converse_calls == 0
    assert provider.judge_calls == 1
    assert meta["pipeline"] == "hero_preview_v3_identity_first_flux"
    assert meta["shot_spec"]["shot_id"] == "closeup"  # force_closeup
    assert meta["shot_spec"]["hero_preview"] is True
    assert meta["budget"]["initial_candidates"] == 1
    assert meta["budget"]["max_regenerations"] == 2
    assert meta["budget"]["max_identity_repairs"] == 1
    assert meta["budget"]["max_total_api_cost"] == 0.6
    assert [c["candidate_id"] for c in meta["candidates"]] == ["hero_cand_1"]
    assert meta["provider_invocations"][0]["invocation_id"] == "hero_create_1"
    assert meta["provider_invocations"][0]["operation"] == "CREATE_FROM_REFERENCES"
    assert meta["strategy"]["generation_mode"] == "identity_first"
    assert meta["strategy"]["identity_threshold_profile"]["realism_accept_threshold"] == 9
    assert meta["strategy"]["template_reference_used"] is False
    assert provider.template_paths == [None]
    assert "Candidate strategy: conservative identity-first result" in provider.generation_prompts[0]
    assert "Do not over-beautify" in provider.generation_prompts[0]
    assert meta["shortlist"][0]["candidate_id"] == "hero_cand_1"
    assert meta["face_swap"]["action"] == "none"
    assert w.swap_calls == 0
    assert [
        variant["stage"]
        for variant in meta["selected_candidate"]["variants"]
    ] == ["generated"]
    assert meta["variant_selection"] == {
        "selected_stage": "generated",
        "reason": "last_stage_most_real",
        "total_variants": 1,
    }


def test_hero_executes_identity_repair_when_qa_requests_it():
    swapped = SimpleNamespace(
        output_path=Path("/tmp/hero_identity.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    w = _make_pipeline_worker(
        [
            _judge_json_scores(
                identity=7, face_quality=9, style_match=9, realism=9,
                artifact=9, commercial_readiness=9, action="face_swap",
            ),
            _judge_json_scores(
                identity=9, face_quality=9, style_match=9, realism=9,
                artifact=9, commercial_readiness=9, action="accept",
            ),
        ],
        swap_result=swapped,
    )

    fp, meta = w.execute_hero_preview(
        "s_hero_repair", "prompt", ["a.jpg", "b.jpg"], "title"
    )

    assert fp == "/tmp/hero_identity.png"
    assert w.swap_calls == 1
    assert meta["budget"]["regenerations_used"] == 0
    assert meta["budget"]["identity_repairs_used"] == 1
    assert meta["selected_candidate"]["deliverable"] is True
    assert [
        variant["stage"]
        for variant in meta["selected_candidate"]["variants"]
    ] == ["generated", "identity_repair"]


def test_hero_changes_strategy_when_identity_repair_hurts_realism():
    swapped = SimpleNamespace(
        output_path=Path("/tmp/hero_identity_synthetic.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    repaired_but_synthetic = (
        '{"scores":{"identity":9,"face_quality":8,"style_match":9,'
        '"realism":7,"artifact":8,"commercial_readiness":7},'
        '"hard_failures":["skin_over_smoothed"],'
        '"recommended_action":"retry","notes":"repair looks synthetic"}'
    )
    w = _make_pipeline_worker(
        [
            _judge_json_scores(
                identity=7, face_quality=9, style_match=9, realism=9,
                artifact=9, commercial_readiness=9, action="face_swap",
            ),
            repaired_but_synthetic,
            _judge_json_scores(
                identity=9, face_quality=9, style_match=9, realism=9,
                artifact=9, commercial_readiness=9, action="accept",
            ),
        ],
        swap_result=swapped,
    )

    fp, meta = w.execute_hero_preview(
        "s_hero_repair_fallback",
        "prompt",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "title",
    )

    assert fp == "/tmp/title_hero_regen1.png"
    assert w.swap_calls == 1
    assert meta["budget"]["identity_repairs_used"] == 1
    assert meta["budget"]["regenerations_used"] == 1
    assert meta["selected_candidate"]["deliverable"] is True
    assert meta["selected_candidate"]["variants"][0]["stage"] == (
        "post_repair_regenerated"
    )
    assert any(
        action["state"] == "POST_REPAIR_EVALUATE"
        and action["action"] == "REGENERATE_FROM_ORIGINAL"
        and action["executed"] is True
        for action in meta["agent_actions"]
    )


def test_hero_preview_uses_second_rescue_sample_only_after_two_failures():
    w = _make_pipeline_worker([
        _judge_json_scores(
            identity=6, face_quality=9, style_match=9, realism=9,
            artifact=9, commercial_readiness=9, action="retry",
        ),
        _judge_json_scores(
            identity=6, face_quality=9, style_match=9, realism=9,
            artifact=9, commercial_readiness=9, action="retry",
        ),
        _judge_json_scores(
            identity=9, face_quality=9, style_match=9, realism=9,
            artifact=9, commercial_readiness=9, action="accept",
        ),
    ])

    fp, meta = w.execute_hero_preview(
        "s_hero_rescue", "prompt", ["a.jpg"], "title", template_path=None
    )

    provider = w._gateway._provider_for("CREATE_FROM_REFERENCES")
    assert fp == "/tmp/title_hero_regen2.png"
    assert provider.start_calls == 3
    assert provider.judge_calls == 3
    assert meta["budget"]["regenerations_used"] == 2
    assert meta["selected_candidate"]["candidate_id"] == "hero_cand_3"
    assert meta["selected_candidate"]["deliverable"] is True
    assert [
        item["invocation_id"] for item in meta["provider_invocations"]
    ] == ["hero_create_1", "hero_regenerate_1", "hero_regenerate_2"]


def test_repeated_failure_escalates_to_configured_alternate_generation_route():
    swapped = SimpleNamespace(
        output_path=Path("/tmp/hero_identity_synthetic.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    synthetic_8 = (
        '{"scores":{"identity":8,"face_quality":9,"style_match":9,'
        '"realism":7,"artifact":9,"commercial_readiness":7},'
        '"hard_failures":["skin_over_smoothed"],'
        '"recommended_action":"retry","notes":"synthetic"}'
    )
    synthetic_9 = synthetic_8.replace('"identity":8', '"identity":9')
    w = _make_pipeline_worker(
        [
            _judge_json_scores(
                identity=7, face_quality=9, style_match=9, realism=9,
                artifact=9, commercial_readiness=9, action="face_swap",
            ),
            synthetic_8,
            synthetic_9,
            _judge_json_scores(
                identity=9, face_quality=9, style_match=9, realism=9,
                artifact=9, commercial_readiness=9, action="accept",
            ),
        ],
        swap_result=swapped,
    )
    alternate = FakePipelineClient([])
    alternate.model = "vendor/recovery-model"
    alternate.image_provider = "vendor"
    alternate.estimated_image_cost = 0.12
    w._gateway._openrouter_recovery = alternate

    _fp, meta = w.execute_hero_preview(
        "s_alternate_route",
        "prompt",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "title",
    )

    assert alternate.start_calls == 1
    assert meta["selected_candidate"]["deliverable"] is True
    assert meta["provider_invocations"][-1]["model"] == "vendor/recovery-model"
    assert meta["provider_invocations"][-1]["recovery"]["route_mode"] == "alternate"
    assert any(
        action.get("route_mode") == "alternate"
        and action.get("recovery_plan", {}).get("failure_streak") == 1
        for action in meta["agent_actions"]
    )


def test_variant_selection_prefers_more_real_earlier_stage(monkeypatch):
    """Delivery must pick the most real hard-gate-passing stage, not the last.

    The face-swap frame scores realism 9 and passes; the later blend triggers
    a local edit whose output passes but scores realism 8. Delivery must fall
    back to the face-swap file even though a later stage exists.
    """
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    swapped = SimpleNamespace(
        output_path=Path("/tmp/composition_identity.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    w = _make_pipeline_worker(
        [
            # scaffold: generic face → identity hard gate fails
            _judge_json_scores(identity=3, realism=9),
            # face swap: the most real passing version
            _judge_json_scores(identity=9, realism=9),
            # blend: artifact dip routes to a local edit (gate fails here)
            _judge_json_scores(identity=9, realism=9, artifact=7),
            # local edit: passes, but less real than the face swap
            _judge_json_scores(identity=9, realism=8),
        ],
        swap_result=swapped,
    )

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s_variant_pick",
        "Korean studio portrait.",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "title",
        template_path=None,
        shot_spec_metadata={
            "shot_id": "half_body",
            "framing": "medium half-body portrait",
            "pose": "standing at a three-quarter angle",
            "lighting": "soft window light",
            "lens": "50mm portrait lens",
        },
    )

    assert fp == "/tmp/composition_identity.png"
    assert meta["variant_selection"] == {
        "selected_stage": "composition_face_swap",
        "reason": "earlier_stage_more_real",
        "total_variants": 4,
    }
    assert meta["selected_candidate"]["filename"] == "composition_identity.png"
    assert meta["selected_candidate"]["deliverable"] is True
    assert meta["selected_candidate"]["identity_score"] == 9
    assert [
        variant["stage"]
        for variant in meta["selected_candidate"]["variants"]
    ] == [
        "composition_scaffold",
        "composition_face_swap",
        "composition_identity_blend",
        "local_edit",
    ]


def test_composition_stage_judge_failure_does_not_kill_pipeline(monkeypatch):
    """A failed scaffold-stage judge must not kill the pipeline.

    The failed stage is recorded with a judge_failed verdict and skipped by
    variant selection; later stages still deliver normally.
    """
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    swapped = SimpleNamespace(
        output_path=Path("/tmp/composition_identity.png"),
        swapped=True,
        message="swapped",
        source_face_count=4,
        target_face_count=1,
    )
    # Two real judge verdicts: the scaffold judge raises, face-swap and
    # identity-blend judges succeed.
    w = _make_pipeline_worker(
        [_judge_json(8), _judge_json(8)],
        swap_result=swapped,
    )
    provider = w._gateway._provider_for("CREATE_FROM_REFERENCES")
    real_judge = provider.judge
    judge_calls = {"count": 0}

    def flaky_judge(*args, **kwargs):
        judge_calls["count"] += 1
        if judge_calls["count"] == 1:
            raise RuntimeError("VLM judge timeout")
        return real_judge(*args, **kwargs)

    provider.judge = flaky_judge

    fp, meta = w.execute_generate_with_quality_pipeline(
        "s_stage_judge_failure",
        "Korean studio portrait.",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        "title",
        template_path=None,
        shot_spec_metadata={
            "shot_id": "half_body",
            "framing": "medium half-body portrait",
            "pose": "standing at a three-quarter angle",
            "lighting": "soft window light",
            "lens": "50mm portrait lens",
        },
    )

    assert fp == "/tmp/title_cand1_identity_blend.png"
    variants = meta["selected_candidate"]["variants"]
    assert [variant["stage"] for variant in variants] == [
        "composition_scaffold",
        "composition_face_swap",
        "composition_identity_blend",
    ]
    # judge_current_candidate converts the exception into a judge_failed hard
    # failure; that variant is skipped by variant selection.
    assert "judge_failed" in variants[0]["judgement"]["hard_failures"]
    assert variants[0]["gate_status"]["hard_gates_pass"] is False
    assert meta["variant_selection"] == {
        "selected_stage": "composition_identity_blend",
        "reason": "last_stage_most_real",
        "total_variants": 3,
    }


def test_synthetic_appearance_is_a_hard_delivery_failure():
    judgement = EvaluationService._parse_quality_judge_response(
        '{"scores":{"identity":null,"face_quality":9,"style_match":9,'
        '"realism":5,"artifact":9,"commercial_readiness":7},'
        '"hard_failures":["synthetic_appearance"],'
        '"recommended_action":"discard","notes":"plastic skin"}'
    )
    judgement["scores"]["identity"] = 9

    gate = EvaluationService._candidate_gate_status(judgement)

    assert gate["quality_pass"] is False
    assert gate["severe_quality_fail"] is True
    assert gate["hard_gates_pass"] is False


def test_pipeline_forwards_session_feedback_to_policy_engine():
    """Task 2: session_feedback must reach PolicyEngine.decide (was inert before).

    Previously the call passed metadata["history"] (judge iterations with no
    "event" field), so _feedback_modifier always returned 0.0. Now the
    session's event-tagged feedback is threaded through verbatim.
    """
    w = _make_pipeline_worker([_judge_json(9), _judge_json(9), _judge_json(9)])
    captured: list = []
    real_decide = w._agent_router.decide

    def spy_decide(judgement, **kwargs):
        captured.append(kwargs.get("session_feedback"))
        return real_decide(judgement, **kwargs)

    w._agent_router.decide = spy_decide  # type: ignore[assignment]

    feedback = [{"event": "not_like_me", "image_id": "img_x"}]
    w.execute_generate_with_quality_pipeline(
        "s_fb", "prompt", ["a.jpg"], "title",
        template_path=None, session_feedback=feedback,
    )

    assert captured, "decide() was never called"
    assert all(entry is feedback for entry in captured), (
        "session_feedback was not forwarded verbatim to the policy engine"
    )


def test_identity_attribute_profile_normalizes_model_output(tmp_path):
    ref_a = tmp_path / "a.jpg"
    ref_b = tmp_path / "b.jpg"
    ref_a.write_bytes(b"a")
    ref_b.write_bytes(b"b")

    class AttributeGateway:
        def judge(self, **kwargs):
            assert kwargs["current_image_path"] == str(ref_a)
            assert kwargs["reference_paths"] == [str(ref_b)]
            return """```json
            {"eyewear":"none","hair_length":"long","hair_color":"dark brown",
             "facial_hair":"none","distinctive_marks":["mole beside left eye"],
             "apparent_age_band":"adult_25_34"}
            ```"""

    profile = EvaluationService().extract_identity_attributes(
        AttributeGateway(), [str(ref_a), str(ref_b)]
    )

    assert profile["eyewear"] == "none"
    assert profile["hair_length"] == "long"
    assert profile["distinctive_marks"] == ["mole beside left eye"]


def test_identity_attribute_contract_forbids_invented_glasses():
    contract = EvaluationService.identity_attribute_contract({
        "eyewear": "none",
        "hair_length": "long",
        "hair_color": "dark brown",
        "facial_hair": "none",
        "apparent_age_band": "adult_25_34",
        "distinctive_marks": [],
    })

    assert "eyewear: none" in contract
    assert "Do not add glasses or sunglasses" in contract
    assert "hard constraint" in contract


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
