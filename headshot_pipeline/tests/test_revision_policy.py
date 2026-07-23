"""Tests for bounded manual revision requests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.delivery_label import AI_LABEL_KEY, AI_LABEL_VALUE, read_ai_metadata  # noqa: E402
from server.job_queue import JobQueue  # noqa: E402
from server.models import GeneratedImage, Job, JobStatus, JobType, SessionState, StyleKey  # noqa: E402
from server import storage  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_revision_db(tmp_path, monkeypatch):
    """Revision tests must not inherit a database initialized by another file."""
    monkeypatch.setattr(storage, "_DB_PATH", tmp_path / "revision-test.db")
    storage.init_db()


def _state(with_consent: bool = True) -> SessionState:
    state = SessionState("s_revision", StyleKey.business, "female", "tok")
    if with_consent:
        state.record_session_consents(
            face_processing_consent=True,
            adult_subject_confirmed=True,
        )
    return state


class _RevisionWorker:
    def __init__(self, output_path: Path, judgement: dict | None = None):
        self.output_path = output_path
        self.judgement = judgement or _revision_judgement()

    def execute_revise(self, session_id, instruction, title, source_image_path):
        Image = pytest.importorskip("PIL.Image")
        Image.new("RGB", (240, 160), color=(80, 120, 160)).save(
            self.output_path, format="PNG"
        )
        return str(self.output_path)

    def _judge_current_candidate(self, image_path, reference_photo_paths=None):
        return self.judgement

    def identity_thresholds_for_shot(self, shot_spec=None):
        return {
            "profile": "closeup",
            "identity_pass_threshold": 8.0,
            "identity_repair_threshold": 7.0,
        }


def _revision_judgement(identity: int = 9, artifact: int = 9) -> dict:
    return {
        "scores": {
            "identity": identity,
            "face_quality": 9,
            "style_match": 9,
            "artifact": artifact,
            "commercial_readiness": 9,
            "realism": 9,
        },
        "hard_failures": [],
        "recommended_action": "accept",
        "notes": "revision qa",
        "quality_evaluation": {
            "identity": {"score": identity / 10, "status": "pass"},
        },
    }


def _parent_image(image_id: str = "img_1234abcd", deliverable: bool = True) -> GeneratedImage:
    return GeneratedImage(
        image_id=image_id,
        url=f"/api/sessions/s_revision/images/{image_id}",
        prompt_id="business_closeup",
        turn=1,
        created_at=storage.utcnow(),
        resemblance={
            "shot_spec": {"shot_id": "closeup", "framing": "close-up portrait"},
            "selected_candidate": {
                "candidate_id": "cand_1",
                "deliverable": deliverable,
                "gate_status": {
                    "hard_gates_pass": deliverable,
                    "hard_gate_failures": [] if deliverable else ["identity_fail"],
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_revision_requires_session_consents():
    q = JobQueue()
    state = _state(with_consent=False)
    q._sessions[state.session_id] = state

    with pytest.raises(ValueError) as exc:
        await q.submit_revision(
            state.session_id,
            "img_1234abcd",
            "Make the lighting brighter",
        )

    assert "Face-processing consent is required" in str(exc.value)
    assert state.revisions_used == 0
    assert q.queue_length() == 0


@pytest.mark.asyncio
async def test_revision_rejects_open_ended_regeneration_requests():
    q = JobQueue()
    state = _state()
    state.generated_images.append(_parent_image())
    q._sessions[state.session_id] = state

    with pytest.raises(ValueError) as exc:
        await q.submit_revision(
            state.session_id,
            "img_1234abcd",
            "Not satisfied, please regenerate a different result",
        )

    assert "Only local retouching revisions" in str(exc.value)
    assert state.revisions_used == 0
    assert q.queue_length() == 0


@pytest.mark.asyncio
async def test_revision_wraps_allowed_request_as_local_edit():
    q = JobQueue()
    state = _state()
    state.generated_images.append(_parent_image())
    q._sessions[state.session_id] = state

    job = await q.submit_revision(
        state.session_id,
        "img_1234abcd",
        "Make the expression more natural and the image sharper",
    )

    assert state.revisions_used == 1
    assert q.queue_length() == 1
    assert job.instruction.startswith("LOCAL_EDIT only.")
    assert "Preserve identity" in job.instruction
    assert "Do not regenerate" in job.instruction


@pytest.mark.asyncio
async def test_revision_rejects_non_deliverable_source_image():
    q = JobQueue()
    state = _state()
    state.generated_images.append(_parent_image(deliverable=False))
    q._sessions[state.session_id] = state

    with pytest.raises(PermissionError):
        await q.submit_revision(
            state.session_id,
            "img_1234abcd",
            "Make the lighting brighter",
        )

    assert state.revisions_used == 0
    assert q.queue_length() == 0


@pytest.mark.asyncio
async def test_revision_delivery_records_local_edit_invocation_and_ai_label(tmp_path):
    q = JobQueue()
    q._worker = _RevisionWorker(tmp_path / "revision.png")
    state = _state()
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    parent = _parent_image("img_parent")
    state.generated_images.append(parent)
    q._sessions[state.session_id] = state
    job = Job(
        session_id=state.session_id,
        job_type=JobType.revise,
        prompt="LOCAL_EDIT only. Apply this bounded retouch request.",
        instruction="LOCAL_EDIT only. Apply this bounded retouch request.",
        revised_image_id=parent.image_id,
    )
    job.turn = 2
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.completed
    assert job.result_image is not None
    delivered = state.output_dir / f"{job.result_image.image_id}.png"
    file_meta = read_ai_metadata(delivered)
    assert file_meta[AI_LABEL_KEY] == AI_LABEL_VALUE
    meta = job.result_image.resemblance
    assert meta["pipeline"] == "manual_local_edit_v1"
    assert meta["provider_invocations"][0]["operation"] == "LOCAL_EDIT"
    assert meta["provider_invocations"][0]["parent_candidate_id"] == parent.image_id
    assert meta["selected_candidate"]["deliverable"] is True
    assert meta["final_evaluate"]["delivery_gate"]["status"] == "pass"
    assert meta["final_asset"]["metadata_ai_label"] is True
    assert meta["final_asset"]["operation"] == "FINAL_RENDER"
    assert meta["final_evaluate"]["status"] == "pass"
    assert meta["final_evaluate"]["ai_label_check"]["status"] == "pass"
    assert meta["final_evaluate"]["final_render"]["operation"] == "FINAL_RENDER"
    assert meta["provider_invocations"][-1]["operation"] == "FINAL_RENDER"


@pytest.mark.asyncio
async def test_revision_delivery_gate_failure_is_not_saved_to_gallery(tmp_path):
    q = JobQueue()
    q._worker = _RevisionWorker(
        tmp_path / "bad_revision.png",
        judgement=_revision_judgement(identity=5),
    )
    state = _state()
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    parent = _parent_image("img_parent")
    state.generated_images.append(parent)
    q._sessions[state.session_id] = state
    job = Job(
        session_id=state.session_id,
        job_type=JobType.revise,
        prompt="LOCAL_EDIT only. Apply this bounded retouch request.",
        instruction="LOCAL_EDIT only. Apply this bounded retouch request.",
        revised_image_id=parent.image_id,
    )
    job.turn = 2
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.failed
    assert job.result_image is None
    assert len(state.generated_images) == 1
    assert state.generated_images[0].image_id == parent.image_id
    assert "Final gate" in job.error


# -- Real GeminiWorker revise-path coverage ----------------------------------
# The tests above use _RevisionWorker, a hand-written fake whose
# execute_revise / _judge_current_candidate signatures mask the two production
# bugs they were written against (NameError on undefined `filepath`, and the
# AttributeError from a nonexistent `_judge_current_candidate` on GeminiWorker).
# The tests below exercise the REAL GeminiWorker methods to lock the fixes in.


def _real_revise_worker(tmp_path):
    """Build a real GeminiWorker with a fake gateway + evaluator, via __new__."""
    from server.gemini_worker import GeminiWorker
    from server.evaluation.evaluator import EvaluationService

    w = GeminiWorker.__new__(GeminiWorker)
    w.active_session_id = "s_revision"
    w._turn_counts = {}

    class _SpyGateway:
        def __init__(self):
            self.local_edit_calls = []
            self.judge_calls = []

        def local_edit(self, current_image_path, reference_paths, edit_prompt, title):
            self.local_edit_calls.append({
                "current_image_path": current_image_path,
                "reference_paths": list(reference_paths),
                "edit_prompt": edit_prompt,
                "title": title,
            })
            return str(tmp_path / "revised.png")

        def judge(self, current_image_path, reference_paths, judge_prompt, timeout=None):
            self.judge_calls.append({
                "current_image_path": current_image_path,
                "reference_paths": list(reference_paths),
                "judge_prompt": judge_prompt,
            })
            return '{"scores":{"identity":9},"hard_failures":[],"notes":"ok"}'

        def end_session(self):
            pass

    spy = _SpyGateway()
    w._gateway = spy
    w._eval_service = EvaluationService()
    return w, spy


def test_real_execute_revise_uses_source_image_path_not_undefined(tmp_path):
    """Regression: execute_revise referenced `filepath` before assignment.

    The local_edit call must now receive the resolved parent path as
    current_image_path, not an undefined local. references stay empty by design.
    """
    w, spy = _real_revise_worker(tmp_path)

    out = w.execute_revise(
        "s_revision",
        "Make the lighting brighter",
        "s_revision_rev_2",
        source_image_path=str(tmp_path / "parent.png"),
    )

    assert out == str(tmp_path / "revised.png")
    assert len(spy.local_edit_calls) == 1
    call = spy.local_edit_calls[0]
    assert call["current_image_path"] == str(tmp_path / "parent.png")
    assert call["reference_paths"] == []
    assert call["edit_prompt"] == "Make the lighting brighter"
    assert call["title"] == "s_revision_rev_2"
    # Turn counter advanced.
    assert w._turn_counts["s_revision"] == 2


def test_real_worker_has_judge_current_candidate_delegate(tmp_path, monkeypatch):
    """Regression: GeminiWorker had no _judge_current_candidate, so the job_queue
    revise path raised AttributeError. It now delegates to the eval service."""
    w, spy = _real_revise_worker(tmp_path)

    candidate_path = tmp_path / "revised.png"
    candidate_path.write_bytes(b"candidate")
    local_identity_refs = []
    monkeypatch.setattr(
        w._eval_service,
        "_local_image_quality_check",
        lambda *_args, **_kwargs: {
            "scores": {}, "hard_failures": [], "measurements": {}, "notes": ""
        },
    )
    monkeypatch.setattr(
        w._eval_service,
        "_local_identity_similarity_check",
        lambda _image_path, refs, **_kwargs: (
            local_identity_refs.extend(refs)
            or {"score": None, "cosine_similarity": None, "hard_failures": [], "notes": ""}
        ),
    )

    judgement = w._judge_current_candidate(
        str(candidate_path),
        ["ref1.jpg", "ref2.jpg"],
    )

    assert isinstance(judgement, dict)
    assert spy.judge_calls[0]["current_image_path"] == str(candidate_path)
    assert spy.judge_calls[0]["reference_paths"] == ["ref1.jpg"]
    assert "Image 1 is the generated candidate" in spy.judge_calls[0]["judge_prompt"]
    assert "Image 2, when present" in spy.judge_calls[0]["judge_prompt"]
    assert local_identity_refs == ["ref1.jpg", "ref2.jpg"]


def test_real_execute_revise_establishes_its_own_session(tmp_path):
    w, _spy = _real_revise_worker(tmp_path)
    w.active_session_id = "other_session"

    result = w.execute_revise(
        "s_revision",
        "brighter",
        "t",
        source_image_path=str(tmp_path / "parent.png"),
    )

    assert result == str(tmp_path / "revised.png")
    assert w.active_session_id == "s_revision"
