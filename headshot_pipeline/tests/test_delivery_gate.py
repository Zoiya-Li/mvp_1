"""Regression tests for the final generated-image delivery gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.job_queue import (  # noqa: E402
    JobQueue,
    _generation_metrics_from_event_rows,
    _is_permanent_provider_error,
    _is_transient_generation_error,
    _record_shot_completion,
    _save_generation_event,
    append_final_render_invocation,
    build_delivery_gate_check,
    final_duplicate_check,
    generation_passed_delivery_gate,
)
from server.config import settings  # noqa: E402
from server.delivery_label import AI_LABEL_KEY, AI_LABEL_VALUE, read_ai_metadata  # noqa: E402
from server.models import (  # noqa: E402
    GeneratedImage,
    Job,
    JobStatus,
    JobType,
    SessionState,
    SessionStatus,
    StyleKey,
)
from server import storage  # noqa: E402


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(storage, "_DB_PATH", None)
    storage.init_db()


def _metadata(deliverable: bool, hard_gates_pass: bool) -> dict:
    return {
        "candidates": [
            {
                "candidate_id": "cand_1",
                "path": "/tmp/flashshot/cand_1.png",
                "filename": "cand_1.png",
            }
        ],
        "provider_invocations": [
            {
                "operation": "CREATE_FROM_REFERENCES",
                "estimated_cost": 0.12,
                "latency_ms": 1000,
            }
        ],
        "selected_candidate": {
            "candidate_id": "cand_1",
            "deliverable": deliverable,
            "gate_status": {
                "hard_gates_pass": hard_gates_pass,
                "hard_gate_failures": [] if hard_gates_pass else ["identity_fail"],
            },
        }
    }


def test_generation_delivery_gate_requires_deliverable_and_hard_gates():
    assert generation_passed_delivery_gate(_metadata(True, True)) is True
    assert generation_passed_delivery_gate(_metadata(True, False)) is False
    assert generation_passed_delivery_gate(_metadata(False, True)) is False
    assert generation_passed_delivery_gate({"selected_candidate": {}}) is False
    assert generation_passed_delivery_gate(None) is False


def test_delivery_gate_check_reports_final_evaluate_issues():
    check = build_delivery_gate_check(_metadata(False, False))

    assert check["pass"] is False
    assert check["status"] == "fail"
    assert check["selected_candidate_id"] == "cand_1"
    assert check["hard_gate_failures"] == ["identity_fail"]
    assert check["issues"] == ["identity_fail", "not_deliverable"]


def test_full_set_batch_status_is_persisted_only_when_all_jobs_finish():
    q = JobQueue()
    state = SessionState("s_batch_status", StyleKey.business, "female", "tok")
    storage.save_session(
        state.session_id, state.owner_token, state.style.value,
        state.gender, state.created_at,
    )
    state.status = SessionStatus.generating
    q._sessions[state.session_id] = state
    jobs = [
        Job(
            session_id=state.session_id,
            job_type=JobType.full_set,
            prompt=f"shot {index}",
            prompt_id=f"shot_{index}",
        )
        for index in range(6)
    ]
    for job in jobs:
        q._jobs[job.job_id] = job
    for job in jobs[:5]:
        job.status = JobStatus.completed
    jobs[-1].status = JobStatus.processing

    q._persist_generation_batch_status(state, jobs[0])
    assert state.status == SessionStatus.generating
    assert storage.load_session_row(state.session_id)["status"] == "generating"

    jobs[-1].status = JobStatus.completed
    q._persist_generation_batch_status(state, jobs[-1])
    assert state.status == SessionStatus.done
    assert storage.load_session_row(state.session_id)["status"] == "done"


def test_full_set_batch_failure_is_persisted_after_batch_settles():
    q = JobQueue()
    state = SessionState("s_batch_failure", StyleKey.business, "female", "tok")
    storage.save_session(
        state.session_id, state.owner_token, state.style.value,
        state.gender, state.created_at,
    )
    q._sessions[state.session_id] = state
    jobs = [
        Job(
            session_id=state.session_id,
            job_type=JobType.full_set,
            prompt=f"shot {index}",
            prompt_id=f"shot_{index}",
        )
        for index in range(6)
    ]
    for job in jobs:
        job.status = JobStatus.completed
        q._jobs[job.job_id] = job
    jobs[2].status = JobStatus.failed

    q._persist_generation_batch_status(state, jobs[2])

    assert state.status == SessionStatus.failed
    assert storage.load_session_row(state.session_id)["status"] == "failed"


@pytest.mark.asyncio
async def test_transient_full_set_failure_gets_one_automatic_whole_job_retry():
    q = JobQueue()
    state = SessionState("s_auto_retry", StyleKey.cinematic, "female", "tok")
    storage.save_session(
        state.session_id, state.owner_token, state.style.value,
        state.gender, state.created_at,
    )
    q._sessions[state.session_id] = state
    failed = Job(
        session_id=state.session_id,
        job_type=JobType.full_set,
        prompt="half body portrait",
        prompt_id="shot_half_body",
        shot_spec={"shot_id": "half_body"},
    )
    failed.status = JobStatus.failed
    q._jobs[failed.job_id] = failed
    _save_generation_event(
        failed, JobStatus.failed, "transient_provider_error", "provider timeout"
    )

    replacement = await q._schedule_automatic_full_set_retry(
        state, failed, "transient_provider_error"
    )

    assert replacement is not None
    assert failed.job_id not in q._jobs
    assert replacement.automatic_retry_count == 1
    assert replacement.automatic_retry_reason == "transient_provider_error"
    assert q._queue.get_nowait() is replacement
    q._queue.task_done()
    assert state.pipeline_metrics["automatic_full_set_retries"] == 1
    assert storage.load_session_row(state.session_id)["status"] == "generating"

    replacement.status = JobStatus.failed
    assert await q._schedule_automatic_full_set_retry(
        state, replacement, "transient_provider_error"
    ) is None
    assert list(q._jobs.values()) == [replacement]


@pytest.mark.asyncio
async def test_quality_gate_failure_uses_lower_risk_shot_variant():
    q = JobQueue()
    state = SessionState("s_no_quality_retry", StyleKey.cinematic, "female", "tok")
    failed = Job(
        session_id=state.session_id,
        job_type=JobType.full_set,
        prompt="half body portrait",
        prompt_id="shot_half_body",
        shot_spec={"shot_id": "half_body"},
    )
    failed.status = JobStatus.failed
    q._jobs[failed.job_id] = failed

    replacement = await q._schedule_automatic_full_set_retry(
        state, failed, "delivery_gate_failed"
    )

    assert replacement is not None
    assert replacement.prompt != failed.prompt
    assert replacement.shot_spec["shot_id"] == "half_body"
    assert replacement.shot_spec["canonical_shot_id"] == "half_body"
    assert replacement.shot_spec["shot_variant"] == "waist_up_relaxed"
    assert replacement.shot_spec["recovery_failure_class"] == "unknown_quality"
    assert q._queue.get_nowait() is replacement
    q._queue.task_done()


def test_automatic_retry_metrics_survive_event_rehydration():
    original = Job(
        "s_retry_metrics", JobType.full_set, "shot",
        prompt_id="shot_profile", shot_spec={"shot_id": "profile"},
    )
    original.status = JobStatus.failed
    _save_generation_event(
        original, JobStatus.failed, "delivery_gate_failed", "quality failed"
    )
    replacement = Job(
        "s_retry_metrics", JobType.full_set, "shot",
        prompt_id="shot_profile", shot_spec={"shot_id": "profile"},
        automatic_retry_count=1,
        automatic_retry_reason="delivery_gate_failed",
    )
    replacement.status = JobStatus.completed
    _save_generation_event(
        replacement, JobStatus.completed, result_image_id="img_recovered"
    )

    metrics = _generation_metrics_from_event_rows(
        storage.load_generation_events("s_retry_metrics")
    )

    assert metrics["generation_attempts"] == 2
    assert metrics["generation_failures"] == 1
    assert metrics["automatic_full_set_retries"] == 1
    assert metrics["automatic_full_set_retry_successes"] == 1


def test_provider_error_retry_classification_is_fail_closed():
    insufficient = RuntimeError(
        'SiliconFlow images/generations failed with HTTP 403: '
        '{"message":"account balance is insufficient"}'
    )
    overloaded = RuntimeError("OpenRouter API error 503: upstream unavailable")
    bug = RuntimeError("unexpected metadata shape")

    assert _is_permanent_provider_error(insufficient) is True
    assert _is_transient_generation_error(insufficient) is False
    assert _is_transient_generation_error(overloaded) is True
    assert _is_permanent_provider_error(overloaded) is False
    assert _is_transient_generation_error(bug) is False
    assert _is_permanent_provider_error(bug) is False


@pytest.mark.asyncio
async def test_provider_account_failure_disables_generation_and_fast_fails_queue():
    q = JobQueue()
    q._worker = type(
        "Worker",
        (),
        {"provider_readiness": {"pass": True, "provider": "siliconflow"}},
    )()
    account_error = RuntimeError(
        "SiliconFlow images/generations failed with HTTP 403: "
        "account balance is insufficient"
    )
    q._mark_provider_unavailable(account_error)

    assert q.generation_ready is False
    assert q.worker_readiness_error == str(account_error)

    state = SessionState("s_provider_down", StyleKey.fashion, "female", "tok")
    storage.save_session(
        state.session_id, state.owner_token, state.style.value,
        state.gender, state.created_at,
    )
    q._sessions[state.session_id] = state
    job = Job(
        state.session_id, JobType.full_set, "profile portrait",
        prompt_id="shot_profile", shot_spec={"shot_id": "profile"},
    )
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.failed
    assert state.pipeline_metrics["generation_attempts"] == 1
    assert state.pipeline_metrics["generation_failures"] == 1
    assert state.pipeline_metrics["failed_generation_reasons"] == {
        "provider_unavailable": 1
    }
    events = storage.load_generation_events(state.session_id)
    assert events[0]["failure_reason"] == "provider_unavailable"


def test_shot_completion_counts_deliverable_only_when_hard_gate_passes():
    state = SessionState("s_gate_metrics", StyleKey.business, "female", "tok")
    job = Job(
        session_id=state.session_id,
        job_type=JobType.generate,
        prompt="prompt",
        prompt_id="business_closeup",
        shot_spec={"shot_id": "closeup"},
    )

    _record_shot_completion(
        state,
        job,
        _metadata(deliverable=True, hard_gates_pass=False),
        result_image_id="img_bad",
    )

    shot = state.pipeline_metrics["shot_metrics"]["closeup"]
    assert shot["completed"] == 1
    assert shot["deliverable_count"] == 0


def test_final_render_invocation_is_recorded_on_delivered_asset():
    meta = _metadata(True, True)
    meta["shot_spec"] = {"shot_id": "closeup"}
    meta["provider_invocations"] = [
        {"invocation_id": "create_1", "operation": "CREATE_FROM_REFERENCES"}
    ]

    append_final_render_invocation(meta, "img_1234abcd", latency_ms=12)

    final = meta["provider_invocations"][-1]
    assert final["operation"] == "FINAL_RENDER"
    assert final["provider"] == "local"
    assert final["model"] == "flashshot_delivery_packager_v1"
    assert final["cost"] == 0.0
    assert final["estimated_cost"] == 0.0
    assert final["final_asset_id"] == "img_1234abcd"
    assert final["parent_candidate_id"] == "cand_1"
    assert final["shot_id"] == "closeup"


def _pattern_image(path: Path, inverted: bool = False) -> None:
    Image = pytest.importorskip("PIL.Image")

    img = Image.new("RGB", (240, 160), color=(0, 0, 0) if not inverted else (255, 255, 255))
    pixels = img.load()
    for x in range(240):
        for y in range(160):
            left_half = x < 120
            bright = left_half if not inverted else not left_half
            pixels[x, y] = (230, 230, 230) if bright else (20, 20, 20)
    img.save(path, format="PNG")


def test_final_duplicate_check_rejects_near_duplicate_asset(tmp_path):
    existing = tmp_path / "img_existing.png"
    candidate = tmp_path / "candidate.png"
    _pattern_image(existing)
    _pattern_image(candidate)

    result = final_duplicate_check(candidate, [existing])

    assert result["pass"] is False
    assert result["status"] == "fail"
    assert "duplicate_final_asset" in result["issues"]
    assert result["measurements"]["closest_match"]["image_id"] == "img_existing"


def test_final_duplicate_check_allows_distinct_asset(tmp_path):
    existing = tmp_path / "img_existing.png"
    candidate = tmp_path / "candidate.png"
    _pattern_image(existing)
    _pattern_image(candidate, inverted=True)

    result = final_duplicate_check(candidate, [existing])

    assert result["pass"] is True
    assert result["status"] == "pass"


class _RejectedWorker:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def execute_generate_with_quality_pipeline(
        self,
        session_id,
        prompt,
        photos,
        title,
        template_path,
        progress_cb,
        shot_spec,
        session_feedback=None,
    ):
        self.output_path.write_bytes(b"not-delivered")
        return str(self.output_path), _metadata(deliverable=False, hard_gates_pass=False)

    execute_hero_preview = execute_generate_with_quality_pipeline


class _DuplicateWorker:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def execute_generate_with_quality_pipeline(
        self,
        session_id,
        prompt,
        photos,
        title,
        template_path,
        progress_cb,
        shot_spec,
        session_feedback=None,
    ):
        _pattern_image(self.output_path)
        return str(self.output_path), _metadata(deliverable=True, hard_gates_pass=True)


class _AcceptedWorker:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def execute_generate_with_quality_pipeline(
        self,
        session_id,
        prompt,
        photos,
        title,
        template_path,
        progress_cb,
        shot_spec,
        session_feedback=None,
    ):
        _pattern_image(self.output_path, inverted=True)
        return str(self.output_path), _metadata(deliverable=True, hard_gates_pass=True)


@pytest.mark.asyncio
async def test_delivered_generation_records_ai_label_audit_metadata(tmp_path):
    q = JobQueue()
    q._worker = _AcceptedWorker(tmp_path / "worker_output.png")

    state = SessionState("s_label", StyleKey.business, "female", "tok")
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    for idx in range(4):
        photo = state.upload_dir / f"selfie{idx}.jpg"
        photo.write_bytes(b"jpg")
        state.uploaded_photos.append(photo)
    state.status = SessionStatus.generating
    q._sessions[state.session_id] = state

    job = Job(
        session_id=state.session_id,
        job_type=JobType.generate,
        prompt="Generate a portrait.",
        prompt_id="business_closeup",
        shot_spec={"shot_id": "closeup"},
    )
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.completed
    assert job.result_image is not None
    delivered = state.output_dir / f"{job.result_image.image_id}.png"
    file_meta = read_ai_metadata(delivered)
    assert file_meta[AI_LABEL_KEY] == AI_LABEL_VALUE
    final_asset = state.generated_images[0].resemblance["final_asset"]
    assert final_asset["image_id"] == job.result_image.image_id
    assert final_asset["metadata_ai_label"] is True
    assert final_asset["visible_label_reserved"] is True
    assert isinstance(final_asset["visible_ai_label"], bool)
    final_eval = state.generated_images[0].resemblance["final_evaluate"]
    assert final_eval["status"] == "pass"
    assert final_eval["delivery_gate"]["status"] == "pass"
    assert final_eval["duplicate_check"]["status"] == "pass"
    assert final_eval["ai_label_check"]["status"] == "pass"
    assert final_eval["final_render"]["operation"] == "FINAL_RENDER"


@pytest.mark.asyncio
async def test_non_deliverable_generation_is_not_saved_to_gallery(tmp_path):
    q = JobQueue()
    q._worker = _RejectedWorker(tmp_path / "worker_output.png")

    state = SessionState("s_gate", StyleKey.business, "female", "tok")
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    for idx in range(4):
        photo = state.upload_dir / f"selfie{idx}.jpg"
        photo.write_bytes(b"jpg")
        state.uploaded_photos.append(photo)
    state.status = SessionStatus.generating
    q._sessions[state.session_id] = state

    job = Job(
        session_id=state.session_id,
        job_type=JobType.generate,
        prompt="Generate a portrait.",
        prompt_id="business_closeup",
        shot_spec={"shot_id": "closeup"},
    )
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.failed
    assert "没有写真通过最终质量检查" in job.error
    assert "identity_fail" in job.error
    assert job.result_image is None
    assert state.status == SessionStatus.failed
    assert state.generated_images == []
    assert list(state.output_dir.glob("img_*.png")) == []
    metrics = state.to_response().pipeline_metrics
    assert metrics["generation_attempts"] == 1
    assert metrics["generation_failures"] == 1
    assert metrics["failed_generation_reasons"] == {"delivery_gate_failed": 1}
    assert metrics["deliverable_rate"] == 0
    assert metrics["total_provider_invocations"] == 1
    assert metrics["estimated_total_cost"] == 0.12
    assert metrics["candidates_generated"] == 1
    assert metrics["shot_metrics"]["closeup"]["attempts"] == 1
    assert metrics["shot_metrics"]["closeup"]["completed"] == 0
    assert metrics["shot_metrics"]["closeup"]["failed"] == 1
    assert metrics["shot_metrics"]["closeup"]["failure_reasons"] == {
        "delivery_gate_failed": 1
    }
    assert metrics["shot_metrics"]["closeup"]["failure_rate"] == 1
    assert metrics["shot_metrics"]["closeup"]["first_pass_rate"] == 0
    rows = storage.load_generation_events(state.session_id)
    assert len(rows) == 1
    assert rows[0]["job_id"] == job.job_id
    assert rows[0]["status"] == JobStatus.failed.value
    assert rows[0]["failure_reason"] == "delivery_gate_failed"
    assert rows[0]["prompt_id"] == "business_closeup"
    metadata = json.loads(rows[0]["metadata_json"])
    assert metadata["final_evaluate"]["delivery_gate"]["status"] == "fail"
    assert metadata["final_evaluate"]["delivery_gate"]["issues"] == [
        "identity_fail",
        "not_deliverable",
    ]
    assert metadata["candidates"][0]["path"] == "cand_1.png"
    assert metadata["provider_invocations"][0]["operation"] == "CREATE_FROM_REFERENCES"


@pytest.mark.asyncio
@pytest.mark.parametrize("job_type", [JobType.hero_preview, JobType.full_set])
async def test_all_generation_products_reject_non_deliverable_images(tmp_path, job_type):
    q = JobQueue()
    q._worker = _RejectedWorker(tmp_path / f"{job_type.value}_worker_output.png")

    state = SessionState(f"s_{job_type.value}", StyleKey.business, "female", "tok")
    state.upload_dir = tmp_path / f"{job_type.value}_uploads"
    state.output_dir = tmp_path / f"{job_type.value}_outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    for idx in range(4):
        photo = state.upload_dir / f"selfie{idx}.jpg"
        photo.write_bytes(b"jpg")
        state.uploaded_photos.append(photo)
    state.status = SessionStatus.generating
    q._sessions[state.session_id] = state

    job = Job(
        session_id=state.session_id,
        job_type=job_type,
        prompt="Generate a portrait.",
        prompt_id="business_closeup",
        shot_spec={"shot_id": "closeup"},
    )
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.failed
    assert "没有写真通过最终质量检查" in job.error
    assert state.generated_images == []
    assert state.hero_preview_generated is False
    assert state.hero_preview_image_id is None
    assert list(state.output_dir.glob("img_*.png")) == []
    assert state.pipeline_metrics["generation_failures"] == 1


@pytest.mark.asyncio
async def test_duplicate_generation_is_not_saved_to_gallery(tmp_path):
    q = JobQueue()
    q._worker = _DuplicateWorker(tmp_path / "worker_output.png")

    state = SessionState("s_dup", StyleKey.business, "female", "tok")
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    for idx in range(4):
        photo = state.upload_dir / f"selfie{idx}.jpg"
        photo.write_bytes(b"jpg")
        state.uploaded_photos.append(photo)
    existing_path = state.output_dir / "img_existing.png"
    _pattern_image(existing_path)
    state.generated_images.append(
        GeneratedImage(
            image_id="img_existing",
            url="/x",
            prompt_id="closeup",
            turn=1,
            created_at=storage.utcnow(),
            resemblance=_metadata(deliverable=True, hard_gates_pass=True),
        )
    )
    state.pipeline_metrics = {"generation_attempts": 1}
    state.status = SessionStatus.reviewing
    q._sessions[state.session_id] = state

    job = Job(
        session_id=state.session_id,
        job_type=JobType.generate,
        prompt="Generate a portrait.",
        prompt_id="business_closeup",
        shot_spec={"shot_id": "closeup"},
    )
    q._jobs[job.job_id] = job

    await q._execute_job(job)

    assert job.status == JobStatus.failed
    assert "too similar to an existing delivered image" in job.error
    assert job.result_image is None
    assert state.status == SessionStatus.reviewing
    assert [img.image_id for img in state.generated_images] == ["img_existing"]
    assert sorted(p.name for p in state.output_dir.glob("img_*.png")) == [
        "img_existing.png"
    ]
    metrics = state.to_response().pipeline_metrics
    assert metrics["generation_attempts"] == 2
    assert metrics["generation_failures"] == 1
    assert metrics["failed_generation_reasons"] == {"duplicate_final_asset": 1}
    assert metrics["deliverable_rate"] == 0.5
    assert metrics["total_provider_invocations"] == 2
    assert metrics["estimated_total_cost"] == 0.24
    assert metrics["estimated_cost_per_deliverable"] == 0.24
    assert metrics["candidates_generated"] == 2
    assert metrics["shot_metrics"]["closeup"]["attempts"] == 1
    assert metrics["shot_metrics"]["closeup"]["completed"] == 0
    assert metrics["shot_metrics"]["closeup"]["failed"] == 1
    assert metrics["shot_metrics"]["closeup"]["failure_reasons"] == {
        "duplicate_final_asset": 1
    }
    rows = storage.load_generation_events(state.session_id)
    assert len(rows) == 1
    assert rows[0]["job_id"] == job.job_id
    assert rows[0]["status"] == JobStatus.failed.value
    assert rows[0]["failure_reason"] == "duplicate_final_asset"
    metadata = json.loads(rows[0]["metadata_json"])
    assert metadata["final_evaluate"]["delivery_gate"]["status"] == "pass"
    assert metadata["final_evaluate"]["duplicate_check"]["status"] == "fail"
    assert metadata["candidates"][0]["path"] == "cand_1.png"
