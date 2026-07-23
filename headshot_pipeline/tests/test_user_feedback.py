"""Tests for user feedback persistence and delivery metrics."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server import storage  # noqa: E402
from server.config import settings  # noqa: E402
from server.job_queue import JobQueue  # noqa: E402
from server.models import (  # noqa: E402
    FeedbackEvent,
    GeneratedImage,
    PaymentStatus,
    SessionState,
    StyleKey,
)


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_DB_PATH", None)
    storage.init_db()
    return tmp_path


def _state_with_images() -> SessionState:
    state = SessionState("s_fb", StyleKey.business, "female", "tok")
    state.generated_images = [
        GeneratedImage(
            image_id="img_good",
            url="/x",
            prompt_id="closeup",
            turn=1,
            created_at=storage.utcnow(),
            resemblance={
                "strategy": {
                    "identity_threshold_profile": {
                        "profile": "closeup",
                        "identity_pass_threshold": 8,
                        "identity_repair_threshold": 7,
                    }
                },
                "selected_candidate": {
                    "candidate_id": "cand_1",
                    "deliverable": True,
                    "identity_score": 9,
                    "gate_status": {
                        "hard_gates_pass": True,
                        "hard_gate_failures": [],
                        "identity_threshold_profile": "closeup",
                        "identity_pass_threshold": 8,
                        "identity_repair_threshold": 7,
                    },
                },
                "candidates": [
                    {
                        "candidate_id": "cand_1",
                        "gate_status": {"identity_pass": True},
                    },
                    {
                        "candidate_id": "cand_2",
                        "gate_status": {"identity_pass": False},
                    },
                    {
                        "candidate_id": "cand_3",
                        "gate_status": {"identity_pass": True},
                    },
                ],
                "budget": {"identity_repairs_used": 1},
                "provider_invocations": [
                    {
                        "provider": "openrouter",
                        "model": "gemini-test",
                        "operation": "CREATE_FROM_REFERENCES",
                        "prompt_version": "controlled_candidate_v2",
                        "estimated_cost": 0.12,
                        "latency_ms": 1000,
                    },
                    {
                        "provider": "local",
                        "model": "inswapper_128",
                        "operation": "IDENTITY_REPAIR",
                        "prompt_version": None,
                        "estimated_cost": 0.0,
                        "latency_ms": 200,
                    },
                    {
                        "provider": "local",
                        "model": "flashshot_delivery_packager_v1",
                        "operation": "FINAL_RENDER",
                        "prompt_version": None,
                        "estimated_cost": 0.0,
                        "latency_ms": 10,
                    },
                ],
            },
        ),
        GeneratedImage(
            image_id="img_bad",
            url="/y",
            prompt_id="half_body",
            turn=1,
            created_at=storage.utcnow(),
            resemblance={
                "strategy": {
                    "identity_threshold_profile": {
                        "profile": "medium",
                        "identity_pass_threshold": 7.5,
                        "identity_repair_threshold": 6.5,
                    }
                },
                "selected_candidate": {
                    "candidate_id": "cand_1",
                    "deliverable": False,
                    "identity_score": 7,
                    "gate_status": {
                        "hard_gates_pass": False,
                        "hard_gate_failures": ["identity_fail"],
                        "identity_threshold_profile": "medium",
                        "identity_pass_threshold": 7.5,
                        "identity_repair_threshold": 6.5,
                    },
                },
                "candidates": [
                    {
                        "candidate_id": "cand_1",
                        "gate_status": {"identity_pass": False},
                    },
                    {
                        "candidate_id": "cand_2",
                        "gate_status": {"identity_pass": False},
                    },
                    {
                        "candidate_id": "cand_3",
                        "gate_status": {"identity_pass": False},
                    },
                ],
                "provider_invocations": [
                    {
                        "provider": "openrouter",
                        "model": "gemini-test",
                        "operation": "CREATE_FROM_REFERENCES",
                        "prompt_version": "controlled_candidate_v2",
                        "estimated_cost": 0.12,
                        "latency_ms": 3000,
                    },
                    {
                        "provider": "local",
                        "model": "flashshot_delivery_packager_v1",
                        "operation": "FINAL_RENDER",
                        "prompt_version": None,
                        "estimated_cost": 0.0,
                        "latency_ms": 20,
                    }
                ],
            },
        ),
    ]
    return state


def test_session_feedback_summary_tracks_saved_and_identity_feedback():
    state = _state_with_images()
    state.user_feedback = [
        {"image_id": "img_good", "event": "downloaded"},
        {"image_id": "img_good", "event": "selected"},
        {"image_id": "img_good", "event": "looks_like_me", "score": 2},
        {"image_id": "img_bad", "event": "not_like_me", "score": 0},
    ]

    summary = state.to_response().feedback_summary

    assert summary["total_generated"] == 2
    assert summary["ai_deliverable_count"] == 1
    assert summary["ai_deliverable_rate"] == 0.5
    assert summary["downloaded_count"] == 1
    assert summary["qualified_saved_count"] == 1
    assert summary["qualified_downloaded_count"] == 1
    assert summary["qualified_selected_count"] == 1
    assert summary["user_saved_rate"] == 0.5
    assert summary["qualified_saved_rate"] == 0.5
    assert summary["qualified_downloaded_rate"] == 0.5
    assert summary["qualified_selected_rate"] == 0.5
    assert summary["not_like_me_rate"] == 0.5

    metrics = state.to_response().pipeline_metrics
    assert metrics["qualified_saved_count"] == 1
    assert metrics["qualified_downloaded_count"] == 1
    assert metrics["qualified_selected_count"] == 1
    assert metrics["north_star_qualified_save_rate"] == 0.5
    assert metrics["qualified_saved_rate"] == 0.5
    assert metrics["qualified_downloaded_rate"] == 0.5
    assert metrics["qualified_selected_rate"] == 0.5
    assert metrics["shot_metrics"]["closeup"]["attempts"] == 1
    assert metrics["shot_metrics"]["closeup"]["completed"] == 1
    assert metrics["shot_metrics"]["closeup"]["deliverable_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["downloaded_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["selected_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["liked_identity_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["user_saved_rate"] == 1
    assert metrics["shot_metrics"]["closeup"]["user_selected_rate"] == 1
    assert metrics["shot_metrics"]["closeup"]["not_like_me_rate"] == 0
    assert metrics["shot_metrics"]["closeup"]["identity_first_pass_rate"] == 0.6667
    assert metrics["shot_metrics"]["half_body"]["attempts"] == 1
    assert metrics["shot_metrics"]["half_body"]["deliverable_count"] == 0
    assert metrics["shot_metrics"]["half_body"]["not_like_me_count"] == 1
    assert metrics["shot_metrics"]["half_body"]["not_like_me_rate"] == 1
    assert metrics["shot_metrics"]["half_body"]["identity_first_pass_rate"] == 0
    assert metrics["identity_threshold_metrics"]["closeup"]["delivered_count"] == 1
    assert metrics["identity_threshold_metrics"]["closeup"]["identity_pass_threshold"] == 8
    assert metrics["identity_threshold_metrics"]["closeup"]["avg_ai_identity_score"] == 9
    assert metrics["identity_threshold_metrics"]["closeup"]["avg_user_identity_score"] == 2
    assert metrics["identity_threshold_metrics"]["closeup"]["not_like_me_rate"] == 0
    assert metrics["identity_threshold_metrics"]["medium"]["delivered_count"] == 1
    assert metrics["identity_threshold_metrics"]["medium"]["identity_pass_threshold"] == 7.5
    assert metrics["identity_threshold_metrics"]["medium"]["avg_ai_identity_score"] == 7
    assert metrics["identity_threshold_metrics"]["medium"]["avg_user_identity_score"] == 0
    assert metrics["identity_threshold_metrics"]["medium"]["not_like_me_rate"] == 1


def test_session_feedback_summary_excludes_non_deliverable_downloads_from_north_star():
    state = _state_with_images()
    state.user_feedback = [
        {"image_id": "img_bad", "event": "downloaded"},
        {"image_id": "img_bad", "event": "selected"},
    ]

    response = state.to_response()
    summary = response.feedback_summary
    metrics = response.pipeline_metrics

    assert summary["downloaded_count"] == 1
    assert summary["selected_count"] == 1
    assert summary["user_saved_rate"] == 0.5
    assert summary["qualified_saved_count"] == 0
    assert summary["qualified_selected_count"] == 0
    assert summary["qualified_saved_rate"] == 0
    assert metrics["qualified_saved_count"] == 0
    assert metrics["qualified_selected_count"] == 0
    assert metrics["north_star_qualified_save_rate"] == 0


def test_metrics_require_hard_gate_for_deliverable_counts():
    state = _state_with_images()
    img = state.generated_images[0]
    img.resemblance["selected_candidate"]["deliverable"] = True
    img.resemblance["selected_candidate"]["gate_status"]["hard_gates_pass"] = False
    img.resemblance["selected_candidate"]["gate_status"]["hard_gate_failures"] = [
        "identity_fail"
    ]
    state.generated_images = [img]
    state.user_feedback = [
        {"image_id": img.image_id, "event": "downloaded"},
        {"image_id": img.image_id, "event": "selected"},
    ]

    response = state.to_response()
    summary = response.feedback_summary
    metrics = response.pipeline_metrics

    assert summary["ai_deliverable_count"] == 0
    assert summary["qualified_saved_count"] == 0
    assert summary["qualified_selected_count"] == 0
    assert metrics["deliverable_count"] == 0
    assert metrics["qualified_saved_count"] == 0
    assert metrics["qualified_selected_count"] == 0
    assert metrics["shot_metrics"]["closeup"]["deliverable_count"] == 0


@pytest.mark.asyncio
async def test_queue_records_and_persists_user_feedback(tmp_db):
    q = JobQueue()
    state = _state_with_images()
    state.upload_dir = settings.upload_dir / state.session_id
    state.output_dir = settings.output_dir / state.session_id
    state.upload_dir.mkdir(parents=True)
    state.output_dir.mkdir(parents=True)
    q._sessions[state.session_id] = state

    record = await q.record_user_feedback(
        state.session_id,
        "img_good",
        FeedbackEvent.downloaded,
        reason="saved",
        score=2,
    )

    rows = storage.load_user_feedback(state.session_id)
    assert len(rows) == 1
    assert rows[0]["feedback_id"] == record["feedback_id"]
    assert rows[0]["event"] == "downloaded"
    assert state.to_response().feedback_summary["downloaded_count"] == 1


@pytest.mark.asyncio
async def test_identity_feedback_triggers_learning_calibration(tmp_db):
    class FakeLearningLayer:
        def __init__(self):
            self.recorded = []
            self.calibrate_calls = 0

        def record_feedback(self, **kwargs):
            self.recorded.append(kwargs)

        def calibrate(self):
            self.calibrate_calls += 1

    q = JobQueue()
    q._learning_layer = FakeLearningLayer()
    state = _state_with_images()
    state.upload_dir = settings.upload_dir / state.session_id
    state.output_dir = settings.output_dir / state.session_id
    state.upload_dir.mkdir(parents=True)
    state.output_dir.mkdir(parents=True)
    q._sessions[state.session_id] = state

    await q.record_user_feedback(
        state.session_id,
        "img_good",
        FeedbackEvent.not_like_me,
        reason="jawline drift",
        score=0,
    )

    assert q._learning_layer.recorded[0]["event"] == "not_like_me"
    assert q._learning_layer.calibrate_calls == 1


@pytest.mark.asyncio
async def test_queue_rejects_saved_feedback_for_non_deliverable_image(tmp_db):
    q = JobQueue()
    state = _state_with_images()
    state.upload_dir = settings.upload_dir / state.session_id
    state.output_dir = settings.output_dir / state.session_id
    state.upload_dir.mkdir(parents=True)
    state.output_dir.mkdir(parents=True)
    q._sessions[state.session_id] = state

    with pytest.raises(PermissionError):
        await q.record_user_feedback(
            state.session_id,
            "img_bad",
            FeedbackEvent.downloaded,
            reason="spoofed_download",
        )

    assert storage.load_user_feedback(state.session_id) == []
    assert state.user_feedback == []


def test_session_pipeline_metrics_tracks_cost_and_invocations():
    state = _state_with_images()
    state.photo_quality = {
        "ref01_front.png": {"filename": "ref01_front.png", "pass": True},
        "ref02_smile.png": {"filename": "ref02_smile.png", "pass": True},
        "ref03_left.png": {"filename": "ref03_left.png", "pass": True},
        "ref04_right.png": {
            "filename": "ref04_right.png",
            "pass": False,
            "issues": ["too_blurry"],
        },
    }
    state.reference_quality = {
        "pass": False,
        "issues": ["ref04_right.png:too_blurry"],
    }

    metrics = state.to_response().pipeline_metrics

    assert metrics["input_photo_count"] == 4
    assert metrics["input_photo_passed"] == 3
    assert metrics["input_photo_failed"] == 1
    assert metrics["input_photo_pass_rate"] == 0.75
    assert metrics["reference_quality_pass"] is False
    assert metrics["reference_quality_issue_count"] == 1
    assert metrics["total_images"] == 2
    assert metrics["deliverable_count"] == 1
    assert metrics["generation_attempts"] == 2
    assert metrics["generation_failures"] == 0
    assert metrics["deliverable_rate"] == 0.5
    assert metrics["delivered_image_deliverable_rate"] == 0.5
    assert metrics["total_provider_invocations"] == 5
    assert metrics["create_from_reference_invocations"] == 2
    assert metrics["operation_counts"]["CREATE_FROM_REFERENCES"] == 2
    assert metrics["operation_counts"]["IDENTITY_REPAIR"] == 1
    assert metrics["operation_counts"]["FINAL_RENDER"] == 2
    invocation_metrics = metrics["provider_invocation_metrics"]
    assert invocation_metrics["by_operation"]["CREATE_FROM_REFERENCES"] == {
        "invocations": 2,
        "successes": 2,
        "failures": 0,
        "success_rate": 1,
        "estimated_cost": 0.24,
        "avg_cost_per_invocation": 0.12,
        "avg_latency_ms": 2000,
        "p50_latency_ms": 1000,
        "p95_latency_ms": 3000,
    }
    assert invocation_metrics["by_operation"]["FINAL_RENDER"]["invocations"] == 2
    assert invocation_metrics["by_provider_model"][
        "openrouter:gemini-test"
    ]["estimated_cost"] == 0.24
    assert invocation_metrics["by_prompt_version"][
        "controlled_candidate_v2"
    ]["invocations"] == 2
    assert metrics["estimated_total_cost"] == 0.24
    assert metrics["estimated_cost_per_image"] == 0.12
    assert metrics["estimated_cost_per_deliverable"] == 0.24
    assert metrics["avg_api_calls_per_image"] == 2.5
    assert metrics["candidates_generated"] == 6
    assert metrics["avg_candidates_per_image"] == 3
    assert metrics["identity_first_pass_candidates"] == 6
    assert metrics["identity_first_passes"] == 2
    assert metrics["identity_first_pass_rate"] == 0.3333
    assert metrics["identity_repairs"] == 1
    assert metrics["identity_repair_rate"] == 0.5
    assert metrics["avg_provider_latency_ms"] == 846
    assert metrics["p50_provider_latency_ms"] == 200
    assert metrics["p95_provider_latency_ms"] == 3000


def test_session_pipeline_metrics_reads_spec_cost_field():
    state = _state_with_images()
    for img in state.generated_images:
        for invocation in img.resemblance.get("provider_invocations", []):
            if "estimated_cost" in invocation:
                invocation["cost"] = invocation.pop("estimated_cost")

    metrics = state.to_response().pipeline_metrics

    assert metrics["total_provider_invocations"] == 5
    assert metrics["estimated_total_cost"] == 0.24
    assert metrics["estimated_cost_per_image"] == 0.12
    assert metrics["estimated_cost_per_deliverable"] == 0.24


def test_session_pipeline_metrics_tracks_agent_action_success_rates():
    state = _state_with_images()
    good_meta = state.generated_images[0].resemblance
    good_meta["selected_candidate"]["candidate_id"] = "cand_4"
    good_meta["budget"] = {
        "identity_repairs_used": 1,
        "local_edits_used": 1,
        "regenerations_used": 1,
    }
    good_meta["face_swap"] = {"applied": True}
    good_meta["local_edit"] = {"applied": True}
    good_meta["candidates"] = [
        {"candidate_id": "cand_1"},
        {"candidate_id": "cand_4", "regenerated_from_candidate_id": "cand_1"},
    ]
    state.pipeline_metrics = {
        "failed_identity_repairs": 1,
        "failed_local_edits": 1,
        "failed_regenerations": 1,
    }

    metrics = state.to_response().pipeline_metrics
    action_metrics = metrics["agent_action_metrics"]

    assert action_metrics["IDENTITY_REPAIR"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }
    assert action_metrics["LOCAL_EDIT"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }
    assert action_metrics["REGENERATE_FROM_ORIGINAL"] == {
        "attempts": 2,
        "successes": 1,
        "success_rate": 0.5,
    }
    assert metrics["identity_repair_success_rate"] == 0.5
    assert metrics["local_edit_success_rate"] == 0.5
    assert metrics["regeneration_success_rate"] == 0.5


def test_session_pipeline_metrics_tracks_p50_p95_delivery_latency():
    state = _state_with_images()
    state.created_at = storage.utcnow()
    state.generated_images[0].created_at = state.created_at + timedelta(seconds=12)
    state.generated_images[1].created_at = state.created_at + timedelta(seconds=44)
    state.generated_images[1].resemblance["selected_candidate"]["deliverable"] = True
    state.generated_images[1].resemblance["selected_candidate"]["gate_status"][
        "hard_gates_pass"
    ] = True
    state.generated_images[1].resemblance["selected_candidate"]["gate_status"][
        "hard_gate_failures"
    ] = []

    metrics = state.to_response().pipeline_metrics

    assert metrics["deliverable_count"] == 2
    assert metrics["p50_delivery_latency_seconds"] == 12
    assert metrics["p95_delivery_latency_seconds"] == 44


def test_session_pipeline_metrics_tracks_refund_rate():
    state = _state_with_images()
    state.payment_id = "pay_refunded"
    state.payment_status = PaymentStatus.refunded

    metrics = state.to_response().pipeline_metrics

    assert metrics["payment_status"] == "refunded"
    assert metrics["paid_payment_count"] == 1
    assert metrics["refunded_payment_count"] == 1
    assert metrics["refund_rate"] == 1


def test_session_pipeline_metrics_use_attempts_for_deliverable_rate():
    state = _state_with_images()
    state.user_feedback = [
        {"image_id": "img_good", "event": "downloaded"},
        {"image_id": "img_good", "event": "selected"},
        {"image_id": "img_good", "event": "looks_like_me", "score": 2},
    ]
    state.pipeline_metrics = {
        "generation_attempts": 3,
        "generation_failures": 1,
        "failed_generation_reasons": {"delivery_gate_failed": 1},
        "failed_provider_invocations": 1,
        "failed_create_from_reference_invocations": 1,
        "failed_operation_counts": {"CREATE_FROM_REFERENCES": 1},
        "failed_estimated_cost": 0.12,
        "failed_candidates_generated": 3,
        "failed_initial_identity_candidates": 3,
        "failed_initial_identity_passes": 1,
        "failed_regenerations": 1,
        "failed_latency_values": [4000],
        "shot_metrics": {
            "closeup": {
                "attempts": 2,
                "completed": 1,
                "failed": 1,
                "deliverable_count": 1,
                "failure_reasons": {"delivery_gate_failed": 1},
                "provider_invocations": 4,
                "estimated_cost": 0.24,
                "candidates_generated": 6,
                "identity_first_pass_candidates": 3,
                "identity_first_passes": 2,
                "regenerations": 1,
            },
            "half_body": {
                "attempts": 1,
                "completed": 0,
                "failed": 1,
                "deliverable_count": 0,
                "failure_reasons": {"duplicate_final_asset": 1},
            },
        },
    }

    metrics = state.to_response().pipeline_metrics

    assert metrics["total_images"] == 2
    assert metrics["deliverable_count"] == 1
    assert metrics["generation_attempts"] == 3
    assert metrics["generation_failures"] == 1
    assert metrics["qualified_saved_count"] == 1
    assert metrics["north_star_qualified_save_rate"] == 0.3333
    assert metrics["failed_generation_reasons"] == {"delivery_gate_failed": 1}
    assert metrics["delivered_image_deliverable_rate"] == 0.5
    assert metrics["deliverable_rate"] == 0.3333
    assert metrics["generation_failure_rate"] == 0.3333
    assert metrics["total_provider_invocations"] == 6
    assert metrics["create_from_reference_invocations"] == 3
    assert metrics["operation_counts"]["CREATE_FROM_REFERENCES"] == 3
    assert metrics["estimated_total_cost"] == 0.36
    assert metrics["estimated_cost_per_image"] == 0.12
    assert metrics["estimated_cost_per_deliverable"] == 0.36
    assert metrics["candidates_generated"] == 9
    assert metrics["avg_candidates_per_image"] == 3
    assert metrics["identity_first_pass_candidates"] == 9
    assert metrics["identity_first_passes"] == 3
    assert metrics["identity_first_pass_rate"] == 0.3333
    assert metrics["regenerations"] == 1
    assert metrics["regeneration_rate"] == 0.3333
    assert metrics["avg_provider_latency_ms"] == 1371.67
    assert metrics["p50_provider_latency_ms"] == 200
    assert metrics["p95_provider_latency_ms"] == 4000
    assert metrics["shot_metrics"]["closeup"]["attempts"] == 2
    assert metrics["shot_metrics"]["closeup"]["first_pass_rate"] == 0.5
    assert metrics["shot_metrics"]["closeup"]["failure_rate"] == 0.5
    assert metrics["shot_metrics"]["closeup"]["deliverable_rate"] == 0.5
    assert metrics["shot_metrics"]["closeup"]["identity_first_pass_rate"] == 0.6667
    assert metrics["shot_metrics"]["closeup"]["estimated_cost_per_deliverable"] == 0.24
    assert metrics["shot_metrics"]["closeup"]["avg_candidates_per_attempt"] == 3
    assert metrics["shot_metrics"]["closeup"]["downloaded_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["selected_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["liked_identity_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["user_saved_rate"] == 1
    assert metrics["shot_metrics"]["closeup"]["user_selected_rate"] == 1
    assert metrics["shot_metrics"]["half_body"]["failure_rate"] == 1
    assert metrics["shot_metrics"]["half_body"]["estimated_cost_per_deliverable"] == 0
    assert metrics["identity_threshold_metrics"]["closeup"]["avg_user_identity_score"] == 2
    assert metrics["identity_threshold_metrics"]["closeup"]["liked_identity_count"] == 1


def test_in_progress_generation_attempt_is_not_reported_as_failure():
    state = _state_with_images()
    state.pipeline_metrics = {
        "generation_attempts": 3,
        "generation_failures": 0,
    }

    metrics = state.to_response().pipeline_metrics

    assert metrics["generation_attempts"] == 3
    assert metrics["generation_failures"] == 0
    assert metrics["generation_failure_rate"] == 0
