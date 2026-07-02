"""Regression tests for the template-based portrait shot planner."""

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

from server import router_jobs  # noqa: E402
from server.shot_planner import build_style_shot_plan  # noqa: E402
from server.job_queue import JobQueue  # noqa: E402
from server.models import SessionState, SessionStatus, StyleKey  # noqa: E402


def _style_data():
    return {
        "label_en": "Urban Professional",
        "templates": [
            {
                "id": "bf_f_01",
                "gender": "female",
                "label": "Classic business",
                "template_image": "templates/bf_f_01.png",
                "prompt": "Use this business style while preserving identity.",
            },
            {
                "id": "bf_f_02",
                "gender": "female",
                "label": "Environmental business",
                "template_image": "templates/bf_f_02.png",
                "prompt": "Use this business style while preserving identity.",
            },
        ],
    }


def test_business_style_plans_three_bounded_shots_from_templates():
    plan = build_style_shot_plan("business", "female", _style_data())
    assert len(plan) == 3
    assert [shot.shot_spec["shot_id"] for shot in plan] == [
        "closeup",
        "half_body",
        "environmental",
    ]
    assert plan[0].prompt_id == "bf_f_01_closeup"
    assert plan[1].prompt_id == "bf_f_02_half_body"
    assert plan[2].prompt_id == "bf_f_02_environmental"
    assert "Identity preservation is a hard constraint" in plan[0].prompt
    assert plan[0].shot_spec["prompt_blocks"]["identity_block"] == (
        "derived_from_task_identity_pack"
    )


def test_id_photo_keeps_single_standard_shot():
    plan = build_style_shot_plan("id_photo", "female", _style_data())
    assert len(plan) == 1
    assert plan[0].shot_spec["shot_id"] == "standard"


def test_multi_style_can_request_one_shot_per_style():
    plan = build_style_shot_plan("business", "female", _style_data(), max_shots=1)
    assert len(plan) == 1
    assert plan[0].shot_spec["shot_id"] == "closeup"


def _ready_state(tmp_path: Path) -> SessionState:
    state = SessionState("s_plan", StyleKey.business, "female", "tok")
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "output"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    for idx in range(4):
        photo = state.upload_dir / f"selfie{idx}.jpg"
        photo.write_bytes(b"jpg")
        state.uploaded_photos.append(photo)
        state.photo_quality[photo.name] = {
            "filename": photo.name,
            "status": "pass",
            "pass": True,
            "issues": [],
        }
    state.reference_quality = {
        "status": "pass",
        "pass": True,
        "total_photos": 4,
        "passed_photos": 4,
        "failed_photos": 0,
        "min_required": 4,
        "issues": [],
    }
    state.status = SessionStatus.ready
    return state


@pytest.mark.asyncio
async def test_submit_generation_requires_session_consents(tmp_path):
    q = JobQueue()
    q._prompts_data = {"styles": {"business": _style_data()}}
    state = _ready_state(tmp_path)
    q._sessions[state.session_id] = state

    with pytest.raises(ValueError) as exc:
        await q.submit_generation(state.session_id)

    assert "Face-processing consent is required" in str(exc.value)
    assert q.queue_length() == 0
    assert state.status == SessionStatus.ready


@pytest.mark.asyncio
async def test_submit_generation_queues_planned_shot_jobs(tmp_path):
    q = JobQueue()
    q._prompts_data = {"styles": {"business": _style_data()}}

    state = _ready_state(tmp_path)
    state.record_session_consents(
        face_processing_consent=True,
        adult_subject_confirmed=True,
    )
    q._sessions[state.session_id] = state

    jobs = await q.submit_generation(state.session_id)

    assert len(jobs) == 3
    assert [job.shot_spec["shot_id"] for job in jobs] == [
        "closeup",
        "half_body",
        "environmental",
    ]
    assert jobs[0].prompt_id == "bf_f_01_closeup"
    assert jobs[0].shot_spec["template_id"] == "bf_f_01"
    assert state.status == SessionStatus.generating


@pytest.mark.asyncio
async def test_multi_style_endpoint_keeps_mvp_scope_to_two_styles(tmp_path):
    state = _ready_state(tmp_path)

    with pytest.raises(router_jobs.HTTPException) as exc:
        await router_jobs.start_multi_style_generation(
            state.session_id,
            router_jobs.MultiStyleRequest(
                styles=[
                    StyleKey.business,
                    StyleKey.academic,
                    StyleKey.social,
                ]
            ),
            state=state,
        )

    assert exc.value.status_code == 400
    assert "exactly 2 styles" in exc.value.detail
