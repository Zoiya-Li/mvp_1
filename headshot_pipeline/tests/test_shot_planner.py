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

from server import router_jobs, storage  # noqa: E402
from server.shot_planner import build_style_shot_plan  # noqa: E402
from server.job_queue import JobQueue  # noqa: E402
from server.models import (  # noqa: E402
    PricingTier,
    SessionState,
    SessionStatus,
    StyleKey,
)


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


def test_business_style_plans_complete_six_shot_set_from_templates():
    plan = build_style_shot_plan("business", "female", _style_data())
    assert len(plan) == 6
    assert [shot.shot_spec["shot_id"] for shot in plan] == [
        "closeup",
        "half_body",
        "environmental",
        "seated",
        "profile",
        "candid",
    ]
    assert plan[0].prompt_id == "bf_f_01_closeup"
    assert plan[1].prompt_id == "bf_f_01_half_body"
    assert plan[2].prompt_id == "bf_f_01_environmental"
    assert [shot.shot_spec["template_id"] for shot in plan] == [
        "bf_f_01", "bf_f_01", "bf_f_01",
        "bf_f_02", "bf_f_02", "bf_f_02",
    ]
    assert "seated posture unmistakably visible" in plan[3].shot_spec["framing"]
    assert "never crop through fingers" in plan[3].prompt
    assert "chair back" in plan[3].prompt
    assert "covered city arcade" in plan[2].shot_spec["environment"]
    assert "environment:" in plan[2].prompt
    assert "Identity preservation is a hard constraint" in plan[0].prompt
    assert plan[0].shot_spec["prompt_blocks"]["identity_block"] == (
        "derived_from_task_identity_pack"
    )
    assert len({shot.shot_spec["environment"] for shot in plan}) == 6
    assert all("office" not in shot.shot_spec["environment"].lower() for shot in plan)
    assert [shot.shot_spec["wardrobe"].split(":", 1)[0] for shot in plan] == [
        "Look A", "Look A", "Look A", "Look B", "Look B", "Look B",
    ]
    assert "covered city arcade" in plan[2].prompt
    assert "book-lined library lounge" in plan[3].prompt
    assert "Do not improvise an unplanned outfit family" in plan[5].prompt
    assert "unretouched real camera photograph" in plan[0].prompt
    assert "No beauty filter" in plan[4].prompt


def test_planner_prefers_descriptive_style_prompt_over_legacy_image_indices():
    style = _style_data()
    style["templates"][0]["style_prompt"] = "Detailed editorial wardrobe and mood."
    style["templates"][0]["gen_prompt"] = "Detailed editorial light and wardrobe."
    style["templates"][0]["prompt"] = "Copy image 1 onto image 2."

    shot = build_style_shot_plan("business", "female", style, max_shots=1)[0]

    assert "Detailed editorial wardrobe and mood" in shot.prompt
    assert "Detailed editorial light and wardrobe" not in shot.prompt
    assert "Copy image 1 onto image 2" not in shot.prompt


def test_non_id_style_fallback_does_not_inherit_legacy_headshot_geometry():
    style = _style_data()
    style["templates"][0].pop("style_prompt", None)
    style["templates"][0]["gen_prompt"] = (
        "Professional headshot. White gradient background. Centered 85mm close-up."
    )

    shot = build_style_shot_plan("business", "female", style, max_shots=1)[0]

    assert "white gradient background" not in shot.prompt.lower()
    assert "centered 85mm" not in shot.prompt.lower()
    assert "ShotSpec is the sole authority" in shot.prompt
    assert "studio headshot" in shot.prompt


def test_hero_inherits_style_without_conflicting_studio_geometry():
    style = _style_data()
    style["templates"][0]["gen_prompt"] = (
        "Professional headshot. Studio lighting. Charcoal gradient background. "
        "Centered 85mm close-up."
    )

    hero = build_style_shot_plan(
        "business", "female", style, max_shots=1, hero_only=True,
    )[0]

    assert "charcoal gradient" not in hero.prompt.lower()
    assert "centered 85mm" not in hero.prompt.lower()
    assert "Classic business" not in hero.prompt
    assert "regional beauty trend" in hero.prompt
    assert "physically readable real editorial location" in hero.prompt
    assert "No solid or gradient backdrop" in hero.prompt


def test_id_photo_keeps_single_standard_shot():
    plan = build_style_shot_plan("id_photo", "female", _style_data())
    assert len(plan) == 1
    assert plan[0].shot_spec["shot_id"] == "standard"


def test_multi_style_can_request_one_shot_per_style():
    plan = build_style_shot_plan("business", "female", _style_data(), max_shots=1)
    assert len(plan) == 1
    assert plan[0].shot_spec["shot_id"] == "closeup"


def test_catalog_direction_pins_template_and_explicit_six_frame_plan():
    style = _style_data()
    explicit = [
        {
            **shot,
            "environment": f"connected room frame {index}",
            "wardrobe": "one unchanged black tailored look",
            "style_prompt": "One concrete unretouched daylight studio session.",
        }
        for index, shot in enumerate([
            {
                "shot_id": "closeup", "label": "Opening",
                "framing": "chest-up", "pose": "three-quarter turn",
                "lighting": "side daylight", "lens": "50mm",
                "narrative": "arrival",
            },
            {
                "shot_id": "half_body", "label": "Half length",
                "framing": "head to waist", "pose": "relaxed standing",
                "lighting": "side daylight", "lens": "50mm",
                "narrative": "settling",
            },
            {
                "shot_id": "environmental", "label": "Room",
                "framing": "environmental", "pose": "off center",
                "lighting": "side daylight", "lens": "35mm",
                "narrative": "context",
            },
            {
                "shot_id": "seated", "label": "Seated",
                "framing": "seated waist-up", "pose": "open shoulders",
                "lighting": "side daylight", "lens": "50mm",
                "narrative": "pause",
            },
            {
                "shot_id": "profile", "label": "Turned",
                "framing": "turned chest-up", "pose": "gaze away",
                "lighting": "side daylight", "lens": "70mm",
                "narrative": "turn",
            },
            {
                "shot_id": "candid", "label": "Closing",
                "framing": "candid half-body", "pose": "in-between breath",
                "lighting": "side daylight", "lens": "50mm",
                "narrative": "close",
            },
        ])
    ]

    plan = build_style_shot_plan(
        "business",
        "female",
        style,
        template_id="bf_f_02",
        shot_overrides=explicit,
    )

    assert len(plan) == 6
    assert {shot.shot_spec["template_id"] for shot in plan} == {"bf_f_02"}
    assert [shot.shot_spec["environment"] for shot in plan] == [
        f"connected room frame {index}" for index in range(6)
    ]
    assert all(
        shot.shot_spec["prompt_blocks"]["style_block"]
        == "One concrete unretouched daylight studio session."
        for shot in plan
    )
    assert "do not invent another outfit" in (
        plan[0].shot_spec["prompt_blocks"]["set_continuity_block"]
    )


def test_catalog_hero_keeps_selected_first_scene_instead_of_generic_cafe():
    explicit = [{
        "shot_id": "closeup",
        "label": "Opening",
        "framing": "chest-up with breathing room",
        "pose": "subtle three-quarter turn",
        "environment": "a rain-dark brick lane-house window",
        "lighting": "mixed dusk and tungsten light",
        "lens": "50mm at f/4",
        "wardrobe": "one dark green qipao",
        "narrative": "arrival",
        "style_prompt": "One grounded location portrait session.",
    }]

    hero = build_style_shot_plan(
        "business",
        "female",
        _style_data(),
        hero_only=True,
        template_id="bf_f_01",
        shot_overrides=explicit,
    )[0]

    assert hero.shot_spec["environment"] == "a rain-dark brick lane-house window"
    assert "rain-dark brick lane-house window" in hero.prompt
    assert "neighborhood cafe" not in hero.prompt


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

    assert len(jobs) == 6
    assert [job.shot_spec["shot_id"] for job in jobs] == [
        "closeup",
        "half_body",
        "environmental",
        "seated",
        "profile",
        "candid",
    ]
    assert jobs[0].prompt_id == "bf_f_01_closeup"
    assert jobs[0].shot_spec["template_id"] == "bf_f_01"
    assert state.status == SessionStatus.generating


@pytest.mark.asyncio
async def test_unlock_uses_hero_as_cover_and_queues_five_remaining_shots(
    tmp_path, monkeypatch,
):
    q = JobQueue()
    q._prompts_data = {"styles": {"business": _style_data()}}
    state = _ready_state(tmp_path)
    state.record_session_consents(
        face_processing_consent=True,
        adult_subject_confirmed=True,
    )
    state.hero_preview_generated = True
    state.hero_preview_image_id = "img_hero"
    state.tier = PricingTier.standard
    q._sessions[state.session_id] = state
    monkeypatch.setattr(storage, "update_session_hero_preview", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(storage, "update_session_status", lambda *_args, **_kwargs: None)

    jobs = await q.submit_unlock(state.session_id)

    assert len(jobs) == 5
    assert [job.shot_spec["shot_id"] for job in jobs] == [
        "half_body",
        "environmental",
        "seated",
        "profile",
        "candid",
    ]
    assert all(job.job_type.value == "full_set" for job in jobs)
    assert all(job.template_path is None for job in jobs)
    assert state.unlocked is True
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
