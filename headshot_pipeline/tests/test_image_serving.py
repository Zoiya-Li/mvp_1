"""Tests for final-delivery image serving policy."""

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

from server import router_sessions  # noqa: E402
from server.models import (  # noqa: E402
    FeedbackEvent,
    GeneratedImage,
    SessionState,
    StyleKey,
    UserFeedbackRequest,
    utcnow,
)


def _meta(deliverable: bool) -> dict:
    return {
        "selected_candidate": {
            "candidate_id": "cand_1",
            "deliverable": deliverable,
            "gate_status": {
                "hard_gates_pass": deliverable,
                "hard_gate_failures": [] if deliverable else ["identity_fail"],
            },
        }
    }


@pytest.mark.asyncio
async def test_download_rejects_registered_non_deliverable_image(tmp_path):
    state = SessionState("s_download", StyleKey.business, "female", "tok")
    state.output_dir = tmp_path
    state.generated_images.append(GeneratedImage(
        image_id="img_bad",
        url="/api/sessions/s_download/images/img_bad",
        prompt_id="closeup",
        turn=1,
        created_at=utcnow(),
        resemblance=_meta(False),
    ))

    with pytest.raises(router_sessions.HTTPException) as exc:
        await router_sessions.get_image(
            "s_download",
            "img_bad",
            download=True,
            state=state,
        )

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_preview_rejects_unregistered_output_file(tmp_path, monkeypatch):
    state = SessionState("s_preview", StyleKey.business, "female", "tok")
    state.output_dir = tmp_path
    (tmp_path / "img_orphan.png").write_bytes(b"not a real png")

    class FakeQueue:
        def get_image_path(self, *_args, **_kwargs):
            return tmp_path / "img_orphan.png"

    monkeypatch.setattr(router_sessions, "queue", FakeQueue())

    with pytest.raises(router_sessions.HTTPException) as exc:
        await router_sessions.get_image(
            "s_preview",
            "img_orphan",
            download=False,
            state=state,
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_feedback_endpoint_maps_non_deliverable_saved_event_to_conflict(monkeypatch):
    state = SessionState("s_feedback", StyleKey.business, "female", "tok")
    state.generated_images.append(GeneratedImage(
        image_id="img_bad",
        url="/api/sessions/s_feedback/images/img_bad",
        prompt_id="closeup",
        turn=1,
        created_at=utcnow(),
        resemblance=_meta(False),
    ))

    class FakeQueue:
        async def record_user_feedback(self, *_args, **_kwargs):
            raise PermissionError("img_bad")

    monkeypatch.setattr(router_sessions, "queue", FakeQueue())

    with pytest.raises(router_sessions.HTTPException) as exc:
        await router_sessions.submit_image_feedback(
            "s_feedback",
            "img_bad",
            UserFeedbackRequest(event=FeedbackEvent.downloaded),
            state=state,
        )

    assert exc.value.status_code == 409
