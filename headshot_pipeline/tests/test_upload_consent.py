"""Tests for explicit face-processing consent on reference-photo upload."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.config import settings  # noqa: E402
from server.job_queue import JobQueue  # noqa: E402
from server.models import SessionState, StyleKey  # noqa: E402
from server import router_sessions  # noqa: E402


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _files(count: int) -> list[UploadFile]:
    return [
        UploadFile(file=io.BytesIO(PNG_BYTES), filename=f"selfie{idx}.png")
        for idx in range(count)
    ]


def _state(tmp_path: Path) -> SessionState:
    state = SessionState("s_consent", StyleKey.business, "female", "tok")
    state.upload_dir = tmp_path / "uploads"
    state.output_dir = tmp_path / "outputs"
    state.upload_dir.mkdir()
    state.output_dir.mkdir()
    return state


@pytest.mark.asyncio
async def test_upload_requires_explicit_face_processing_consent(tmp_path):
    with pytest.raises(HTTPException) as exc:
        await router_sessions.upload_photos(
            "s_consent",
            files=_files(settings.min_photos),
            face_processing_consent=False,
            adult_subject_confirmed=True,
            state=_state(tmp_path),
        )

    assert exc.value.status_code == 400
    assert "Face-processing consent is required" in exc.value.detail


@pytest.mark.asyncio
async def test_upload_requires_adult_subject_confirmation(tmp_path):
    with pytest.raises(HTTPException) as exc:
        await router_sessions.upload_photos(
            "s_consent",
            files=_files(settings.min_photos),
            face_processing_consent=True,
            adult_subject_confirmed=False,
            state=_state(tmp_path),
        )

    assert exc.value.status_code == 400
    assert "Adult-subject confirmation is required" in exc.value.detail


@pytest.mark.asyncio
async def test_upload_with_consent_reaches_save_path(tmp_path, monkeypatch):
    state = _state(tmp_path)
    saved: list[str] = []

    class FakeQueue:
        async def record_session_consents(
            self,
            session_id,
            *,
            face_processing_consent,
            adult_subject_confirmed,
        ):
            state.record_session_consents(
                face_processing_consent=face_processing_consent,
                adult_subject_confirmed=adult_subject_confirmed,
            )
            return state.session_consents.model_dump(mode="json")

        async def save_uploaded_photo(self, session_id, filename, content):
            saved.append(filename)
            path = state.upload_dir / filename
            path.write_bytes(content)
            state.uploaded_photos.append(path)
            return path

    monkeypatch.setattr(router_sessions, "queue", FakeQueue())

    response = await router_sessions.upload_photos(
        state.session_id,
        files=_files(settings.min_photos),
        face_processing_consent=True,
        adult_subject_confirmed=True,
        state=state,
    )

    assert len(saved) == settings.min_photos
    assert response.session_id == state.session_id
    assert response.session_consents.face_processing_consent is True
    assert response.session_consents.adult_subject_confirmed is True
    assert response.session_consents.no_training_by_default is True
    assert response.session_consents.cross_user_search_prohibited is True
    assert response.session_consents.long_term_face_library_prohibited is True
    assert response.session_consents.consented_at is not None


@pytest.mark.asyncio
async def test_uploaded_reference_files_get_stable_slot_prefixes(tmp_path, monkeypatch):
    state = _state(tmp_path)
    q = JobQueue()
    q._sessions[state.session_id] = state

    monkeypatch.setattr(
        "server.job_queue.assess_reference_photo",
        lambda path: {"filename": Path(path).name, "pass": True, "issues": []},
    )
    monkeypatch.setattr(
        "server.job_queue.assess_reference_identity_consistency",
        lambda paths: {"status": "unchecked", "pass": True, "issues": []},
    )

    first = await q.save_uploaded_photo(state.session_id, "z-last.png", PNG_BYTES)
    second = await q.save_uploaded_photo(state.session_id, "a-first.png", PNG_BYTES)

    assert first.name == "ref01_z-last.png"
    assert second.name == "ref02_a-first.png"
    assert [p.name for p in state.uploaded_photos] == [
        "ref01_z-last.png",
        "ref02_a-first.png",
    ]
