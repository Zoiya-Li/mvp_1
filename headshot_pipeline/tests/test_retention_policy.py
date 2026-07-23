from __future__ import annotations

from datetime import datetime, timezone

import pytest

from server import main, portrait_storage, storage
from server.config import settings


@pytest.mark.asyncio
async def test_layered_retention_expires_sources_outputs_then_metadata(tmp_path, monkeypatch):
    rows = iter([
        [{"session_id": "s_source"}],
        [{"session_id": "s_generated"}],
        [{"session_id": "s_metadata"}],
    ])
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(storage, "utcnow", lambda: datetime(2026, 7, 13, tzinfo=timezone.utc))
    monkeypatch.setattr(storage, "list_stale_sessions", lambda cutoff: next(rows))
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")

    async def expire_sources(session_id):
        calls.append(("sources", session_id))

    async def expire_outputs(session_id):
        calls.append(("outputs", session_id))

    async def delete_session(session_id):
        calls.append(("metadata", session_id))

    monkeypatch.setattr(main.queue, "expire_session_sources", expire_sources)
    monkeypatch.setattr(main.queue, "expire_session_outputs", expire_outputs)
    monkeypatch.setattr(main.queue, "delete_session", delete_session)
    monkeypatch.setattr(
        portrait_storage,
        "project_for_legacy_session",
        lambda session_id: {
            "project_id": f"prj_{session_id}",
            "user_id": "usr_retention",
        } if session_id != "s_metadata" else None,
    )
    monkeypatch.setattr(
        portrait_storage,
        "expire_project_sources",
        lambda project_id, user_id: [],
    )
    monkeypatch.setattr(
        portrait_storage,
        "delete_project_data",
        lambda project_id, user_id: calls.append(("project", project_id)),
    )

    await main._run_retention_sweep_once()

    assert calls == [
        ("sources", "s_source"),
        ("project", "prj_s_generated"),
        ("outputs", "s_generated"),
        ("metadata", "s_metadata"),
    ]
