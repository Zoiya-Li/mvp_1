"""WebSocket endpoint for real-time generation updates.

Handshake auth: the caller must pass the session's owner token as ``?token=``.
The Origin header is checked against the allowed CORS origins to prevent
cross-site WS hijacking. If either check fails the connection is closed with a
4401 before any session state is leaked.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .config import settings
from .job_queue import queue
from .security import require_owner_ws

router = APIRouter(tags=["websocket"])


def _origin_allowed(origin: str | None) -> bool:
    """A WebSocket is allowed only from a configured origin (CSWSH defense)."""
    if not origin:
        # No Origin = same-origin / non-browser client; allow in dev only.
        # In production behind a browser, Origin is always present.
        return any("localhost" in o or "127.0.0.1" in o for o in settings.cors_origins)
    return origin in settings.cors_origins


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    """Real-time updates for a session: job started, progress, image ready, failed."""
    # Origin check BEFORE accepting — a disallowed origin never gets a session.
    if not _origin_allowed(ws.headers.get("origin")):
        await ws.close(code=4403)
        return

    # Token check BEFORE accepting (require_owner_ws closes with 4401 on fail).
    state = await require_owner_ws(ws, session_id)
    if state is None:
        return  # already closed

    await ws.accept()
    queue.add_ws(session_id, ws)

    # Send current state on connect
    await ws.send_json({
        "type": "state",
        "status": state.status.value,
        "generated_images": [img.model_dump(mode="json") for img in state.generated_images],
        "revisions_used": state.revisions_used,
        "tier": state.tier.value,
    })

    try:
        # Keep connection alive — client can also send pings
        while True:
            try:
                data = await ws.receive_text()
                # Client can send "ping" to keep alive
                if data == "ping":
                    await ws.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    finally:
        queue.remove_ws(session_id, ws)
