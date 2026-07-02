"""Job endpoints: submit generation/revision, check status.

Every mutation is ownership-gated (the caller must hold the session's owner
token). Revision is also tier-gated (free/standard/premium revision counts).
Generation is rate-limited per session to prevent draining the Gemini account.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .job_queue import queue
from .models import JobResponse, ReviseRequest, StyleKey
from .payment import check_tier_permission, TIER_LIMITS
from .security import limit_generation, require_owner, safe_id

router = APIRouter(prefix="/api", tags=["jobs"])
MVP_MAX_MULTI_STYLE_COUNT = 2


@router.post("/sessions/{session_id}/hero-preview", response_model=list[JobResponse])
async def start_hero_preview(
    session_id: str,
    style: str | None = None,
    state=Depends(require_owner),
):
    """Start generating the hero preview portrait for a session.

    Optional ``style`` query param overrides the session's default style
    (used for multi-style bundles where the hero preview uses the first
    selected style before the user unlocks the full set).
    """
    limit_generation(session_id)
    if not state.uploaded_photos:
        raise HTTPException(400, "Upload photos first")
    if state.status.value in ("generating",):
        raise HTTPException(409, "Generation already in progress")
    if state.hero_preview_generated:
        raise HTTPException(409, "Hero preview already generated for this session")

    try:
        jobs = await queue.submit_hero_preview(session_id, style_override=style)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not jobs:
        raise HTTPException(400, "No matching templates for this style/gender")
    return [j.to_response(position=queue.get_queue_position(j.job_id)) for j in jobs]


@router.post("/sessions/{session_id}/unlock", response_model=list[JobResponse])
async def unlock_full_set(
    session_id: str,
    state=Depends(require_owner),
):
    """Unlock the full portrait set after hero preview and payment."""
    limit_generation(session_id)
    if not state.hero_preview_generated:
        raise HTTPException(400, "Generate hero preview first")
    if state.unlocked:
        raise HTTPException(409, "Full set already unlocked")
    if state.tier == "free":
        raise HTTPException(403, "Upgrade to a paid tier to unlock the full set")

    try:
        jobs = await queue.submit_unlock(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not jobs:
        raise HTTPException(400, "No matching templates for this style/gender")
    return [j.to_response(position=queue.get_queue_position(j.job_id)) for j in jobs]


@router.post("/sessions/{session_id}/generate", response_model=list[JobResponse])
async def start_generation(
    session_id: str,
    state=Depends(require_owner),
):
    """Start generating portraits for all matching prompts in the session's style."""
    limit_generation(session_id)
    if not state.uploaded_photos:
        raise HTTPException(400, "Upload photos first")
    if state.status.value in ("generating",):
        raise HTTPException(409, "Generation already in progress")

    try:
        jobs = await queue.submit_generation(session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not jobs:
        raise HTTPException(400, "No matching templates for this style/gender")
    return [j.to_response(position=queue.get_queue_position(j.job_id)) for j in jobs]


@router.post("/sessions/{session_id}/revise/{image_id}", response_model=JobResponse)
async def submit_revision(
    session_id: str,
    image_id: str,
    req: ReviseRequest,
    state=Depends(require_owner),
):
    """Submit a revision instruction for a generated image."""
    limit_generation(session_id)
    # Validate image_id format before any lookup (path-safety + injection guard).
    safe = safe_id(image_id, label="image_id")

    if not check_tier_permission(state, "revise"):
        raise HTTPException(
            400, f"No revisions remaining (max {state.max_revisions}); upgrade to revise"
        )

    # Verify image exists in this session
    img = next((i for i in state.generated_images if i.image_id == safe), None)
    if not img:
        raise HTTPException(404, "Image not found in session")

    try:
        job = await queue.submit_revision(session_id, safe, req.instruction)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Image not found in session") from exc
    except PermissionError as exc:
        raise HTTPException(
            409,
            "Source image has not passed final QA and cannot be revised",
        ) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return job.to_response(position=queue.get_queue_position(job.job_id))


@router.get("/sessions/{session_id}/jobs", response_model=list[JobResponse])
async def list_jobs(session_id: str, state=Depends(require_owner)):
    """List all jobs for a session."""
    jobs = queue.get_jobs(session_id)
    return [j.to_response(position=queue.get_queue_position(j.job_id)) for j in jobs]


@router.get("/queue/status")
async def queue_status():
    """Current queue length and estimated wait.

    Returns ``is_busy`` (boolean) rather than the active session id — exposing
    which session is mid-generation is a privacy leak across users.
    """
    q_len = queue.queue_length()
    est_wait = q_len * 90  # rough: 90s per job
    return {
        "queue_length": q_len,
        "is_busy": queue.is_busy,
        "estimated_wait_seconds": est_wait,
    }


# ── Multi-style generation ──────────────────────────────

class MultiStyleRequest(BaseModel):
    styles: list[StyleKey]  # MVP comparison: exactly 2 style keys


@router.post("/sessions/{session_id}/generate-multi", response_model=list[JobResponse])
async def start_multi_style_generation(
    session_id: str,
    req: MultiStyleRequest,
    state=Depends(require_owner),
):
    """Generate one portrait per selected style for side-by-side comparison."""
    limit_generation(session_id)
    if not state.uploaded_photos:
        raise HTTPException(400, "Upload photos first")
    if len(req.styles) != MVP_MAX_MULTI_STYLE_COUNT:
        raise HTTPException(
            400,
            f"Select exactly {MVP_MAX_MULTI_STYLE_COUNT} styles for MVP comparison",
        )

    # Check tier allows this many styles
    limits = TIER_LIMITS[state.tier]
    if limits["max_styles"] < len(req.styles):
        raise HTTPException(403, f"当前套餐最多{limits['max_styles']}种风格，请升级")

    try:
        jobs = await queue.submit_multi_style_generation(session_id, req.styles)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not jobs:
        raise HTTPException(400, "No matching templates found for selected styles")

    return [j.to_response(position=queue.get_queue_position(j.job_id)) for j in jobs]
