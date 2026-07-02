"""Session endpoints: create, upload photos, get status, get images.

All session-scoped reads/writes are guarded by the ownership dependency
(``require_owner`` / ``require_owner_query``) — the caller must present the
session's secret owner token. Photo filenames and image_ids are validated to
prevent path traversal. Uploaded bytes are checked for a real image magic
signature and size-capped before they touch disk.
"""

from __future__ import annotations

import asyncio
import mimetypes

import fastapi
from fastapi import (
    APIRouter, Depends, Form, HTTPException, Request, UploadFile, File, Query,
)
from fastapi.responses import FileResponse

from .config import settings
from .delivery_policy import (
    find_registered_image,
    image_or_source_passed_final_gate,
)
from .job_queue import queue
from .models import (
    CreateSessionRequest,
    SessionResponse,
    StyleListResponse,
    UserFeedbackRequest,
    UserFeedbackResponse,
)
from .security import (
    is_within,
    limit_session_create,
    require_owner,
    require_owner_query,
    sanitize_filename,
    safe_id,
    validate_image_bytes,
)

router = APIRouter(prefix="/api", tags=["sessions"])

# Cap how many bytes we read into memory per uploaded file at once. The full
# file is capped by max_file_size_mb; this chunk size is just for streaming.
_READ_CHUNK = 1024 * 1024  # 1 MiB


@router.get("/styles", response_model=StyleListResponse)
async def get_styles():
    """Return available styles and their template images (public catalog)."""
    data = queue.get_styles()
    styles = []
    for key, style_data in data.get("styles", {}).items():
        # v2: "templates" key; fallback to "prompts" for v1 compat
        template_list = style_data.get("templates", style_data.get("prompts", []))
        templates = []
        for t in template_list:
            templates.append({
                "id": t["id"],
                "gender": t.get("gender", "unknown"),
                "label": t.get("label", t["id"]),
                "template_image": t.get("template_image"),
            })
        styles.append({
            "key": key,
            "label": style_data.get("label", key),
            "label_en": style_data.get("label_en", ""),
            "use_cases": style_data.get("use_cases", []),
            "templates": templates,
        })
    return StyleListResponse(styles=styles)


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    req: CreateSessionRequest,
    request: Request,
    _rl=Depends(limit_session_create),
):
    """Create a new portrait session.

    Rate-limited per client IP. Returns the secret ``owner_token`` ONCE — the
    client must store it and send it back on every session-scoped call.
    """
    state = queue.create_session(req.style, req.gender)
    return state.to_response(include_token=True)


async def _read_capped(file: UploadFile) -> bytes:
    """Read an UploadFile into memory, rejecting anything over the size cap.

    Checks ``file.size`` (if the transport reports it) up front so a multi-GB
    upload can't exhaust memory before we finish reading. Then streams in
    chunks and aborts the moment the cap is exceeded.
    """
    cap = settings.max_file_size_mb * 1024 * 1024
    declared = getattr(file, "size", None)
    if declared is not None and declared > cap:
        raise HTTPException(
            400, f"File {file.filename} exceeds {settings.max_file_size_mb}MB"
        )
    buf = bytearray()
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > cap:
            raise HTTPException(
                400, f"File {file.filename} exceeds {settings.max_file_size_mb}MB"
            )
    return bytes(buf)


@router.post("/sessions/{session_id}/photos", response_model=SessionResponse)
async def upload_photos(
    session_id: str,
    files: list[UploadFile] = File(...),
    face_processing_consent: bool = Form(default=False),
    adult_subject_confirmed: bool = Form(default=False),
    state=Depends(require_owner),
):
    """Upload reference photos (4-6 images). Ownership-gated + validated."""
    if not face_processing_consent:
        raise HTTPException(
            400,
            "Face-processing consent is required before uploading reference photos",
        )
    if not adult_subject_confirmed:
        raise HTTPException(
            400,
            "Adult-subject confirmation is required for portrait generation",
        )
    if len(files) < settings.min_photos:
        raise HTTPException(400, f"Upload at least {settings.min_photos} photos")
    if len(files) > settings.max_photos:
        raise HTTPException(400, f"Maximum {settings.max_photos} photos")
    if len(state.uploaded_photos) + len(files) > settings.max_photos:
        raise HTTPException(400, f"Total photos cannot exceed {settings.max_photos}")

    await queue.record_session_consents(
        session_id,
        face_processing_consent=face_processing_consent,
        adult_subject_confirmed=adult_subject_confirmed,
    )

    for f in files:
        content = await _read_capped(f)
        validate_image_bytes(content)
        # Filename is validated + reduced to a safe basename (no traversal).
        safe_name = sanitize_filename(f.filename or "photo.jpg")
        await queue.save_uploaded_photo(session_id, safe_name, content)

    return state.to_response()


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, state=Depends(require_owner)):
    """Get session state and generated images."""
    return state.to_response()


@router.get("/sessions/{session_id}/images/{image_id}")
async def get_image(
    session_id: str,
    image_id: str,
    download: bool = Query(default=False),
    state=Depends(require_owner_query),
):
    """Serve a generated image. Token via ``?token=`` (loaded as <img src>).

    With ``?download=1`` the response carries ``Content-Disposition: attachment``
    so the browser saves the file. This is the ONLY reliable way to force a
    download when the API origin differs from the page origin — the HTML
    ``download`` attribute is ignored for cross-origin URLs, so without an
    attachment header the browser would navigate to / preview the image instead.
    """
    safe = safe_id(image_id, label="image_id")
    if find_registered_image(state, safe) is None:
        raise HTTPException(404, "Image not found")
    if download and not image_or_source_passed_final_gate(state, safe):
        raise HTTPException(
            409,
            "Image has not passed final QA and cannot be downloaded",
        )
    path = queue.get_image_path(session_id, safe)
    if not path:
        raise HTTPException(404, "Image not found")
    # Defense-in-depth: confirm the resolved path is still inside the session dir.
    if not is_within(state.output_dir, path):
        raise HTTPException(404, "Image not found")
    media_type, _ = mimetypes.guess_type(str(path))
    mt = media_type or "image/png"
    if download:
        # filename= makes FileResponse emit Content-Disposition: attachment.
        return FileResponse(path, media_type=mt, filename=path.name)
    return FileResponse(path, media_type=mt)


@router.post(
    "/sessions/{session_id}/images/{image_id}/feedback",
    response_model=UserFeedbackResponse,
)
async def submit_image_feedback(
    session_id: str,
    image_id: str,
    req: UserFeedbackRequest,
    state=Depends(require_owner),
):
    """Record user feedback for quality metrics and future calibration."""
    safe = safe_id(image_id, label="image_id")
    if not any(img.image_id == safe for img in state.generated_images):
        raise HTTPException(404, "Image not found in session")
    try:
        record = await queue.record_user_feedback(
            session_id,
            safe,
            req.event,
            req.reason,
            req.score,
        )
    except FileNotFoundError:
        raise HTTPException(404, "Image not found in session")
    except PermissionError:
        raise HTTPException(
            409,
            "Image has not passed final QA and cannot be recorded as saved or selected",
        )
    return UserFeedbackResponse(
        feedback_id=record["feedback_id"],
        session_id=record["session_id"],
        image_id=record["image_id"],
        event=record["event"],
        reason=record["reason"],
        score=record["score"],
        created_at=record["created_at"],
    )


@router.get("/sessions/{session_id}/photos/{filename}")
async def get_uploaded_photo(
    session_id: str,
    filename: str,
    state=Depends(require_owner_query),
):
    """Serve an uploaded reference photo (for before/after comparison).

    ``filename`` is validated to a safe basename and the resolved path must
    stay inside the session upload dir.
    """
    if not state.upload_dir:
        raise HTTPException(404, "Session not found")
    safe = sanitize_filename(filename)
    path = state.upload_dir / safe
    if not is_within(state.upload_dir, path) or not path.exists():
        raise HTTPException(404, "Photo not found")
    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=media_type or "image/jpeg")


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, state=Depends(require_owner)):
    """Delete a session and its files."""
    await queue.delete_session(session_id)
    return {"ok": True}
