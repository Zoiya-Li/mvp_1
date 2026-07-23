"""API v2 for the mobile-first portrait platform."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import mimetypes
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import storage
from .config import settings
from .apple_identity import AppleIdentityError, apple_identity_verifier
from .apple_iap import AppleIAPError, apple_iap_verifier
from .delivery_policy import image_or_source_passed_final_gate
from .delivery_label import (
    CLEAN_EXPORT_RETENTION_DAYS,
    CLEAN_EXPORT_TERMS_VERSION,
    clean_export_path,
)
from .evaluation.set_evaluator import evaluate_portrait_set
from .inspiration_analyzer import analyze_with_provider, inspiration_generation_prompt
from .job_queue import queue
from .models import (
    FeedbackEvent,
    JobResponse,
    PricingTier,
    SessionStatus,
    StyleKey,
    UserFeedbackResponse,
)
from .payment import PaymentService
from .portrait_domain import (
    CreateProjectRequest,
    AppleNotificationRequest,
    ApplePurchaseClaimRequest,
    ApplePurchaseClaimResponse,
    AppleSignInRequest,
    AuthenticatedUserResponse,
    CreatePortraitOrderRequest,
    CreateShareRecipeRequest,
    CleanExportRequest,
    EntitlementBalanceResponse,
    GuestUserResponse,
    InspirationUploadResponse,
    PhotoSetResponse,
    ProjectListResponse,
    PortraitProjectResponse,
    PortraitOrderResponse,
    PreviewRetryRequest,
    PreviewRetryResponse,
    ReferenceUploadResponse,
    SharedRecipeResponse,
    ThemeDetail,
    ThemeListResponse,
    ThemeSummary,
)
from .portrait_storage import (
    attach_legacy_session,
    check_rate_limit,
    claim_apple_transaction,
    create_guest_user,
    create_project,
    create_share_recipe,
    credit_balance,
    delete_project_data,
    delete_user_record,
    deliver_photo_set,
    get_project,
    get_asset,
    get_photo_set,
    get_portrait_order,
    get_share_recipe,
    get_theme,
    has_paid_project_entitlement,
    link_apple_identity,
    list_projects,
    list_themes,
    paid_project_order,
    record_operational_event,
    record_clean_export_request,
    reserve_project_preview_retry,
    restore_credit_spend,
    rollback_project_preview_retry,
    revoke_apple_transaction,
    save_reference_asset,
    save_inspiration,
    save_portrait_order,
    set_project_status,
    set_project_preview_ready,
    spend_credit_once,
    update_portrait_order_status,
    user_for_token,
)
from .security import sanitize_filename, validate_image_bytes


router = APIRouter(prefix="/api/v2", tags=["portrait-platform-v2"])
_CATALOG_IMAGE_RE = re.compile(r"^[A-Za-z0-9_-]+\.jpg$")
_CATALOG_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_EXCLUDED_PORTRAIT_PLATFORM_STYLES = {"id_photo"}
_PROJECT_FAILURE_MESSAGES = {
    "delivery_gate_failed": (
        "这次没能同时保留足够好的本人相似度和画面质量。"
        "请换用几张角度不同、更加清晰的照片后重试。"
    ),
    "duplicate_final_asset": (
        "生成的写真过于重复。免费预览次数已经恢复，你可以重新开始创作。"
    ),
    "exception": (
        "写真生成意外中断。免费预览次数已经恢复，你可以重新开始创作。"
    ),
}


def _is_visible_portrait_theme(theme: dict | None) -> bool:
    return bool(
        theme
        and theme.get("source_style_key") not in _EXCLUDED_PORTRAIT_PLATFORM_STYLES
    )


def _theme_catalog_contract(item: dict) -> dict:
    payload = dict(item)
    detailed = get_theme(item["theme_id"])
    blueprint = (detailed or {}).get("blueprint") or {}
    payload["presentation"] = blueprint.get("presentation", "unspecified")
    payload["preview_integrity"] = blueprint.get(
        "preview_integrity", "single_direction_study"
    )
    payload["shot_labels"] = [
        str(shot.get("label"))
        for shot in blueprint.get("shots", [])
        if isinstance(shot, dict) and shot.get("label")
    ]
    return payload


def _render_catalog_thumbnail(source: Path, destination: Path) -> None:
    """Build a small progressive JPEG cache entry without touching source art."""
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((720, 960), Image.Resampling.LANCZOS)
            image.save(
                temporary,
                format="JPEG",
                quality=82,
                optimize=True,
                progressive=True,
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@router.get("/catalog-images/{filename}")
async def catalog_image(filename: str):
    if not _CATALOG_IMAGE_RE.fullmatch(filename):
        raise HTTPException(404, "Catalog image not found")
    stem = Path(filename).stem
    sources = list(_CATALOG_TEMPLATE_DIR.glob(f"{stem}.*"))
    source = next(
        (item for item in sources if item.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        None,
    )
    if not source or not source.is_file():
        raise HTTPException(404, "Catalog image not found")

    destination = settings.data_dir / "catalog-thumbnails-v2" / filename
    if (
        not destination.is_file()
        or destination.stat().st_mtime_ns < source.stat().st_mtime_ns
    ):
        await asyncio.to_thread(_render_catalog_thumbnail, source, destination)
    return FileResponse(
        destination,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def require_user(x_user_token: str = Header(default="")) -> dict:
    user = user_for_token(x_user_token)
    if not user:
        raise HTTPException(401, "Missing or invalid X-User-Token")
    return user


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    candidate = peer
    if peer in settings.trusted_proxy_ips:
        candidate = (
            request.headers.get("CF-Connecting-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or peer
        )
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return "unknown"


def _enforce_rate(
    *, scope: str, subject: str, max_calls: int,
    window_seconds: int,
) -> None:
    allowed, retry_after = check_rate_limit(
        scope=scope,
        subject=subject,
        max_calls=max_calls,
        window_seconds=window_seconds,
        now=storage.utcnow(),
    )
    if not allowed:
        raise HTTPException(
            429,
            detail={"code": "rate_limited", "scope": scope},
            headers={"Retry-After": str(retry_after)},
        )


def _project_failure_details(
    session_id: str | None,
    *,
    unlocked: bool = False,
) -> dict[str, str | None]:
    if not session_id:
        return {"failure_code": "generation_failed", "failure_message": None}
    failed_event = next(
        (
            row for row in reversed(storage.load_generation_events(session_id))
            if row["status"] == "failed"
        ),
        None,
    )
    code = (
        str(failed_event["failure_reason"])
        if failed_event and failed_event["failure_reason"]
        else "generation_failed"
    )
    if unlocked:
        message = (
            "有一张或多张写真没有通过最终检查。你的购买权益已保留，可联系客服协助重试。"
        )
    elif code == "delivery_gate_failed":
        message = (
            "这次没能同时保留足够好的本人相似度和画面质量。免费预览次数已经恢复；"
            "请换用几张角度不同、更加清晰的照片重新创作。"
        )
    else:
        message = _PROJECT_FAILURE_MESSAGES.get(
            code,
            "这张写真没有通过最终检查。免费预览次数已经恢复，你可以重新创作。",
        )
    return {
        "failure_code": code,
        "failure_message": message,
    }


def _sync_project_generation(project: dict, user_id: str) -> dict:
    """Project v2 is authoritative; mirror terminal state from the v1 engine."""
    project = dict(project)
    session_id = project.get("legacy_session_id")
    state = queue.get_session(session_id) if session_id else None
    if state and state.hero_preview_image_id:
        project["preview_confirmed"] = any(
            item.get("image_id") == state.hero_preview_image_id
            and item.get("event") == FeedbackEvent.looks_like_me.value
            for item in state.user_feedback
        )
    if project.get("status") == "delivered":
        return project
    if not state:
        return project

    now = storage.utcnow()
    status = state.status.value
    if state.unlocked and status == "done":
        deliverable = []
        for image in state.generated_images:
            if (
                image.parent_image_id is not None
                or image.operation is not None
                or not image_or_source_passed_final_gate(state, image.image_id)
            ):
                continue
            path = queue.get_image_path(session_id, image.image_id)
            if path:
                clean_path = clean_export_path(path)
                deliverable.append({
                    "image_id": image.image_id,
                    "storage_path": str(path),
                    "clean_storage_path": (
                        str(clean_path) if clean_path.is_file() else None
                    ),
                    "mime_type": mimetypes.guess_type(str(path))[0] or "image/png",
                    "prompt_id": image.prompt_id,
                    "resemblance": image.resemblance,
                })
        if len(deliverable) == 6:
            set_quality = evaluate_portrait_set(deliverable[-6:])
            if set_quality["pass"]:
                record_operational_event(
                    "portrait_set_quality_passed",
                    project_id=project["project_id"],
                    metadata=set_quality,
                    now=now,
                )
                theme = (
                    get_theme(project["theme_id"])
                    if project.get("theme_id") else None
                )
                deliver_photo_set(
                    user_id=user_id,
                    project_id=project["project_id"],
                    title=(
                        f"{theme['title_en']} portrait set"
                        if theme else "Inspired portrait set"
                    ),
                    images=deliverable[-6:],
                    now=now,
                )
            else:
                set_quality["support_retry_shot_ids"] = (
                    queue.prepare_set_quality_retry(session_id)
                )
                record_operational_event(
                    "portrait_set_quality_failed",
                    project_id=project["project_id"],
                    metadata=set_quality,
                    now=now,
                )
                set_project_status(
                    project_id=project["project_id"], user_id=user_id,
                    status="failed", now=now,
                )
        else:
            set_project_status(
                project_id=project["project_id"], user_id=user_id,
                status="failed", now=now,
            )
    elif state.unlocked and status == "failed":
        set_project_status(
            project_id=project["project_id"], user_id=user_id,
            status="failed", now=now,
        )
    elif state.unlocked and status in {"generating", "reviewing"}:
        set_project_status(
            project_id=project["project_id"], user_id=user_id,
            status="set_generating", now=now,
        )
    elif (
        status == SessionStatus.hero_preview_ready.value
        and state.hero_preview_image_id
    ):
        set_project_preview_ready(
            project_id=project["project_id"], user_id=user_id,
            image_id=state.hero_preview_image_id, now=now,
        )
    elif status == "failed":
        set_project_status(
            project_id=project["project_id"], user_id=user_id,
            status="failed", now=now,
        )
        if restore_credit_spend(
            user_id=user_id,
            reason="hero_preview",
            reference_id=project["project_id"],
        ):
            record_operational_event(
                "preview_credit_restored",
                project_id=project["project_id"],
                metadata={"reason": "preview_generation_failed"},
                now=now,
            )
    synced = get_project(project["project_id"], user_id) or project
    synced["preview_confirmed"] = bool(project.get("preview_confirmed"))
    if synced.get("status") == "failed":
        synced.update(
            _project_failure_details(
                synced.get("legacy_session_id"),
                unlocked=bool(state and state.unlocked),
            )
        )
    return synced


@router.post("/users/guest", response_model=GuestUserResponse)
async def create_guest(
    request: Request,
    x_device_id: str = Header(default="", alias="X-Device-ID"),
):
    client_ip = _client_ip(request)
    _enforce_rate(
        scope="guest_create_10m", subject=client_ip,
        max_calls=settings.guest_create_limit_10m, window_seconds=600,
    )
    _enforce_rate(
        scope="guest_create_day", subject=client_ip,
        max_calls=settings.guest_create_limit_day, window_seconds=86_400,
    )
    device_id = x_device_id.strip()[:128]
    fingerprint = f"device:{device_id}" if device_id else f"ip:{client_ip}"
    user, token = create_guest_user(
        storage.utcnow(), preview_fingerprint=fingerprint,
    )
    return GuestUserResponse(**user, access_token=token)


@router.post("/auth/apple", response_model=AuthenticatedUserResponse)
async def authenticate_with_apple(
    req: AppleSignInRequest,
    user=Depends(require_user),
):
    try:
        claims = await asyncio.to_thread(
            apple_identity_verifier.verify, req.identity_token, req.raw_nonce,
        )
        linked_user, token, merged = link_apple_identity(
            current_user_id=user["user_id"],
            subject=str(claims["sub"]),
            email=str(claims["email"]) if claims.get("email") else None,
            display_name=req.display_name,
            now=storage.utcnow(),
        )
    except (AppleIdentityError, ValueError) as exc:
        raise HTTPException(401, str(exc)) from exc
    return AuthenticatedUserResponse(
        user_id=linked_user["user_id"],
        access_token=token,
        merged_guest_workspace=merged,
        created_at=linked_user["created_at"],
    )


@router.get("/themes", response_model=ThemeListResponse)
async def themes():
    return ThemeListResponse(themes=[
        ThemeSummary(**_theme_catalog_contract(item))
        for item in list_themes()
        if _is_visible_portrait_theme(item)
    ])


@router.get("/themes/{identifier}", response_model=ThemeDetail)
async def theme_detail(identifier: str):
    item = get_theme(identifier)
    if not _is_visible_portrait_theme(item):
        raise HTTPException(404, "Theme not found")
    blueprint = item.pop("blueprint")
    item.pop("theme_version_id", None)
    item["presentation"] = blueprint.get("presentation", "unspecified")
    item["preview_integrity"] = blueprint.get(
        "preview_integrity", "single_direction_study"
    )
    item["shot_labels"] = [
        str(shot.get("label"))
        for shot in blueprint.get("shots", [])
        if isinstance(shot, dict) and shot.get("label")
    ]
    return ThemeDetail(
        **item,
        shot_count=int(blueprint.get("set_size", 6)),
        reference_min=int(blueprint.get("reference_min", settings.min_photos)),
        reference_max=int(blueprint.get("reference_max", settings.max_photos)),
        blueprint=blueprint,
    )


@router.post("/projects", response_model=PortraitProjectResponse)
async def new_project(req: CreateProjectRequest, user=Depends(require_user)):
    if req.source == "official_theme" and req.theme_id:
        theme = get_theme(req.theme_id)
        if not _is_visible_portrait_theme(theme):
            raise HTTPException(404, "Theme not found")
    try:
        project = create_project(user["user_id"], req.model_dump(), storage.utcnow())
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return PortraitProjectResponse(**project)


@router.get("/projects", response_model=ProjectListResponse)
async def projects(user=Depends(require_user)):
    return ProjectListResponse(
        projects=[
            PortraitProjectResponse(**_sync_project_generation(item, user["user_id"]))
            for item in list_projects(user["user_id"])
        ]
    )


@router.get("/projects/{project_id}", response_model=PortraitProjectResponse)
async def project_detail(project_id: str, user=Depends(require_user)):
    project = get_project(project_id, user["user_id"])
    if not project:
        raise HTTPException(404, "Project not found")
    project = _sync_project_generation(project, user["user_id"])
    return PortraitProjectResponse(**project)


@router.get("/projects/{project_id}/hero")
async def project_hero(project_id: str, user=Depends(require_user)):
    project = get_project(project_id, user["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Preview not found")
    state = queue.get_session(project["legacy_session_id"])
    if not state or not state.hero_preview_image_id:
        raise HTTPException(404, "Preview not ready")
    path = queue.get_image_path(state.session_id, state.hero_preview_image_id)
    if not path:
        raise HTTPException(404, "Preview not found")
    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return FileResponse(path, media_type=media_type)


@router.get(
    "/projects/{project_id}/sets/{photo_set_id}",
    response_model=PhotoSetResponse,
)
async def project_photo_set(
    project_id: str, photo_set_id: str, user=Depends(require_user),
):
    item = get_photo_set(photo_set_id, project_id, user["user_id"])
    if not item or len(item["assets"]) != 6:
        raise HTTPException(404, "Delivered portrait set not found")
    return PhotoSetResponse(**item)


@router.get("/projects/{project_id}/assets/{asset_id}")
async def project_asset(
    project_id: str, asset_id: str, download: bool = False,
    user=Depends(require_user),
):
    asset = get_asset(asset_id, user["user_id"])
    if (
        not asset
        or asset.get("project_id") != project_id
        or asset.get("asset_type") != "generated_portrait"
    ):
        raise HTTPException(404, "Portrait not found")
    path = Path(asset["storage_path"])
    if not path.is_file():
        raise HTTPException(404, "Portrait not found")
    return FileResponse(
        path,
        media_type=asset["mime_type"],
        filename=f"flashshot-{asset_id}{path.suffix}" if download else None,
    )


@router.post("/projects/{project_id}/assets/{asset_id}/clean-export")
async def project_clean_asset(
    project_id: str,
    asset_id: str,
    payload: CleanExportRequest,
    user=Depends(require_user),
):
    """Return a user-requested clean copy while preserving the audit trail."""
    if payload.terms_version != CLEAN_EXPORT_TERMS_VERSION:
        raise HTTPException(409, "Clean-export terms have changed. Review them again.")
    if (
        not payload.ai_generated_acknowledged
        or not payload.redistribution_responsibility_accepted
    ):
        raise HTTPException(400, "Clean-export acknowledgement is required")

    project = get_project(project_id, user["user_id"])
    asset = get_asset(asset_id, user["user_id"])
    if (
        not project
        or project.get("status") != "delivered"
        or not asset
        or asset.get("project_id") != project_id
        or asset.get("asset_type") != "generated_portrait"
    ):
        raise HTTPException(404, "Delivered portrait not found")

    try:
        metadata = json.loads(asset.get("metadata_json") or "{}")
    except (TypeError, ValueError):
        metadata = {}
    raw_clean_path = metadata.get("clean_storage_path")
    clean_path = Path(raw_clean_path) if raw_clean_path else None
    if not clean_path or not clean_path.is_file():
        raise HTTPException(409, "A clean export is unavailable for this portrait")

    record_clean_export_request(
        user_id=user["user_id"],
        project_id=project_id,
        asset_id=asset_id,
        terms_version=payload.terms_version,
        now=storage.utcnow(),
        retention_days=CLEAN_EXPORT_RETENTION_DAYS,
    )
    return FileResponse(
        clean_path,
        media_type=asset["mime_type"],
        filename=f"flashshot-{asset_id}-clean{clean_path.suffix}",
    )


async def _delete_owned_project(project_id: str, user_id: str) -> bool:
    payload = delete_project_data(project_id, user_id)
    if not payload:
        return False
    session_id = payload.get("legacy_session_id")
    if session_id:
        await queue.delete_session(session_id)
    private_dir = settings.upload_dir / "portrait-v2" / user_id / project_id
    await asyncio.to_thread(shutil.rmtree, private_dir, True)
    for raw_path in payload.get("paths", []):
        path = Path(raw_path)
        if path.exists():
            await asyncio.to_thread(path.unlink, missing_ok=True)
    return True


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str, user=Depends(require_user)):
    if not await _delete_owned_project(project_id, user["user_id"]):
        raise HTTPException(404, "Project not found")
    return {"deleted": True, "project_id": project_id}


@router.delete("/users/me")
async def delete_workspace(user=Depends(require_user)):
    owned = list_projects(user["user_id"])
    for project in owned:
        await _delete_owned_project(project["project_id"], user["user_id"])
    delete_user_record(user["user_id"])
    return {
        "deleted": True,
        "projects_deleted": len(owned),
        "financial_records": "pseudonymized_and_retained_as_required",
    }


@router.post(
    "/projects/{project_id}/inspiration",
    response_model=InspirationUploadResponse,
)
async def upload_inspiration(
    project_id: str,
    file: UploadFile = File(...),
    rights_confirmed: bool = Form(default=False),
    private_style_reference_only: bool = Form(default=True),
    user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    if not project:
        raise HTTPException(404, "Project not found")
    if not rights_confirmed:
        raise HTTPException(400, "Confirm you may use this image as a private style reference")
    if not private_style_reference_only:
        raise HTTPException(400, "Inspiration images must remain private and may not be redistributed")
    content = await file.read()
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(400, f"File exceeds {settings.max_file_size_mb}MB")
    validate_image_bytes(content)

    safe_name = sanitize_filename(file.filename or "inspiration.jpg")
    directory = settings.upload_dir / "portrait-v2" / user["user_id"] / project_id
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(safe_name).suffix or ".jpg"
    path = directory / f"inspiration-{uuid.uuid4().hex}{suffix}"
    path.write_bytes(content)

    spec = None
    status = "pending"
    message = "灵感图已保存，服务准备好后会开始分析。"
    worker = getattr(queue, "_worker", None)
    provider = getattr(getattr(worker, "_gateway", None), "provider", None)
    if provider is not None:
        try:
            spec = analyze_with_provider(provider, str(path))
            status = "analyzed"
            message = "已提取风格信息，不会使用灵感图中人物的身份。"
        except ValueError as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc
        except Exception:
            # Keep the private upload and retry asynchronously instead of losing
            # the user's work on a transient provider failure.
            pass

    asset_id = save_inspiration(
        user_id=user["user_id"], project_id=project_id,
        storage_path=str(path),
        mime_type=file.content_type or mimetypes.guess_type(safe_name)[0] or "image/jpeg",
        spec=spec, now=storage.utcnow(),
    )
    return InspirationUploadResponse(
        asset_id=asset_id,
        project_id=project_id,
        analysis_status=status,
        inspiration_spec=spec,
        message=message,
    )


@router.post(
    "/projects/{project_id}/references",
    response_model=ReferenceUploadResponse,
)
async def upload_references(
    project_id: str,
    files: list[UploadFile] = File(...),
    gender: str = Form(...),
    face_processing_consent: bool = Form(default=False),
    adult_subject_confirmed: bool = Form(default=False),
    user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    if not project:
        raise HTTPException(404, "Project not found")
    presentation = _project_catalog_presentation(project)
    if presentation in {"male", "female"} and gender != presentation:
        raise HTTPException(
            400,
            "This shoot has a fixed wardrobe presentation. Start from its catalog page.",
        )
    if gender not in {"female", "male"}:
        raise HTTPException(400, "请选择女性或男性造型方向，以便匹配服装")
    if not face_processing_consent or not adult_subject_confirmed:
        raise HTTPException(400, "需要确认人脸处理授权及人物已成年")
    if not files:
        raise HTTPException(400, "请选择参考照片")

    validated: list[tuple[str, str, bytes]] = []
    for file in files:
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(400, f"File exceeds {settings.max_file_size_mb}MB")
        validate_image_bytes(content)
        safe_name = sanitize_filename(file.filename or "reference.jpg")
        mime = file.content_type or mimetypes.guess_type(safe_name)[0] or "image/jpeg"
        validated.append((safe_name, mime, content))

    state = None
    if project.get("legacy_session_id"):
        state = queue.get_session(project["legacy_session_id"])
    if state is None:
        theme = get_theme(project["theme_id"]) if project.get("theme_id") else None
        engine_style_key, _, _ = _project_catalog_direction(project)
        style_key = engine_style_key or (theme["source_style_key"] if theme else "cinematic")
        try:
            style = StyleKey(style_key)
        except ValueError as exc:
            raise HTTPException(409, "This theme is not generation-ready") from exc
        state = queue.create_session(style, gender)
        attach_legacy_session(
            project_id=project_id, user_id=user["user_id"],
            session_id=state.session_id, gender=gender,
            status="awaiting_references", now=storage.utcnow(),
        )

    if len(state.uploaded_photos) + len(validated) > settings.max_photos:
        raise HTTPException(400, f"Maximum {settings.max_photos} reference photos")
    await queue.record_session_consents(
        state.session_id,
        face_processing_consent=True,
        adult_subject_confirmed=True,
    )
    for safe_name, mime, content in validated:
        path = await queue.save_uploaded_photo(state.session_id, safe_name, content)
        save_reference_asset(
            user_id=user["user_id"], project_id=project_id,
            storage_path=str(path), mime_type=mime,
            quality=state.photo_quality.get(path.name, {}), now=storage.utcnow(),
        )

    quality = queue.reference_quality_gate(state)
    project_status = "ready" if quality.get("pass") else "awaiting_references"
    attach_legacy_session(
        project_id=project_id, user_id=user["user_id"],
        session_id=state.session_id, gender=gender,
        status=project_status, now=storage.utcnow(),
    )
    return ReferenceUploadResponse(
        project_id=project_id,
        legacy_session_id=state.session_id,
        reference_count=len(state.uploaded_photos),
        status=project_status,
        reference_quality=quality,
    )


@router.post(
    "/projects/{project_id}/preview",
    response_model=list[JobResponse],
)
async def start_project_preview(
    project_id: str,
    request: Request,
    x_device_id: str = Header(default="", alias="X-Device-ID"),
    user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    if not project:
        raise HTTPException(404, "Project not found")
    if not project.get("legacy_session_id"):
        raise HTTPException(400, "Upload identity references first")
    if not await queue.refresh_provider_readiness(max_age_seconds=0):
        raise HTTPException(503, "Portrait generation is temporarily unavailable")
    if credit_balance(user["user_id"]) < 1:
        raise HTTPException(402, "No preview credit available")
    state = queue.get_session(project["legacy_session_id"])
    if state is None:
        raise HTTPException(409, "Generation session could not be restored")
    if state.hero_preview_generated or state.status.value == "generating":
        raise HTTPException(409, "Preview already generated or in progress")
    client_ip = _client_ip(request)
    _enforce_rate(
        scope="preview_ip_day", subject=client_ip,
        max_calls=settings.preview_limit_ip_day, window_seconds=86_400,
    )
    _enforce_rate(
        scope="preview_user_10m", subject=user["user_id"],
        max_calls=3, window_seconds=600,
    )
    if x_device_id.strip():
        _enforce_rate(
            scope="preview_device_day", subject=x_device_id.strip()[:128],
            max_calls=3, window_seconds=86_400,
        )

    custom_template_path, custom_prompt = _project_preview_inputs(
        project, user["user_id"]
    )
    _, template_id, shot_overrides = _project_catalog_direction(project)

    try:
        debited = spend_credit_once(
            user_id=user["user_id"], reason="hero_preview",
            reference_id=project_id, now=storage.utcnow(),
        )
        if not debited:
            raise HTTPException(409, "Preview already started for this project")
        jobs = await queue.submit_hero_preview(
            state.session_id,
            custom_template_path=custom_template_path,
            custom_prompt=custom_prompt,
            template_id=template_id,
            shot_overrides=shot_overrides,
        )
        if not jobs:
            raise ValueError("No matching template for this presentation")
    except HTTPException:
        raise
    except ValueError as exc:
        restore_credit_spend(
            user_id=user["user_id"], reason="hero_preview", reference_id=project_id,
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception:
        restore_credit_spend(
            user_id=user["user_id"], reason="hero_preview", reference_id=project_id,
        )
        raise
    set_project_status(
        project_id=project_id, user_id=user["user_id"],
        status="preview_generating", now=storage.utcnow(),
    )
    return [job.to_response(position=queue.get_queue_position(job.job_id)) for job in jobs]


def _project_preview_inputs(
    project: dict, user_id: str,
) -> tuple[str | None, str | None]:
    custom_template_path = None
    custom_prompt = None
    if project["source"] == "private_inspiration":
        if not project.get("inspiration_spec") or not project.get("inspiration_asset_id"):
            raise HTTPException(409, "Inspiration analysis must finish before generation")
        asset = get_asset(project["inspiration_asset_id"], user_id)
        if not asset:
            raise HTTPException(409, "Private inspiration asset is unavailable")
        custom_template_path = asset["storage_path"]
        custom_prompt = inspiration_generation_prompt(project["inspiration_spec"])
    elif project["source"] == "shared_recipe" and project.get("inspiration_spec"):
        custom_prompt = inspiration_generation_prompt(project["inspiration_spec"])
    return custom_template_path, custom_prompt


def _project_catalog_direction(
    project: dict,
) -> tuple[str | None, str | None, list[dict] | None]:
    """Resolve the exact catalog shoot selected by the user."""
    if not project.get("theme_id"):
        return None, None, None
    theme = get_theme(project["theme_id"])
    if not theme:
        return None, None, None
    blueprint = theme.get("blueprint") or {}
    engine_style_key = blueprint.get("engine_style_key")
    template_id = blueprint.get("template_id")
    shots = blueprint.get("shots")
    return (
        str(engine_style_key) if engine_style_key else None,
        str(template_id) if template_id else None,
        shots if isinstance(shots, list) and shots else None,
    )


def _project_catalog_presentation(project: dict) -> str | None:
    if not project.get("theme_id"):
        return None
    theme = get_theme(project["theme_id"])
    presentation = ((theme or {}).get("blueprint") or {}).get("presentation")
    return str(presentation) if presentation in {"male", "female"} else None


@router.post(
    "/projects/{project_id}/preview/confirm",
    response_model=UserFeedbackResponse,
)
async def confirm_project_preview(
    project_id: str,
    user=Depends(require_user),
):
    """Record the user's explicit likeness decision for the current hero."""
    project = get_project(project_id, user["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Project not found")
    state = queue.get_session(project["legacy_session_id"])
    if not state or not state.hero_preview_image_id:
        raise HTTPException(409, "Open the finished preview before confirming it")

    record = await queue.record_user_feedback(
        state.session_id,
        state.hero_preview_image_id,
        FeedbackEvent.looks_like_me,
        reason="preview_confirmed",
        score=2,
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


@router.post(
    "/projects/{project_id}/preview/retry",
    response_model=PreviewRetryResponse,
)
async def retry_project_preview(
    project_id: str,
    req: PreviewRetryRequest,
    user=Depends(require_user),
):
    """Generate one bounded, feedback-conditioned replacement hero preview."""
    project = get_project(project_id, user["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Project not found")
    if not await queue.refresh_provider_readiness(max_age_seconds=0):
        raise HTTPException(503, "Portrait generation is temporarily unavailable")
    state = queue.get_session(project["legacy_session_id"])
    if not state or not state.hero_preview_image_id:
        raise HTTPException(409, "Open the finished preview before requesting a retry")
    if state.unlocked:
        raise HTTPException(409, "The complete portrait set has already started")

    old_hero_image_id = state.hero_preview_image_id
    now = storage.utcnow()
    try:
        retries_remaining = reserve_project_preview_retry(
            project_id=project_id,
            user_id=user["user_id"],
            now=now,
        )
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

    try:
        await queue.record_user_feedback(
            state.session_id,
            old_hero_image_id,
            FeedbackEvent.not_like_me,
            reason=f"preview_retry:{req.reason}",
            score=0,
        )
        custom_template_path, custom_prompt = _project_preview_inputs(
            project, user["user_id"]
        )
        _, template_id, shot_overrides = _project_catalog_direction(project)
        jobs = await queue.submit_hero_preview(
            state.session_id,
            custom_template_path=custom_template_path,
            custom_prompt=custom_prompt,
            replaces_image_id=old_hero_image_id,
            template_id=template_id,
            shot_overrides=shot_overrides,
        )
        if not jobs:
            raise ValueError("No matching template for this presentation")
    except Exception:
        state.hero_preview_image_id = old_hero_image_id
        state.hero_preview_generated = True
        state.status = SessionStatus.hero_preview_ready
        storage.update_session_hero_preview(
            state.session_id, old_hero_image_id, unlocked=False
        )
        storage.update_session_status(state.session_id, state.status.value)
        rollback_project_preview_retry(
            project_id=project_id,
            user_id=user["user_id"],
            hero_asset_id=old_hero_image_id,
            now=storage.utcnow(),
        )
        raise

    record_operational_event(
        "preview_identity_retry",
        project_id=project_id,
        metadata={"reason": req.reason},
        now=storage.utcnow(),
    )
    return PreviewRetryResponse(
        project_id=project_id,
        retries_remaining=retries_remaining,
        jobs=[
            job.to_response(position=queue.get_queue_position(job.job_id))
            for job in jobs
        ],
    )


def _shared_recipe_response(shared: dict) -> SharedRecipeResponse:
    theme = get_theme(shared["theme_id"]) if shared.get("theme_id") else None
    project = get_project(shared["project_id"], shared["user_id"])
    return SharedRecipeResponse(
        share_token=shared["share_token"],
        title=shared["title"],
        theme_id=shared.get("theme_id"),
        theme_slug=theme["slug"] if theme else None,
        source=project["source"] if project else shared["recipe"].get("source", "official_theme"),
        recipe=shared["recipe"],
        portrait_available=bool(shared["include_portrait"] and shared.get("hero_image_id")),
    )


@router.post(
    "/projects/{project_id}/share-recipe",
    response_model=SharedRecipeResponse,
)
async def share_project_recipe(
    project_id: str,
    req: CreateShareRecipeRequest,
    user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    if not project:
        raise HTTPException(404, "Project not found")
    state = queue.get_session(project["legacy_session_id"]) if project.get("legacy_session_id") else None
    hero_image_id = state.hero_preview_image_id if state else None
    if req.include_portrait and not hero_image_id:
        raise HTTPException(409, "A finished hero portrait is required for portrait sharing")
    theme = get_theme(project["theme_id"]) if project.get("theme_id") else None
    title = theme["title_en"] if theme else "Inspired portrait"
    inspiration_spec = project.get("inspiration_spec") or None
    if inspiration_spec:
        inspiration_spec = {
            key: value for key, value in inspiration_spec.items()
            if key not in {"safety", "forbidden_transfer"}
        }
    shared = create_share_recipe(
        user_id=user["user_id"], project=project, title=title,
        recipe={
            "source": project["source"],
            "theme_id": project.get("theme_id"),
            "inspiration_spec": inspiration_spec,
        },
        include_portrait=req.include_portrait,
        hero_image_id=hero_image_id,
        now=storage.utcnow(),
    )
    return _shared_recipe_response(shared)


@router.get("/shares/{share_token}", response_model=SharedRecipeResponse)
async def public_share_recipe(share_token: str):
    shared = get_share_recipe(share_token)
    if not shared:
        raise HTTPException(404, "Shared portrait direction not found")
    return _shared_recipe_response(shared)


@router.get("/shares/{share_token}/hero")
async def public_shared_hero(share_token: str):
    shared = get_share_recipe(share_token)
    if not shared or not shared["include_portrait"] or not shared.get("hero_image_id"):
        raise HTTPException(404, "Shared portrait not found")
    project = get_project(shared["project_id"], shared["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Shared portrait not found")
    state = queue.get_session(project["legacy_session_id"])
    if not state or not image_or_source_passed_final_gate(state, shared["hero_image_id"]):
        raise HTTPException(404, "Shared portrait not found")
    path = queue.get_image_path(project["legacy_session_id"], shared["hero_image_id"])
    if not path:
        raise HTTPException(404, "Shared portrait not found")
    return FileResponse(path, media_type=mimetypes.guess_type(str(path))[0] or "image/png")


def _tier_for_product(product_code: str) -> PricingTier:
    return (
        PricingTier.premium
        if product_code == "portrait_set_hd"
        else PricingTier.standard
    )


@router.post(
    "/projects/{project_id}/orders",
    response_model=PortraitOrderResponse,
)
async def create_project_order(
    project_id: str,
    req: CreatePortraitOrderRequest,
    user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Portrait project is not ready for checkout")
    state = queue.get_session(project["legacy_session_id"])
    if not state or not state.hero_preview_generated:
        raise HTTPException(409, "See your hero preview before unlocking the set")
    try:
        record = await asyncio.to_thread(
            PaymentService.create_order,
            state.session_id,
            _tier_for_product(req.product_code),
            {"project": project_id},
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    PaymentService.schedule_mock_confirmation(record.payment_id)
    save_portrait_order(
        order_id=record.payment_id, user_id=user["user_id"],
        project_id=project_id, product_code=req.product_code,
        amount_cents=record.amount_cents, status=record.status.value,
        now=record.created_at,
    )
    return PortraitOrderResponse(
        order_id=record.payment_id,
        project_id=project_id,
        product_code=req.product_code,
        status=record.status.value,
        amount_cents=record.amount_cents,
        checkout_url=record.checkout_url,
    )


@router.get(
    "/projects/{project_id}/orders/{order_id}",
    response_model=PortraitOrderResponse,
)
async def project_order_status(
    project_id: str, order_id: str, user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    order = get_portrait_order(order_id, user["user_id"])
    if not project or not order or order["project_id"] != project_id:
        raise HTTPException(404, "Order not found")
    payment = PaymentService.get_payment(order_id)
    status = payment.status.value if payment else order["status"]
    if status != order["status"]:
        update_portrait_order_status(
            order_id=order_id, user_id=user["user_id"], status=status,
            paid_at=storage.utcnow() if status == "paid" else None,
        )
    return PortraitOrderResponse(
        order_id=order_id,
        project_id=project_id,
        product_code=order["product_code"],
        status=status,
        amount_cents=order["amount_cents"],
        currency=order["currency"],
    )


@router.post(
    "/projects/{project_id}/apple-purchases/claim",
    response_model=ApplePurchaseClaimResponse,
)
async def claim_project_apple_purchase(
    project_id: str,
    req: ApplePurchaseClaimRequest,
    user=Depends(require_user),
):
    project = get_project(project_id, user["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Portrait project is not ready for purchase")
    state = queue.get_session(project["legacy_session_id"])
    if not state or not state.hero_preview_generated:
        raise HTTPException(409, "See your hero preview before purchasing the set")
    try:
        verified = await asyncio.to_thread(
            apple_iap_verifier.verify_transaction, req.signed_transaction,
        )
        if verified.product_id != settings.apple_iap_product_id:
            raise AppleIAPError("Unexpected StoreKit product")
        if verified.bundle_id != settings.apple_bundle_id:
            raise AppleIAPError("StoreKit bundle does not match FlashShot")
        if verified.environment != settings.apple_iap_environment:
            raise AppleIAPError("StoreKit environment mismatch")
        if verified.revoked_at is not None:
            raise AppleIAPError("StoreKit transaction has been revoked")
        order, newly_claimed = claim_apple_transaction(
            user_id=user["user_id"], project_id=project_id,
            transaction_id=verified.transaction_id,
            original_transaction_id=verified.original_transaction_id,
            product_id=verified.product_id,
            environment=verified.environment,
            bundle_id=verified.bundle_id,
            signed_payload=req.signed_transaction,
            purchased_at=verified.purchased_at,
            now=storage.utcnow(),
        )
    except (AppleIAPError, ValueError) as exc:
        record_operational_event(
            "iap_verification_failed",
            project_id=project_id,
            metadata={"error_type": type(exc).__name__},
            now=storage.utcnow(),
        )
        raise HTTPException(422, str(exc)) from exc
    queue.grant_verified_project_purchase(
        project["legacy_session_id"], order["order_id"],
    )
    return ApplePurchaseClaimResponse(
        order_id=order["order_id"],
        project_id=project_id,
        product_id=verified.product_id,
        transaction_id=verified.transaction_id,
        status=order["status"],
        newly_claimed=newly_claimed,
    )


@router.post("/apple/notifications")
async def apple_server_notification(req: AppleNotificationRequest):
    try:
        notification_type, transaction = await asyncio.to_thread(
            apple_iap_verifier.verify_notification, req.signedPayload,
        )
    except AppleIAPError as exc:
        record_operational_event(
            "iap_notification_verification_failed",
            metadata={"error_type": type(exc).__name__},
            now=storage.utcnow(),
        )
        raise HTTPException(401, str(exc)) from exc
    if transaction and (
        transaction.revoked_at is not None
        or notification_type in {"refund", "revoked"}
    ):
        revoked_at = transaction.revoked_at or storage.utcnow()
        existing = revoke_apple_transaction(
            transaction_id=transaction.transaction_id,
            revoked_at=revoked_at,
            now=storage.utcnow(),
        )
        if existing:
            record_operational_event(
                "iap_refund",
                project_id=existing["project_id"],
                transaction_id=transaction.transaction_id,
                now=storage.utcnow(),
            )
            project = get_project(existing["project_id"], existing["user_id"])
            if project and project.get("legacy_session_id"):
                queue.revoke_verified_project_purchase(
                    project["legacy_session_id"],
                    f"app_{transaction.transaction_id}",
                )
    return {"accepted": True}


@router.post(
    "/projects/{project_id}/unlock",
    response_model=list[JobResponse],
)
async def unlock_project_set(project_id: str, user=Depends(require_user)):
    project = get_project(project_id, user["user_id"])
    if not project or not project.get("legacy_session_id"):
        raise HTTPException(404, "Project not found")
    if not await queue.refresh_provider_readiness(max_age_seconds=0):
        raise HTTPException(503, "Portrait generation is temporarily unavailable")
    state = queue.get_session(project["legacy_session_id"])
    if not state:
        raise HTTPException(409, "Generation session could not be restored")
    order = paid_project_order(user["user_id"], project_id)
    if not order or not has_paid_project_entitlement(user["user_id"], project_id):
        raise HTTPException(402, "A verified purchase is required to unlock this set")
    queue.grant_verified_project_purchase(
        project["legacy_session_id"], order["order_id"],
    )
    custom_template_path = None
    custom_style_prompt = None
    if project["source"] == "private_inspiration":
        if not project.get("inspiration_spec") or not project.get("inspiration_asset_id"):
            raise HTTPException(409, "Inspiration analysis must finish before generation")
        asset = get_asset(project["inspiration_asset_id"], user["user_id"])
        if not asset:
            raise HTTPException(409, "Private inspiration asset is unavailable")
        custom_template_path = asset["storage_path"]
        custom_style_prompt = inspiration_generation_prompt(
            project["inspiration_spec"], hero_only=False,
        )
    elif project["source"] == "shared_recipe" and project.get("inspiration_spec"):
        custom_style_prompt = inspiration_generation_prompt(
            project["inspiration_spec"], hero_only=False,
        )
    _, template_id, shot_overrides = _project_catalog_direction(project)
    try:
        jobs = await queue.submit_unlock(
            state.session_id,
            custom_template_path=custom_template_path,
            custom_style_prompt=custom_style_prompt,
            template_id=template_id,
            shot_overrides=shot_overrides,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not jobs:
        raise HTTPException(400, "No remaining shots are available")
    set_project_status(
        project_id=project_id, user_id=user["user_id"],
        status="set_generating", now=storage.utcnow(),
    )
    return [job.to_response(position=queue.get_queue_position(job.job_id)) for job in jobs]


@router.get("/entitlements/balance", response_model=EntitlementBalanceResponse)
async def entitlements(user=Depends(require_user)):
    return EntitlementBalanceResponse(
        user_id=user["user_id"],
        credit_balance=credit_balance(user["user_id"]),
    )
