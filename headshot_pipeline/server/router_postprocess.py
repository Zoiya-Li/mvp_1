"""Post-processing endpoints: ID photo crop, background replacement, upscale.

All endpoints are ownership-gated. ``image_id`` lookups are validated with
``safe_id`` so a crafted id cannot traverse the output dir. Tier-gating is
enforced per feature (id_photo / bg_replace / hd_download). The ONNX upscaler
is serialized with a module-level semaphore so concurrent HD requests don't
OOM the model loader.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .config import settings
from .delivery_label import copy_with_ai_metadata
from .delivery_policy import (
    find_registered_image,
    image_passed_final_gate,
    image_or_source_passed_final_gate,
)
from .image_gateway import build_provider_invocation_metadata
from .job_queue import append_final_render_invocation, build_ai_label_check
from .models import (
    BgColor,
    GeneratedImage,
    IDPhotoSpec,
    PostProcessBgRequest,
    PostProcessCombinedRequest,
    PostProcessCropRequest,
    PostProcessResponse,
)
from . import storage
from .postprocess import PostProcessService
from .payment import check_tier_permission
from .security import require_owner, safe_id, is_within
from .upscaler import upscale_image

router = APIRouter(prefix="/api", tags=["postprocess"])

# The upscaler loads a ~100MB ONNX model and allocates GPU/CPU buffers per run.
# Serialize HD upscales so two simultaneous requests don't OOM the host.
_UPSCALE_SEM = asyncio.Semaphore(1)


def _build_postprocess_delivery_gate_check(state, source_image_id: str) -> dict:
    """Record final-QA ancestry for a derived post-process deliverable."""
    deliverable_ancestor_id: str | None = None
    current_id = source_image_id
    seen: set[str] = set()
    for _ in range(12):
        if current_id in seen:
            break
        seen.add(current_id)
        img = find_registered_image(state, current_id)
        if img is None:
            break
        if image_passed_final_gate(img):
            deliverable_ancestor_id = current_id
            break
        current_id = img.parent_image_id or img.revised_image_id or ""
        if not current_id:
            break

    passed = bool(
        deliverable_ancestor_id
        and image_or_source_passed_final_gate(state, source_image_id)
    )
    return {
        "pass": passed,
        "status": "pass" if passed else "fail",
        "source_image_id": source_image_id,
        "deliverable_ancestor_image_id": deliverable_ancestor_id,
        "inherited_from_source": deliverable_ancestor_id != source_image_id,
        "issues": [] if passed else ["source_not_deliverable"],
    }


def _get_image_path(session_id: str, image_id: str, state) -> Path:
    """Resolve image path from session output dir, rejecting traversal.

    ``image_id`` is validated to ``[A-Za-z0-9_-]`` only.
    """
    safe = safe_id(image_id, label="image_id")
    if not state.output_dir:
        raise HTTPException(404, "Session not found")
    if find_registered_image(state, safe) is None:
        raise HTTPException(404, f"Image {image_id} not found in session")
    if not image_or_source_passed_final_gate(state, safe):
        raise HTTPException(
            409,
            "Source image has not passed final QA and cannot be delivered",
        )
    path = state.output_dir / f"{safe}.png"
    if not is_within(state.output_dir, path) or not path.exists():
        raise HTTPException(404, f"Image {image_id} not found")
    return path


def _save_and_register(
    state,
    session_id: str,
    original_image_id: str,
    processed_path: Path,
    operation: str,
    prompt_id: str,
    provider_invocations: list[dict] | None = None,
) -> PostProcessResponse:
    """Save processed image with a new ID and register it in the session."""
    image_id = f"pp_{uuid.uuid4().hex[:8]}"
    dest = state.output_dir / f"{image_id}.png"

    # Copy with AI-content metadata so all delivered derivatives preserve the
    # output labeling contract.
    ai_label = copy_with_ai_metadata(
        processed_path,
        dest,
        operation=operation,
        source="flashshot_postprocess",
    )
    # Clean up temp file if different from dest
    if processed_path != dest:
        processed_path.unlink(missing_ok=True)

    delivery_gate = _build_postprocess_delivery_gate_check(state, original_image_id)
    ai_label_check = build_ai_label_check(ai_label)
    resemblance = {
        "final_evaluate": {
            "status": (
                "pass"
                if delivery_gate.get("pass") and ai_label_check.get("pass")
                else "fail"
            ),
            "delivery_gate": delivery_gate,
            "ai_label_check": ai_label_check,
            "final_render": {
                "pass": True,
                "status": "pass",
                "operation": "FINAL_RENDER",
                "postprocess_operation": operation,
                "final_asset_id": image_id,
            },
        },
        "final_asset": {
            "image_id": image_id,
            **ai_label,
            "ai_label_operation": ai_label.get("operation"),
            "operation": operation,
        }
    }
    if provider_invocations:
        for invocation in provider_invocations:
            if not isinstance(invocation, dict):
                continue
            if not invocation.get("final_asset_id"):
                invocation["final_asset_id"] = image_id
        resemblance["provider_invocations"] = provider_invocations
    append_final_render_invocation(resemblance, image_id, latency_ms=0)

    img = GeneratedImage(
        image_id=image_id,
        url=f"/api/sessions/{session_id}/images/{image_id}",
        prompt_id=prompt_id,
        turn=1,
        parent_image_id=original_image_id,
        operation=operation,
        resemblance=resemblance,
        created_at=storage.utcnow(),
    )
    state.generated_images.append(img)
    state.processed_images[image_id] = original_image_id
    # Persist the variant so it survives a backend restart alongside its parent.
    try:
        storage.save_generated_image(
            image_id=image_id,
            session_id=session_id,
            prompt_id=prompt_id,
            turn=1,
            revised_image_id=None,
            parent_image_id=original_image_id,
            operation=operation,
            resemblance=img.resemblance,
            created_at=img.created_at,
        )
    except Exception:
        pass

    return PostProcessResponse(
        original_image_id=original_image_id,
        processed_image_id=image_id,
        url=img.url,
        operation=operation,
    )


@router.post("/sessions/{session_id}/crop", response_model=PostProcessResponse)
async def crop_id_photo(
    session_id: str, req: PostProcessCropRequest, state=Depends(require_owner)
):
    """Crop an image to standard ID photo dimensions."""
    if not check_tier_permission(state, "id_photo"):
        raise HTTPException(403, "证件照裁切需要标准版或更高版本")
    image_path = _get_image_path(session_id, req.image_id, state)

    service = PostProcessService(output_dir=settings.output_dir / session_id)
    result_path = await asyncio.to_thread(service.crop_id_photo, image_path, req.spec)

    return _save_and_register(
        state, session_id, req.image_id, result_path,
        operation=f"crop_{req.spec.value}",
        prompt_id=f"crop_{req.spec.value}",
    )


@router.post("/sessions/{session_id}/background", response_model=PostProcessResponse)
async def replace_background(
    session_id: str, req: PostProcessBgRequest, state=Depends(require_owner)
):
    """Replace background with a solid color or gradient."""
    if not check_tier_permission(state, "bg_replace"):
        raise HTTPException(403, "换背景需要标准版或更高版本")
    image_path = _get_image_path(session_id, req.image_id, state)

    service = PostProcessService(output_dir=settings.output_dir / session_id)
    result_path = await asyncio.to_thread(
        service.replace_background, image_path, req.color.value
    )

    return _save_and_register(
        state, session_id, req.image_id, result_path,
        operation=f"bg_{req.color.value}",
        prompt_id=f"bg_{req.color.value}",
    )


@router.post(
    "/sessions/{session_id}/crop-background", response_model=PostProcessResponse
)
async def crop_and_replace_bg(
    session_id: str, req: PostProcessCombinedRequest, state=Depends(require_owner)
):
    """Crop to ID photo dimensions AND replace background in one step."""
    if not check_tier_permission(state, "id_photo") or not check_tier_permission(state, "bg_replace"):
        raise HTTPException(403, "证件照裁切+换背景需要标准版或更高版本")
    image_path = _get_image_path(session_id, req.image_id, state)

    service = PostProcessService(output_dir=settings.output_dir / session_id)
    result_path = await asyncio.to_thread(
        service.crop_and_replace_bg, image_path, req.spec, req.color.value
    )

    return _save_and_register(
        state, session_id, req.image_id, result_path,
        operation=f"cropbg_{req.spec.value}_{req.color.value}",
        prompt_id=f"cropbg_{req.spec.value}_{req.color.value}",
    )


# ── Upscale ────────────────────────────────────────────

class UpscaleRequest(BaseModel):
    image_id: str


@router.post("/sessions/{session_id}/upscale", response_model=PostProcessResponse)
async def upscale_hd(
    session_id: str, req: UpscaleRequest, state=Depends(require_owner)
):
    """Upscale an image to 2x resolution (premium feature)."""
    if not check_tier_permission(state, "hd_download"):
        raise HTTPException(403, "高清下载需要高级版")
    image_path = _get_image_path(session_id, req.image_id, state)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Serialize: one upscale at a time to bound memory.
    async with _UPSCALE_SEM:
        _, method = await asyncio.to_thread(upscale_image, image_path, tmp_path, 2)

    # Record the real method so the gallery/audit trail is honest about whether
    # HD was true super-resolution (realesrgan_x2) or interpolated fallback
    # (lanczos_x2). Both pass tier-gating; the marker is for provenance.
    op = "upscale_x2" if method == "realesrgan_x2" else "upscale_x2_lanczos"
    invocation = build_provider_invocation_metadata(
        invocation_id=f"upscale_{uuid.uuid4().hex[:10]}",
        operation="UPSCALE",
        prompt_version=None,
        reference_ids=[],
        parent_candidate_id=req.image_id,
        result_status="success",
    )
    invocation["upscale_method"] = method
    return _save_and_register(
        state, session_id, req.image_id, tmp_path,
        operation=op,
        prompt_id=op,
        provider_invocations=[invocation],
    )
