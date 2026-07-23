"""In-memory job queue with single-worker processing.

All blocking Gemini API calls run via asyncio.to_thread() so the FastAPI event
loop stays responsive. Exactly one job runs at a time (the generation backend
serializes anyway — one headshot generation + its judge loop at a time).
"""

from __future__ import annotations

import asyncio
import functools
import sys

# Force unbuffered stdout for all print() calls
print = functools.partial(print, flush=True)
import json
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from . import storage
from .config import settings
from .delivery_label import clean_export_path, copy_with_ai_metadata
from .delivery_policy import (
    find_registered_image,
    image_passed_final_gate,
    image_or_source_passed_final_gate,
)
from .gemini_worker import GeminiWorker
from .evaluation import EvaluationService, classify_selected_failure
from .image_gateway import build_provider_invocation_metadata
from .input_quality import (
    assess_reference_diversity,
    assess_reference_identity_consistency,
    assess_reference_photo,
    order_reference_paths_by_pose,
    summarize_reference_set,
)
from .learning import LearningLayer
from .models import (
    FeedbackEvent,
    GeneratedImage,
    Job,
    JobStatus,
    JobType,
    PaymentStatus,
    PricingTier,
    SessionConsents,
    SessionState,
    SessionStatus,
    StyleKey,
    TIER_LIMITS,
)
from .payment import PaymentService
from .security import generate_token, safe_id
from .shot_planner import (
    build_recovery_shot_spec,
    build_style_shot_plan,
    compose_recovery_shot_prompt,
)


DEFAULT_DELIVERY_GATE_ERROR = (
    "没有写真通过最终质量检查；请上传更清晰的参考照片，或尝试其他主题。"
)
FINAL_DUPLICATE_HAMMING_THRESHOLD = 4
MAX_AUTOMATIC_FULL_SET_RETRIES = 1

_TRANSIENT_GENERATION_ERROR_MARKERS = (
    "http 429",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "api error 429",
    "api error 500",
    "api error 502",
    "api error 503",
    "api error 504",
    "request failed",
    "transport error",
    "timed out",
    "timeout",
    "could not download generated image",
)

_PERMANENT_PROVIDER_ERROR_MARKERS = (
    "account balance is insufficient",
    "insufficient balance",
    "insufficient credit",
    "http 400",
    "http 401",
    "http 402",
    "http 403",
    "api error 400",
    "api error 401",
    "api error 402",
    "api error 403",
    "api key is not set",
    "api_key is not set",
)

REVISION_DENY_KEYWORDS = (
    "regenerate",
    "different result",
    "different person",
    "another person",
    "new person",
    "change identity",
    "change face",
    "change pose",
    "change body",
    "change outfit",
    "change clothes",
    "swap background",
    "replace background",
    "new background",
    "different background",
    "change style",
    "full body",
    "couple",
    "group",
    "child",
    "kid",
    "teen",
    "celebrity",
    "star",
    "video",
    "voice",
    "换人",
    "重新生成",
    "重生成",
    "换背景",
    "换衣服",
    "换风格",
    "换姿势",
    "全身",
    "多人",
    "合照",
    "情侣",
    "儿童",
    "小孩",
    "明星",
    "视频",
    "声音",
)

REVISION_ALLOW_KEYWORDS = (
    "natural",
    "realistic",
    "sharper",
    "clarity",
    "detail",
    "relaxed",
    "expression",
    "smile",
    "skin texture",
    "smoothing",
    "lighting",
    "brighter",
    "exposure",
    "artifact",
    "cleanup",
    "clean up",
    "color",
    "contrast",
    "像本人",
    "更像",
    "自然",
    "真实",
    "清晰",
    "锐",
    "细节",
    "表情",
    "微笑",
    "皮肤",
    "磨皮",
    "光线",
    "亮",
    "曝光",
    "瑕疵",
    "修复",
    "清理",
    "颜色",
    "对比度",
)


_PATH_METADATA_KEYS = {
    "path",
    "filepath",
    "file_path",
    "image_path",
    "candidate_path",
    "generated_path",
    "output_path",
    "template_path",
    "photo_paths",
    "reference_photo_paths",
}


def constrain_revision_instruction(instruction: str) -> str:
    """Convert a user revision into a bounded LOCAL_EDIT instruction."""
    normalized = " ".join(instruction.strip().split())
    if len(normalized) < 2:
        raise ValueError("Revision request is too short")
    lowered = normalized.lower()
    for keyword in REVISION_DENY_KEYWORDS:
        if keyword in lowered:
            raise ValueError(
                "Only local retouching revisions are supported in this MVP"
            )
    if not any(keyword in lowered for keyword in REVISION_ALLOW_KEYWORDS):
        raise ValueError(
            "Revision must be a local retouch request, such as clarity, "
            "lighting, expression, skin texture, or artifact cleanup"
        )
    return (
        "LOCAL_EDIT only. Apply this bounded retouch request: "
        f"{normalized}. Preserve identity, age, face shape, expression intent, "
        "pose, camera angle, clothing, background, lighting style, framing, "
        "and overall composition. Do not regenerate a new image or new person."
    )


def _bump_generation_metric(session: SessionState, key: str, amount: int = 1) -> None:
    """Record task-level generation funnel metrics outside the image gallery."""
    if not isinstance(getattr(session, "pipeline_metrics", None), dict):
        session.pipeline_metrics = {}
    current = session.pipeline_metrics.get(key, 0)
    try:
        session.pipeline_metrics[key] = int(current or 0) + amount
    except Exception:
        session.pipeline_metrics[key] = amount


def _record_generation_failure(session: SessionState, reason: str) -> None:
    _bump_generation_metric(session, "generation_failures")
    if not isinstance(getattr(session, "pipeline_metrics", None), dict):
        session.pipeline_metrics = {}
    reasons = session.pipeline_metrics.setdefault("failed_generation_reasons", {})
    if not isinstance(reasons, dict):
        reasons = {}
        session.pipeline_metrics["failed_generation_reasons"] = reasons
    reasons[reason] = int(reasons.get(reason, 0) or 0) + 1


def _is_transient_generation_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_GENERATION_ERROR_MARKERS)


def _is_permanent_provider_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_PROVIDER_ERROR_MARKERS)


def _ensure_failed_generation_metric_defaults(metrics: dict) -> None:
    metrics.setdefault("automatic_full_set_retries", 0)
    metrics.setdefault("automatic_full_set_retry_successes", 0)
    metrics.setdefault("failed_provider_invocations", 0)
    metrics.setdefault("failed_create_from_reference_invocations", 0)
    metrics.setdefault("failed_operation_counts", {})
    metrics.setdefault("failed_estimated_cost", 0.0)
    metrics.setdefault("failed_candidates_generated", 0)
    metrics.setdefault("failed_identity_repairs", 0)
    metrics.setdefault("failed_local_edits", 0)
    metrics.setdefault("failed_regenerations", 0)
    metrics.setdefault("failed_initial_identity_candidates", 0)
    metrics.setdefault("failed_initial_identity_passes", 0)
    metrics.setdefault("failed_latency_values", [])
    metrics.setdefault("shot_metrics", {})


def _shot_id_from_spec(shot_spec: Any) -> str:
    if isinstance(shot_spec, dict):
        shot_id = shot_spec.get("shot_id") or shot_spec.get("id")
        if shot_id:
            return str(shot_id)
    return "unknown"


def _shot_spec_from_event_row(row: Any) -> dict:
    raw = None
    try:
        raw = row["shot_spec_json"]
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _shot_metric_item(metrics: dict, shot_id: str) -> dict:
    shot_metrics = metrics.setdefault("shot_metrics", {})
    if not isinstance(shot_metrics, dict):
        shot_metrics = {}
        metrics["shot_metrics"] = shot_metrics
    item = shot_metrics.setdefault(shot_id, {})
    if not isinstance(item, dict):
        item = {}
        shot_metrics[shot_id] = item
    item.setdefault("attempts", 0)
    item.setdefault("completed", 0)
    item.setdefault("failed", 0)
    item.setdefault("deliverable_count", 0)
    item.setdefault("failure_reasons", {})
    item.setdefault("provider_invocations", 0)
    item.setdefault("estimated_cost", 0.0)
    item.setdefault("candidates_generated", 0)
    item.setdefault("identity_first_pass_candidates", 0)
    item.setdefault("identity_first_passes", 0)
    item.setdefault("identity_repairs", 0)
    item.setdefault("local_edits", 0)
    item.setdefault("regenerations", 0)
    return item


def _initial_identity_stats(metadata: dict | None) -> tuple[int, int]:
    if not metadata:
        return 0, 0
    candidates = metadata.get("candidates") or []
    candidate_count = 0
    pass_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("regenerated_from_candidate_id"):
            continue
        candidate_count += 1
        gate = candidate.get("gate_status") or {}
        if isinstance(gate, dict) and gate.get("identity_pass"):
            pass_count += 1
    return candidate_count, pass_count


def _add_metadata_to_shot_metric(item: dict, metadata: dict | None) -> None:
    if not metadata:
        return
    item["candidates_generated"] += len(metadata.get("candidates") or [])
    identity_candidates, identity_passes = _initial_identity_stats(metadata)
    item["identity_first_pass_candidates"] += identity_candidates
    item["identity_first_passes"] += identity_passes
    budget = metadata.get("budget") or {}
    item["identity_repairs"] += int(budget.get("identity_repairs_used") or 0)
    item["local_edits"] += int(budget.get("local_edits_used") or 0)
    item["regenerations"] += int(budget.get("regenerations_used") or 0)
    for inv in metadata.get("provider_invocations") or []:
        if not isinstance(inv, dict):
            continue
        item["provider_invocations"] += 1
        try:
            cost_value = inv.get("cost")
            if cost_value is None:
                cost_value = inv.get("estimated_cost")
            item["estimated_cost"] += float(cost_value or 0.0)
        except Exception:
            pass


def _record_shot_attempt(session: SessionState, job: Job) -> None:
    if not isinstance(getattr(session, "pipeline_metrics", None), dict):
        session.pipeline_metrics = {}
    _ensure_failed_generation_metric_defaults(session.pipeline_metrics)
    item = _shot_metric_item(session.pipeline_metrics, _shot_id_from_spec(job.shot_spec))
    item["attempts"] += 1


def _record_shot_completion(
    session: SessionState,
    job: Job,
    metadata: dict | None,
    result_image_id: str | None,
) -> None:
    if not isinstance(getattr(session, "pipeline_metrics", None), dict):
        session.pipeline_metrics = {}
    _ensure_failed_generation_metric_defaults(session.pipeline_metrics)
    item = _shot_metric_item(session.pipeline_metrics, _shot_id_from_spec(job.shot_spec))
    item["completed"] += 1
    if result_image_id and generation_passed_delivery_gate(metadata):
        item["deliverable_count"] += 1
    _add_metadata_to_shot_metric(item, _sanitize_generation_metadata(metadata))
    item["estimated_cost"] = round(item["estimated_cost"], 4)


def _record_shot_failure(
    session: SessionState,
    job: Job,
    reason: str,
    metadata: dict | None = None,
) -> None:
    if not isinstance(getattr(session, "pipeline_metrics", None), dict):
        session.pipeline_metrics = {}
    _ensure_failed_generation_metric_defaults(session.pipeline_metrics)
    item = _shot_metric_item(session.pipeline_metrics, _shot_id_from_spec(job.shot_spec))
    item["failed"] += 1
    reasons = item.setdefault("failure_reasons", {})
    reasons[reason] = int(reasons.get(reason, 0) or 0) + 1
    _add_metadata_to_shot_metric(item, _sanitize_generation_metadata(metadata))
    item["estimated_cost"] = round(item["estimated_cost"], 4)


def _record_failed_generation_metadata(
    session: SessionState,
    metadata: dict | None,
) -> None:
    if not metadata:
        return
    if not isinstance(getattr(session, "pipeline_metrics", None), dict):
        session.pipeline_metrics = {}
    _ensure_failed_generation_metric_defaults(session.pipeline_metrics)
    _add_failed_generation_metadata(
        session.pipeline_metrics,
        _sanitize_generation_metadata(metadata),
    )
    session.pipeline_metrics["failed_estimated_cost"] = round(
        session.pipeline_metrics["failed_estimated_cost"],
        4,
    )


def _public_path_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_public_path_value(item) for item in value]
    if isinstance(value, tuple):
        return [_public_path_value(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str):
        return Path(value).name
    return value


def _sanitize_generation_metadata(value: Any, key: str | None = None) -> Any:
    if key:
        normalized = key.lower()
        if normalized in _PATH_METADATA_KEYS or normalized.endswith("_path"):
            return _public_path_value(value)
        if normalized.endswith("_paths"):
            return _public_path_value(value)
    if isinstance(value, dict):
        return {
            str(k): _sanitize_generation_metadata(v, str(k))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_generation_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_generation_metadata(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _event_metadata_from_row(row: Any) -> dict:
    raw = None
    try:
        raw = row["metadata_json"]
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _add_failed_generation_metadata(metrics: dict, metadata: dict) -> None:
    metrics["failed_candidates_generated"] += len(metadata.get("candidates") or [])
    identity_candidates, identity_passes = _initial_identity_stats(metadata)
    metrics["failed_initial_identity_candidates"] += identity_candidates
    metrics["failed_initial_identity_passes"] += identity_passes
    budget = metadata.get("budget") or {}
    metrics["failed_identity_repairs"] += int(budget.get("identity_repairs_used") or 0)
    metrics["failed_local_edits"] += int(budget.get("local_edits_used") or 0)
    metrics["failed_regenerations"] += int(budget.get("regenerations_used") or 0)

    for inv in metadata.get("provider_invocations") or []:
        if not isinstance(inv, dict):
            continue
        metrics["failed_provider_invocations"] += 1
        operation = str(inv.get("operation") or "UNKNOWN")
        op_counts = metrics["failed_operation_counts"]
        op_counts[operation] = op_counts.get(operation, 0) + 1
        if operation == "CREATE_FROM_REFERENCES":
            metrics["failed_create_from_reference_invocations"] += 1
        try:
            cost_value = inv.get("cost")
            if cost_value is None:
                cost_value = inv.get("estimated_cost")
            metrics["failed_estimated_cost"] += float(cost_value or 0.0)
        except Exception:
            pass
        latency = inv.get("latency_ms")
        if isinstance(latency, int):
            metrics["failed_latency_values"].append(latency)


def _generation_metrics_from_event_rows(rows: list[Any]) -> dict:
    metrics: dict[str, Any] = {
        "generation_attempts": 0,
        "generation_failures": 0,
        "failed_generation_reasons": {},
    }
    _ensure_failed_generation_metric_defaults(metrics)
    for row in rows:
        try:
            status = row["status"]
        except Exception:
            continue
        shot_id = _shot_id_from_spec(_shot_spec_from_event_row(row))
        shot_item = _shot_metric_item(metrics, shot_id)
        metadata = _event_metadata_from_row(row)
        automatic_retry = metadata.get("automatic_retry") or {}
        if isinstance(automatic_retry, dict) and automatic_retry.get("count"):
            metrics["automatic_full_set_retries"] += 1
            if status == JobStatus.completed.value:
                metrics["automatic_full_set_retry_successes"] += 1
        metrics["generation_attempts"] += 1
        shot_item["attempts"] += 1
        if status == JobStatus.completed.value:
            shot_item["completed"] += 1
            try:
                if row["result_image_id"]:
                    shot_item["deliverable_count"] += 1
            except Exception:
                pass
            _add_metadata_to_shot_metric(shot_item, metadata)
            shot_item["estimated_cost"] = round(shot_item["estimated_cost"], 4)
            continue
        if status != JobStatus.failed.value:
            continue
        metrics["generation_failures"] += 1
        reason = row["failure_reason"] or "unknown"
        reasons = metrics["failed_generation_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        shot_item["failed"] += 1
        shot_reasons = shot_item.setdefault("failure_reasons", {})
        shot_reasons[reason] = int(shot_reasons.get(reason, 0) or 0) + 1
        _add_failed_generation_metadata(metrics, metadata)
        _add_metadata_to_shot_metric(shot_item, metadata)
        shot_item["estimated_cost"] = round(shot_item["estimated_cost"], 4)
    metrics["failed_estimated_cost"] = round(metrics["failed_estimated_cost"], 4)
    return metrics


def _save_generation_event(
    job: Job,
    status: JobStatus,
    failure_reason: str | None = None,
    error: str | None = None,
    result_image_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    try:
        event_metadata = _sanitize_generation_metadata(metadata) or {}
        if job.automatic_retry_count:
            event_metadata["automatic_retry"] = {
                "count": job.automatic_retry_count,
                "reason": job.automatic_retry_reason,
            }
        storage.save_generation_event(
            event_id=job.job_id,
            session_id=job.session_id,
            job_id=job.job_id,
            prompt_id=job.prompt_id,
            shot_spec=job.shot_spec,
            metadata=event_metadata,
            status=status.value,
            failure_reason=failure_reason,
            error=error,
            result_image_id=result_image_id,
            created_at=job.created_at,
            completed_at=storage.utcnow() if status != JobStatus.processing else None,
        )
    except Exception as exc:
        print(f"⚠ Could not persist generation event ({exc})")


def reference_slot_filename(filename: str, slot_index: int) -> str:
    """Prefix uploaded reference photos with their Identity Pack slot order."""
    clean = Path(filename).name
    if re.match(r"^ref\d{2}_", clean):
        return clean
    return f"ref{slot_index:02d}_{clean}"


def _session_media_dirs(
    session_id: str,
    state: SessionState | None = None,
) -> list[Path]:
    """Return upload/output dirs for cleanup, even if state was not hydrated."""
    dirs: list[Path] = []
    if state is not None:
        if state.upload_dir:
            dirs.append(state.upload_dir)
        if state.output_dir:
            dirs.append(state.output_dir)
    try:
        safe_session_id = safe_id(session_id, label="session_id")
    except Exception:
        return dirs
    for root in (settings.upload_dir, settings.output_dir):
        if root is None:
            continue
        path = root / safe_session_id
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                continue
        except Exception:
            continue
        if path not in dirs:
            dirs.append(path)
    return dirs


def _delete_session_intermediate_outputs(session_id: str) -> int:
    """Remove provider artifacts written beside per-session output folders.

    Providers save candidate stages in the global output root before the queue
    copies the accepted image into ``output/<session_id>/``. Candidate titles
    always begin with the validated session id, so an exact ``<id>_`` prefix
    lets us remove only this job's undelivered pixels.
    """
    try:
        safe_session_id = safe_id(session_id, label="session_id")
    except Exception:
        return 0
    root = settings.output_dir
    if root is None or not root.exists():
        return 0
    prefix = f"{safe_session_id}_"
    removed = 0
    for path in root.iterdir():
        if not path.name.startswith(prefix):
            continue
        try:
            if path.parent.resolve() != root.resolve() or not path.is_file():
                continue
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def _selected_candidate(metadata: dict | None) -> dict:
    if not isinstance(metadata, dict):
        return {}
    selected = metadata.get("selected_candidate")
    return selected if isinstance(selected, dict) else {}


def build_delivery_gate_check(metadata: dict | None) -> dict:
    """Return a structured FINAL_EVALUATE delivery-gate check."""
    selected = _selected_candidate(metadata)
    gate = selected.get("gate_status")
    issues: list[str] = []
    hard_gate_failures: list[str] = []
    if not isinstance(gate, dict):
        issues.append("missing_gate_status")
    else:
        raw_failures = gate.get("hard_gate_failures")
        if isinstance(raw_failures, list):
            hard_gate_failures = [str(item) for item in raw_failures]
        if not gate.get("hard_gates_pass"):
            issues.extend(hard_gate_failures or ["hard_gates_failed"])
    if not selected:
        issues.append("missing_selected_candidate")
    elif not selected.get("deliverable"):
        issues.append("not_deliverable")
    passed = bool(
        selected.get("deliverable")
        and isinstance(gate, dict)
        and gate.get("hard_gates_pass")
    )
    if passed:
        issues = []
    return {
        "pass": passed,
        "status": "pass" if passed else "fail",
        "selected_candidate_id": selected.get("candidate_id"),
        "hard_gate_failures": hard_gate_failures,
        "issues": issues,
    }


def attach_delivery_gate_check(metadata: dict | None) -> dict:
    """Attach the delivery hard-gate result to final_evaluate metadata."""
    check = build_delivery_gate_check(metadata)
    if isinstance(metadata, dict):
        final_eval = metadata.setdefault("final_evaluate", {})
        if isinstance(final_eval, dict):
            final_eval["delivery_gate"] = check
    return check


def generation_passed_delivery_gate(metadata: dict | None) -> bool:
    """Return whether generated metadata is allowed into the delivery gallery."""
    return bool(build_delivery_gate_check(metadata).get("pass"))


def restored_image_passed_delivery_policy(state: SessionState, img: GeneratedImage) -> bool:
    """Return whether a hydrated image has enough evidence for the gallery."""
    if image_passed_final_gate(img):
        return True
    source_id = img.parent_image_id or img.revised_image_id
    if source_id:
        return image_or_source_passed_final_gate(state, source_id)
    return False


def delivery_gate_failure_message(metadata: dict | None) -> str:
    check = build_delivery_gate_check(metadata)
    reasons = list(check.get("issues") or [])[:3]
    if not reasons:
        return DEFAULT_DELIVERY_GATE_ERROR
    return f"{DEFAULT_DELIVERY_GATE_ERROR} Final gate: {', '.join(reasons)}."


def build_ai_label_check(
    ai_label: dict[str, Any],
    clean_ai_label: dict[str, Any] | None = None,
) -> dict:
    """Return final-evaluate metadata for the generated-content label step."""
    issues: list[str] = []
    if not ai_label.get("metadata_ai_label"):
        issues.append("missing_png_ai_metadata")
    if not ai_label.get("visible_label_reserved"):
        issues.append("visible_label_not_reserved")
    if clean_ai_label is not None:
        if not clean_ai_label.get("metadata_ai_label"):
            issues.append("clean_export_missing_png_ai_metadata")
        if clean_ai_label.get("visible_ai_label"):
            issues.append("clean_export_has_visible_label")
    passed = not issues
    return {
        "pass": passed,
        "status": "pass" if passed else "fail",
        "metadata_ai_label": bool(ai_label.get("metadata_ai_label")),
        "visible_ai_label": bool(ai_label.get("visible_ai_label")),
        "visible_label_reserved": bool(ai_label.get("visible_label_reserved")),
        "clean_export_ready": bool(
            clean_ai_label
            and clean_ai_label.get("metadata_ai_label")
            and not clean_ai_label.get("visible_ai_label")
        ),
        "issues": issues,
    }


def append_final_render_invocation(
    metadata: dict | None,
    image_id: str,
    latency_ms: int,
) -> None:
    """Record deterministic delivery packaging as the FINAL_RENDER step."""
    if not isinstance(metadata, dict):
        return
    invocations = metadata.setdefault("provider_invocations", [])
    if not isinstance(invocations, list):
        metadata["provider_invocations"] = []
        invocations = metadata["provider_invocations"]

    selected = _selected_candidate(metadata)
    shot_spec = metadata.get("shot_spec")
    shot_id = shot_spec.get("shot_id") if isinstance(shot_spec, dict) else None
    invocations.append(build_provider_invocation_metadata(
        invocation_id=f"final_render_{image_id}",
        operation="FINAL_RENDER",
        prompt_version=None,
        reference_ids=[],
        parent_candidate_id=selected.get("candidate_id"),
        shot_id=shot_id,
        final_asset_id=image_id,
        latency_ms=latency_ms,
        result_status="success",
    ))


def _average_image_hash(path: str | Path, size: int = 8) -> tuple[bool, ...]:
    """Small perceptual hash for final-gallery duplicate checks."""
    from PIL import Image

    with Image.open(path) as img:
        gray = img.convert("L")
        width, height = gray.size
        # Existing delivered images may carry the visible AI label at bottom
        # right. Hash the main portrait area so the label does not dominate.
        crop_bottom = max(1, int(height * 0.90))
        gray = gray.crop((0, 0, width, crop_bottom))
        gray = gray.resize((size, size))
        pixels = list(gray.tobytes())
    avg = sum(pixels) / len(pixels)
    return tuple(pixel >= avg for pixel in pixels)


def _hamming_distance(left: tuple[bool, ...], right: tuple[bool, ...]) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def final_duplicate_check(
    candidate_path: str | Path,
    existing_paths: list[Path],
    threshold: int = FINAL_DUPLICATE_HAMMING_THRESHOLD,
) -> dict:
    """Return a final-evaluate duplicate verdict for gallery delivery."""
    result = {
        "status": "pass",
        "pass": True,
        "issues": [],
        "measurements": {
            "existing_count": len(existing_paths),
            "hamming_threshold": threshold,
        },
    }
    if not existing_paths:
        return result

    try:
        candidate_hash = _average_image_hash(candidate_path)
    except Exception as exc:
        result["status"] = "unchecked"
        result["notes"] = f"duplicate_check_unavailable: {exc}"
        return result

    closest: dict | None = None
    for path in existing_paths:
        if not path.exists():
            continue
        try:
            distance = _hamming_distance(candidate_hash, _average_image_hash(path))
        except Exception:
            continue
        item = {
            "image_id": path.stem,
            "filename": path.name,
            "hamming_distance": distance,
        }
        if closest is None or distance < closest["hamming_distance"]:
            closest = item

    if closest is not None:
        result["measurements"]["closest_match"] = closest
        if closest["hamming_distance"] <= threshold:
            result["status"] = "fail"
            result["pass"] = False
            result["issues"].append("duplicate_final_asset")
    return result


def final_duplicate_failure_message(duplicate_check: dict) -> str:
    closest = (duplicate_check.get("measurements") or {}).get("closest_match") or {}
    image_id = closest.get("image_id", "existing_image")
    return (
        "Generated portrait is too similar to an existing delivered image; "
        f"dropped duplicate candidate near {image_id}."
    )


class JobQueue:
    """Single-worker, in-memory, asyncio-based job queue.

    Concurrency model:
      - Generation runs over the OpenRouter Gemini API. The ``_process_loop``
        dequeues and runs exactly one job at a time, so generation is already
        serialized — one resemblance loop (generate → judge → revise) at a time.
      - Async route handlers (upload, submit, status) can run concurrently with
        a running job. A per-session asyncio.Lock guards the state-mutating
        submit/upload critical sections so a double-click or a reload cannot
        enqueue two generation passes for the same session or corrupt counters.
    """

    def __init__(self):
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._sessions: dict[str, SessionState] = {}
        self._jobs: dict[str, Job] = {}
        self._ws_connections: dict[str, list[WebSocket]] = {}
        self._worker_task: asyncio.Task | None = None
        self._worker: GeminiWorker | None = None
        self._worker_readiness_error: str | None = None
        self._provider_readiness_checked_at: float = 0.0
        self._provider_readiness_lock = asyncio.Lock()
        self._prompts_data: dict | None = None
        self._learning_layer: LearningLayer | None = None
        # Per-session locks, created on demand. Guards submit/upload mutations.
        self._session_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    @property
    def generation_ready(self) -> bool:
        """Whether the process can accept generation work right now."""
        if self._worker is None:
            return False
        readiness = getattr(self._worker, "provider_readiness", None)
        return bool(isinstance(readiness, dict) and readiness.get("pass"))

    @property
    def worker_readiness_error(self) -> str | None:
        return self._worker_readiness_error

    def _mark_provider_unavailable(self, error: Exception) -> None:
        """Fail closed after a permanent provider account/configuration error."""
        self._worker_readiness_error = str(error)
        self._provider_readiness_checked_at = time.monotonic()
        if self._worker is None:
            return
        previous = self._worker.provider_readiness or {}
        self._worker.provider_readiness = {
            "pass": False,
            "provider": previous.get("provider", settings.gemini_backend),
            "model": previous.get("model"),
            "error": "provider_account_or_configuration_error",
        }

    async def refresh_provider_readiness(
        self,
        *,
        max_age_seconds: float = 30.0,
    ) -> bool:
        """Refresh model access and credits without requiring a process restart."""
        now = time.monotonic()
        if (
            max_age_seconds > 0
            and self._provider_readiness_checked_at > 0
            and now - self._provider_readiness_checked_at < max_age_seconds
        ):
            return self.generation_ready
        async with self._provider_readiness_lock:
            if self._worker is None:
                try:
                    self._worker = GeminiWorker(learning_layer=self._learning_layer)
                except Exception as exc:
                    self._worker_readiness_error = str(exc)
                    self._provider_readiness_checked_at = time.monotonic()
                    self._worker = None
                    return False
            now = time.monotonic()
            if (
                max_age_seconds > 0
                and self._provider_readiness_checked_at > 0
                and now - self._provider_readiness_checked_at < max_age_seconds
            ):
                return self.generation_ready
            try:
                await asyncio.to_thread(self._worker.connect)
                self._worker_readiness_error = None
            except Exception as exc:
                self._mark_provider_unavailable(exc)
            finally:
                self._provider_readiness_checked_at = time.monotonic()
        return self.generation_ready

    # Raw generated images use img_{8hex}; post-processed variants use
    # pp_{8hex}. Intermediates ({stem}_crop_* etc.) are deleted after copy, so
    # only these two families should ever be on disk in a session output_dir.
    _IMG_FILE_RE = re.compile(r"^(img|pp)_[0-9a-fA-F]{8}\.png$")

    # ── Startup / Shutdown ────────────────────────────

    async def start(self):
        """Initialize worker and start processing loop."""
        # Persist durable state (tier/payment/session existence) across restarts.
        storage.init_db()
        try:
            from . import payment
            payment._load_from_db()
        except Exception as exc:
            print(f"⚠ Could not reload payment cache from DB ({exc})")

        self._load_prompts()
        # Initialize learning layer for feedback-driven calibration
        try:
            self._learning_layer = LearningLayer()
            cal = self._learning_layer.get_calibration()
            print(f"✓ Learning layer ready (samples={cal.sample_count})")
        except Exception as exc:
            print(f"⚠ Learning layer init failed ({exc}); running with static thresholds")
            self._learning_layer = None
        try:
            self._worker = GeminiWorker(learning_layer=self._learning_layer)
            await asyncio.to_thread(self._worker.connect)
            self._worker_readiness_error = None
            self._provider_readiness_checked_at = time.monotonic()
            readiness = self._worker.provider_readiness or {}
            print(
                "✓ Image provider ready: %s / %s"
                % (readiness.get("provider", "unknown"), readiness.get("model", "unknown"))
            )
        except Exception as exc:
            self._worker_readiness_error = str(exc)
            print(
                "⚠ Image generation provider not ready (%s); verify the API key, "
                "configured model, provider region availability, and outbound "
                "network before restarting the API." % exc
            )
            self._worker = None
        self._worker_task = asyncio.create_task(self._process_loop())

    async def stop(self):
        """Graceful shutdown."""
        if self._worker_task:
            self._worker_task.cancel()
        if self._worker:
            await asyncio.to_thread(self._worker.disconnect)

    def _load_prompts(self):
        prompts_path = Path(__file__).resolve().parent.parent / "prompts.json"
        with open(prompts_path, "r", encoding="utf-8") as f:
            self._prompts_data = json.load(f)

    @staticmethod
    def _resolve_template_path(template: dict[str, Any]) -> str | None:
        template_image = template.get("template_image")
        if not template_image:
            return None
        return str(Path(__file__).resolve().parent.parent / template_image)

    # ── Session management ────────────────────────────

    def create_session(self, style: StyleKey, gender: str) -> SessionState:
        session_id = f"s_{uuid.uuid4().hex[:8]}"
        owner_token = generate_token()
        state = SessionState(
            session_id=session_id, style=style, gender=gender,
            owner_token=owner_token,
        )
        state.upload_dir = settings.upload_dir / session_id
        state.output_dir = settings.output_dir / session_id
        state.upload_dir.mkdir(parents=True, exist_ok=True)
        state.output_dir.mkdir(parents=True, exist_ok=True)
        self._sessions[session_id] = state
        # Persist the session row so its existence/tier survive a restart.
        storage.save_session(
            session_id, owner_token, style.value, gender, state.created_at,
        )
        return state

    def get_session(self, session_id: str) -> SessionState | None:
        """Return a session, hydrating it from SQLite + disk on a cache miss.

        A restart wipes ``_sessions``. Without this, every session becomes an
        orphan: SQLite holds the correct owner_token/tier, but the live API only
        knows about in-memory state, so a correct token gets a 401 and a paid
        user loses access to their tier + gallery. Hydration reconstructs the
        scalar fields from the sessions row and the image gallery from the
        generated_images table when the backing pixel file and final-QA
        provenance are both present.
        """
        state = self._sessions.get(session_id)
        if state is not None:
            return state
        return self._hydrate_session(session_id)

    def _hydrate_session(self, session_id: str) -> SessionState | None:
        """Rebuild a SessionState from SQLite + disk after a restart."""
        row = storage.load_session_row(session_id)
        if row is None:
            return None
        try:
            style = StyleKey(row["style"])
        except Exception:
            style = StyleKey.business
        try:
            status = SessionStatus(row["status"])
        except Exception:
            status = SessionStatus.created

        state = SessionState(
            session_id=row["session_id"],
            style=style,
            gender=row["gender"],
            owner_token=row["owner_token"],
        )
        try:
            state.created_at = datetime.fromisoformat(row["created_at"])
        except Exception:
            state.created_at = storage.utcnow()
        state.tier = PricingTier(row["tier"])
        state.max_revisions = row["max_revisions"]
        state.payment_id = row["payment_id"]
        state.hero_preview_image_id = (
            row["hero_preview_image_id"]
            if "hero_preview_image_id" in row.keys()
            else None
        )
        state.hero_preview_generated = bool(state.hero_preview_image_id)
        state.unlocked = bool(
            row["unlocked"] if "unlocked" in row.keys() else False
        )
        consent_json = row["consent_json"] if "consent_json" in row.keys() else None
        if consent_json:
            try:
                state.session_consents = SessionConsents(**json.loads(consent_json))
            except Exception:
                state.session_consents = SessionConsents()
        if state.payment_id:
            payment_row = storage.load_payment_row(state.payment_id)
            if payment_row is not None:
                try:
                    state.payment_status = PaymentStatus(payment_row["status"])
                except Exception:
                    state.payment_status = None
        state.status = status
        state.upload_dir = settings.upload_dir / session_id
        state.output_dir = settings.output_dir / session_id

        # Re-list uploaded photos from disk (metadata = the filename).
        if state.upload_dir.exists():
            state.uploaded_photos = sorted(
                p for p in state.upload_dir.iterdir() if p.is_file()
            )
            self._refresh_reference_quality(state)

        # Image gallery: persisted metadata is the delivery evidence source, and
        # disk is only the pixel-existence check. Do not resurrect orphaned
        # output files into the gallery without final-QA provenance.
        metadata_rows = storage.load_generated_images(session_id)
        if state.output_dir.exists():
            disk_by_id = {
                p.stem: p
                for p in state.output_dir.iterdir()
                if self._IMG_FILE_RE.match(p.name)
            }
            for r in sorted(
                metadata_rows,
                key=lambda row: (row["created_at"], row["image_id"]),
            ):
                p = disk_by_id.get(r["image_id"])
                if p is None:
                    continue
                if not self._IMG_FILE_RE.match(p.name):
                    continue
                image_id = p.stem
                try:
                    created = datetime.fromisoformat(r["created_at"])
                except Exception:
                    created = storage.utcnow()
                resemblance = None
                if r["resemblance_json"]:
                    try:
                        resemblance = json.loads(r["resemblance_json"])
                    except Exception:
                        resemblance = None
                img = GeneratedImage(
                    image_id=image_id,
                    url=f"/api/sessions/{session_id}/images/{image_id}",
                    prompt_id=r["prompt_id"] or "recovered",
                    turn=r["turn"],
                    revised_image_id=r["revised_image_id"],
                    parent_image_id=r["parent_image_id"],
                    operation=r["operation"],
                    resemblance=resemblance,
                    created_at=created,
                )
                if not restored_image_passed_delivery_policy(state, img):
                    continue
                state.generated_images.append(img)

        state.user_feedback = [
            {
                "feedback_id": r["feedback_id"],
                "session_id": r["session_id"],
                "image_id": r["image_id"],
                "event": r["event"],
                "reason": r["reason"],
                "score": r["score"],
                "created_at": r["created_at"],
            }
            for r in storage.load_user_feedback(session_id)
        ]
        interrupted = storage.fail_interrupted_generation_events(
            session_id,
            completed_at=storage.utcnow(),
        )
        if interrupted and state.status in {
            SessionStatus.generating,
            SessionStatus.reviewing,
        }:
            state.status = SessionStatus.failed
            storage.update_session_status(session_id, state.status.value)
        state.pipeline_metrics = _generation_metrics_from_event_rows(
            storage.load_generation_events(session_id)
        )

        # Status in SQLite is only ever 'created' (it is set in-memory during
        # the lifecycle, rarely re-persisted). If images exist, the session is
        # at least reviewing — don't leave a paid user stuck on 'created'.
        if state.generated_images and state.status == SessionStatus.created:
            state.status = SessionStatus.reviewing
        # revisions_used is ephemeral; approximate from completed revision turns.
        state.revisions_used = sum(1 for im in state.generated_images if im.turn >= 2)

        self._sessions[session_id] = state
        return state

    def apply_payment_tier_upgrade(
        self,
        payment_id: str,
        amount_cents: int | None = None,
        provider_transaction_id: str | None = None,
    ):
        """The single tier-upgrade path.

        Marks the payment paid and promotes the session's tier — in memory AND
        in SQLite. Called by the verified Paddle webhook handler and (in dev
        mock mode) by the auto-confirm task. Nothing else may raise a tier.
        """
        record = PaymentService.apply_paid_webhook(
            payment_id, amount_cents, provider_transaction_id
        )
        if record is None:
            return None
        limits = TIER_LIMITS[record.tier]
        session = self.get_session(record.session_id)
        if session:
            session.tier = record.tier
            session.max_revisions = limits["max_revisions"]
            session.payment_id = record.payment_id
            session.payment_status = record.status
        # Persist regardless (audit trail + survives restart).
        storage.update_session_tier(
            record.session_id, record.tier.value,
            limits["max_revisions"], record.payment_id,
        )
        return record

    def grant_verified_project_purchase(
        self, session_id: str, order_id: str,
        tier: PricingTier = PricingTier.standard,
    ) -> None:
        """Promote a v2 project after its provider transaction was verified.

        The caller must persist and validate the provider transaction before
        invoking this method. Keeping this method provider-neutral lets the
        native StoreKit path and optional web checkout share one queue gate.
        """
        limits = TIER_LIMITS[tier]
        session = self.get_session(session_id)
        if session:
            session.tier = tier
            session.max_revisions = limits["max_revisions"]
            session.payment_id = order_id
            session.payment_status = PaymentStatus.paid
        storage.update_session_tier(
            session_id, tier.value, limits["max_revisions"], order_id,
        )

    def revoke_verified_project_purchase(
        self, session_id: str, order_id: str,
    ) -> None:
        """Revoke an unused entitlement without interrupting delivered work."""
        session = self.get_session(session_id)
        if session and session.payment_id == order_id and not session.unlocked:
            session.tier = PricingTier.free
            session.max_revisions = TIER_LIMITS[PricingTier.free]["max_revisions"]
            session.payment_status = PaymentStatus.refunded
            storage.update_session_tier(
                session_id,
                PricingTier.free.value,
                session.max_revisions,
                order_id,
            )

    def apply_payment_refund(
        self,
        payment_id: str | None = None,
        provider_transaction_id: str | None = None,
    ):
        """Mark a payment refunded and update the hot session metrics state."""
        record = PaymentService.apply_refunded_webhook(
            payment_id=payment_id,
            provider_transaction_id=provider_transaction_id,
        )
        if record is None:
            return None
        session = self.get_session(record.session_id)
        if session:
            session.payment_id = record.payment_id
            session.payment_status = record.status
        return record

    async def delete_session(self, session_id: str):
        """Delete a session's in-memory state, on-disk files (off the event
        loop), and its DB row."""
        state = self._sessions.pop(session_id, None)
        self._session_locks.pop(session_id, None)
        # Filesystem cleanup is blocking — run it off the event loop.
        for media_dir in _session_media_dirs(session_id, state):
            await asyncio.to_thread(
                shutil.rmtree, media_dir, True  # ignore_errors
            )
        await asyncio.to_thread(_delete_session_intermediate_outputs, session_id)
        if self._worker:
            self._worker.end_session(session_id)
        try:
            storage.delete_session_images(session_id)
        except Exception:
            pass
        try:
            storage.delete_session_generation_events(session_id)
        except Exception:
            pass
        try:
            storage.delete_session_feedback(session_id)
        except Exception:
            pass
        try:
            storage.delete_session_row(session_id)
        except Exception:
            pass

    async def expire_session_sources(self, session_id: str) -> None:
        """Delete identity references while retaining non-biometric metadata."""
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id) or self._hydrate_session(session_id)
            upload_dir = state.upload_dir if state else settings.upload_dir / session_id
            await asyncio.to_thread(shutil.rmtree, upload_dir, True)
            if state:
                state.uploaded_photos = []
                state.photo_quality = {}
                state.reference_quality = None
                state.status = SessionStatus.created
            storage.update_session_status(session_id, SessionStatus.created.value)

    async def expire_session_outputs(self, session_id: str) -> None:
        """Delete generated pixels while retaining minimum audit metadata."""
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id) or self._hydrate_session(session_id)
            output_dir = state.output_dir if state else settings.output_dir / session_id
            await asyncio.to_thread(shutil.rmtree, output_dir, True)
            await asyncio.to_thread(
                _delete_session_intermediate_outputs, session_id
            )
            if state:
                state.generated_images = []
                state.hero_preview_image_id = None
                state.hero_preview_generated = False
            storage.delete_session_images(session_id)
            storage.update_session_hero_preview(session_id, None, unlocked=False)

    async def record_user_feedback(
        self,
        session_id: str,
        image_id: str,
        event: FeedbackEvent,
        reason: str | None = None,
        score: int | None = None,
    ) -> dict:
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id) or self._hydrate_session(session_id)
            if state is None:
                raise KeyError(session_id)
            if find_registered_image(state, image_id) is None:
                raise FileNotFoundError(image_id)
            if (
                event in {FeedbackEvent.downloaded, FeedbackEvent.selected}
                and not image_or_source_passed_final_gate(state, image_id)
            ):
                raise PermissionError(image_id)

            created_at = storage.utcnow()
            feedback_id = f"fb_{uuid.uuid4().hex[:10]}"
            record = {
                "feedback_id": feedback_id,
                "session_id": session_id,
                "image_id": image_id,
                "event": event.value,
                "reason": reason,
                "score": score,
                "created_at": created_at.isoformat(),
            }
            await asyncio.to_thread(
                storage.save_user_feedback,
                feedback_id,
                session_id,
                image_id,
                event.value,
                reason,
                score,
                created_at,
            )
            state.user_feedback.append(record)
            # Also write identity feedback to the learning layer for threshold calibration
            if self._learning_layer is not None and event in {
                FeedbackEvent.looks_like_me,
                FeedbackEvent.not_like_me,
            }:
                try:
                    self._learning_layer.record_feedback(
                        image_id=image_id,
                        session_id=session_id,
                        event=event.value,
                        score=score,
                        reason=reason,
                    )
                    self._learning_layer.calibrate()
                except Exception as exc:
                    print(f"⚠ Learning layer feedback recording failed: {exc}")
            return record

    # ── Photo uploads ─────────────────────────────────

    async def record_session_consents(
        self,
        session_id: str,
        *,
        face_processing_consent: bool,
        adult_subject_confirmed: bool,
    ) -> dict:
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id) or self._hydrate_session(session_id)
            if state is None:
                raise KeyError(session_id)
            state.record_session_consents(
                face_processing_consent=face_processing_consent,
                adult_subject_confirmed=adult_subject_confirmed,
                consented_at=storage.utcnow(),
            )
            payload = state.session_consents.model_dump(mode="json")
            await asyncio.to_thread(
                storage.update_session_consent,
                session_id,
                payload,
            )
            return payload

    async def save_uploaded_photo(self, session_id: str, filename: str,
                                  content: bytes) -> Path:
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            slot_index = len(state.uploaded_photos) + 1
            stored_filename = reference_slot_filename(filename, slot_index)
            filepath = state.upload_dir / stored_filename
            # Disk write is blocking — run it off the event loop.
            await asyncio.to_thread(filepath.write_bytes, content)
            if filepath not in state.uploaded_photos:
                state.uploaded_photos.append(filepath)
            state.photo_quality[filepath.name] = await asyncio.to_thread(
                assess_reference_photo, filepath
            )
            identity_consistency = await asyncio.to_thread(
                assess_reference_identity_consistency, state.uploaded_photos
            )
            diversity = await asyncio.to_thread(
                assess_reference_diversity,
                state.uploaded_photos,
                settings.min_photos,
            )
            state.reference_quality = summarize_reference_set(
                state.photo_quality,
                settings.min_photos,
                identity_consistency,
                diversity,
                identity_consistency.get("pose_diversity"),
            )
            if state.reference_quality.get("pass"):
                state.uploaded_photos = order_reference_paths_by_pose(
                    state.uploaded_photos,
                    identity_consistency.get("pose_diversity"),
                )
                state.status = SessionStatus.ready
            else:
                state.status = SessionStatus.uploading
            return filepath

    def _refresh_reference_quality(self, state: SessionState) -> dict:
        state.photo_quality = {
            p.name: assess_reference_photo(p)
            for p in state.uploaded_photos
        }
        identity_consistency = assess_reference_identity_consistency(
            state.uploaded_photos
        )
        diversity = assess_reference_diversity(
            state.uploaded_photos,
            settings.min_photos,
        )
        state.reference_quality = summarize_reference_set(
            state.photo_quality,
            settings.min_photos,
            identity_consistency,
            diversity,
            identity_consistency.get("pose_diversity"),
        )
        if state.reference_quality.get("pass"):
            state.uploaded_photos = order_reference_paths_by_pose(
                state.uploaded_photos,
                identity_consistency.get("pose_diversity"),
            )
        return state.reference_quality

    def reference_quality_gate(self, state: SessionState) -> dict:
        """Return the current reference-photo gate, recomputing if needed."""
        known = set(state.photo_quality)
        current = {p.name for p in state.uploaded_photos}
        if known != current or state.reference_quality is None:
            return self._refresh_reference_quality(state)
        return state.reference_quality

    @staticmethod
    def generation_consent_gate(state: SessionState) -> None:
        """Require explicit face-processing and adult-subject consent."""
        consents = state.session_consents
        if not consents.face_processing_consent:
            raise ValueError(
                "Face-processing consent is required before generation"
            )
        if not consents.adult_subject_confirmed:
            raise ValueError(
                "Adult-subject confirmation is required before generation"
            )
        if not consents.no_training_by_default:
            raise ValueError("Training opt-out policy must be enabled")
        if not consents.cross_user_search_prohibited:
            raise ValueError("Cross-user face search must be prohibited")
        if not consents.long_term_face_library_prohibited:
            raise ValueError("Long-term face library must be prohibited")

    # ── Job submission ────────────────────────────────

    async def submit_hero_preview(
        self,
        session_id: str,
        style_override: str | None = None,
        *,
        custom_template_path: str | None = None,
        custom_prompt: str | None = None,
        replaces_image_id: str | None = None,
        template_id: str | None = None,
        shot_overrides: list[dict] | None = None,
    ) -> list[Job]:
        """Create a hero preview job: one close-up portrait for the Aha Moment.

        ``style_override`` lets multi-style bundles pick the first selected style
        for the hero preview instead of the session's default style.
        """
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            self.generation_consent_gate(state)
            gate = self.reference_quality_gate(state)
            if not gate.get("pass"):
                state.status = SessionStatus.uploading
                raise ValueError(
                    "Reference photos need replacement before generation: "
                    + "; ".join(gate.get("issues", [])[:8])
                )

            style_key = style_override if style_override else state.style.value
            style_data = self._prompts_data["styles"][style_key]
            plan = build_style_shot_plan(
                style_key,
                state.gender,
                style_data,
                hero_only=True,
                template_id=template_id,
                shot_overrides=shot_overrides,
            )
            if not plan:
                state.status = (
                    SessionStatus.ready
                    if state.uploaded_photos
                    else SessionStatus.created
                )
                return []

            state.status = SessionStatus.generating
            storage.update_session_status(session_id, state.status.value)
            jobs = []
            shot = plan[0]
            prompt = custom_prompt or shot.prompt
            template_path = custom_template_path or self._resolve_template_path(shot.template)
            shot_spec = dict(shot.shot_spec)
            if custom_template_path:
                shot_spec["style_id"] = "private_inspiration"
                shot_spec["template_id"] = "private_user_reference"
                shot_spec["template_label"] = "Private inspiration"
                shot_spec["prompt_blocks"] = dict(shot_spec["prompt_blocks"])
                shot_spec["prompt_blocks"]["style_block"] = custom_prompt or "private inspiration"
            job = Job(
                session_id=session_id,
                job_type=JobType.hero_preview,
                prompt=prompt,
                prompt_id=shot.prompt_id,
                template_path=template_path,
                shot_spec=shot_spec,
                replaces_image_id=replaces_image_id,
            )
            self._jobs[job.job_id] = job
            await self._queue.put(job)
            print(
                "  📥 Queued hero preview job %s (shot=%s), queue size=%s"
                % (
                    job.job_id,
                    shot.shot_spec.get("shot_id"),
                    self._queue.qsize(),
                )
            )
            jobs.append(job)
            return jobs

    @staticmethod
    def _restore_replaced_hero(session: SessionState, job: Job) -> bool:
        """Keep the prior preview usable when its requested replacement fails."""
        image_id = job.replaces_image_id
        if job.job_type != JobType.hero_preview or not image_id:
            return False
        original = next(
            (image for image in session.generated_images if image.image_id == image_id),
            None,
        )
        if original is None:
            return False
        original.operation = None
        session.hero_preview_image_id = image_id
        session.hero_preview_generated = True
        session.status = SessionStatus.hero_preview_ready
        storage.update_session_hero_preview(
            session.session_id, image_id, unlocked=session.unlocked,
        )
        storage.update_session_status(
            session.session_id, session.status.value,
        )
        storage.mark_generated_image_operation(
            session.session_id, image_id, None,
        )
        return True

    @staticmethod
    def _supersede_replaced_hero(session: SessionState, job: Job) -> bool:
        """Hide the prior preview from delivery after its replacement is durable."""
        image_id = job.replaces_image_id
        if job.job_type != JobType.hero_preview or not image_id:
            return False
        original = next(
            (image for image in session.generated_images if image.image_id == image_id),
            None,
        )
        if original is None:
            return False
        original.operation = "superseded_preview"
        storage.mark_generated_image_operation(
            session.session_id, image_id, "superseded_preview",
        )
        return True

    async def submit_unlock(
        self,
        session_id: str,
        *,
        custom_template_path: str | None = None,
        custom_style_prompt: str | None = None,
        template_id: str | None = None,
        shot_overrides: list[dict] | None = None,
    ) -> list[Job]:
        """Unlock the full portrait set after hero preview + payment."""
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            self.generation_consent_gate(state)

            if not state.hero_preview_generated:
                raise ValueError(
                    "Hero preview must be generated before unlocking the full set"
                )

            # Check tier: paid users can unlock; free users must pay first
            if state.tier == PricingTier.free:
                raise ValueError(
                    "Please upgrade to a paid tier to unlock the full portrait set"
                )

            style_data = self._prompts_data["styles"][state.style.value]
            plan = build_style_shot_plan(
                state.style.value,
                state.gender,
                style_data,
                template_id=template_id,
                shot_overrides=shot_overrides,
            )
            # The accepted Hero Preview is the paid set's cover image. Generate
            # only the remaining five planned compositions so the customer gets
            # six distinct finals instead of paying for a near-duplicate second
            # close-up.
            if plan:
                plan = plan[1:]
            completed_shot_ids: set[str] = set()
            for event in storage.load_generation_events(session_id):
                if event["status"] != JobStatus.completed.value:
                    continue
                try:
                    event_spec = json.loads(event["shot_spec_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if event_spec.get("shot_id"):
                    completed_shot_ids.add(str(event_spec["shot_id"]))
            active_shot_ids = {
                str((existing.shot_spec or {}).get("shot_id"))
                for existing in self._jobs.values()
                if existing.session_id == session_id
                and existing.job_type == JobType.full_set
                and existing.status in {JobStatus.queued, JobStatus.processing}
                and (existing.shot_spec or {}).get("shot_id")
            }
            plan = [
                shot for shot in plan
                if str(shot.shot_spec.get("shot_id"))
                not in completed_shot_ids | active_shot_ids
            ]
            if not plan:
                if len(state.generated_images) >= 6:
                    state.status = SessionStatus.done
                    storage.update_session_status(
                        session_id, state.status.value,
                    )
                return []

            state.status = SessionStatus.generating
            state.unlocked = True
            storage.update_session_hero_preview(
                session_id, state.hero_preview_image_id, unlocked=True,
            )
            storage.update_session_status(session_id, state.status.value)
            jobs = []
            for shot in plan:
                shot_spec = dict(shot.shot_spec)
                prompt = shot.prompt
                # The accepted Hero already establishes the look. Supplying the
                # same close-up style image again makes edit models copy its
                # framing despite a different ShotSpec, producing duplicates.
                # Remaining shots use identity references plus textual style.
                template_path = None
                if custom_style_prompt:
                    shot_spec["style_id"] = "private_inspiration"
                    shot_spec["template_id"] = "private_user_reference"
                    shot_spec["template_label"] = "Private inspiration"
                    blocks = dict(shot_spec["prompt_blocks"])
                    blocks["style_block"] = custom_style_prompt
                    shot_spec["prompt_blocks"] = blocks
                    prompt = (
                        f"{custom_style_prompt}\n\nShotSpec:\n"
                        f"- framing: {shot_spec['framing']}\n"
                        f"- pose: {shot_spec['pose']}\n"
                        f"- lighting: {shot_spec['lighting']}\n"
                        f"- lens: {shot_spec['lens']}"
                    )
                job = Job(
                    session_id=session_id,
                    job_type=JobType.full_set,
                    prompt=prompt,
                    prompt_id=shot.prompt_id,
                    template_path=template_path,
                    shot_spec=shot_spec,
                )
                self._jobs[job.job_id] = job
                await self._queue.put(job)
                print(
                    "  📥 Queued full-set job %s (shot=%s), queue size=%s"
                    % (
                        job.job_id,
                        shot.shot_spec.get("shot_id"),
                        self._queue.qsize(),
                    )
                )
                jobs.append(job)

            return jobs

    async def submit_generation(self, session_id: str) -> list[Job]:
        """Create generation jobs from the template-based shot planner."""
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            self.generation_consent_gate(state)
            gate = self.reference_quality_gate(state)
            if not gate.get("pass"):
                state.status = SessionStatus.uploading
                raise ValueError(
                    "Reference photos need replacement before generation: "
                    + "; ".join(gate.get("issues", [])[:8])
                )

            style_data = self._prompts_data["styles"][state.style.value]
            plan = build_style_shot_plan(
                state.style.value,
                state.gender,
                style_data,
            )
            if not plan:
                # Nothing to generate for this gender/style — do NOT flip the
                # status to generating (it would strand the session). Restore the
                # pre-generation status and return empty.
                state.status = (
                    SessionStatus.ready
                    if state.uploaded_photos
                    else SessionStatus.created
                )
                return []

            state.status = SessionStatus.generating
            jobs = []
            for shot in plan:
                job = Job(
                    session_id=session_id,
                    job_type=JobType.generate,
                    prompt=shot.prompt,
                    prompt_id=shot.prompt_id,
                    template_path=self._resolve_template_path(shot.template),
                    shot_spec=shot.shot_spec,
                )
                self._jobs[job.job_id] = job
                await self._queue.put(job)
                print(
                    "  📥 Queued job %s (shot=%s), queue size=%s"
                    % (
                        job.job_id,
                        shot.shot_spec.get("shot_id"),
                        self._queue.qsize(),
                    )
                )
                jobs.append(job)

            return jobs

    async def submit_multi_style_generation(
        self, session_id: str, style_keys: list[StyleKey]
    ) -> list[Job]:
        """Create generation jobs for multiple styles (one template per style).

        Used for multi-style comparison: pick the first matching template
        from each style to give the user a side-by-side comparison.
        """
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            self.generation_consent_gate(state)
            gate = self.reference_quality_gate(state)
            if not gate.get("pass"):
                state.status = SessionStatus.uploading
                raise ValueError(
                    "Reference photos need replacement before generation: "
                    + "; ".join(gate.get("issues", [])[:8])
                )
            state.status = SessionStatus.generating

            jobs = []
            for style_key in style_keys:
                style_data = self._prompts_data["styles"][style_key.value]
                plan = build_style_shot_plan(
                    style_key.value,
                    state.gender,
                    style_data,
                    max_shots=1,
                )
                if not plan:
                    continue

                # Multi-style mode remains one planned shot per style so users
                # can compare styles without exploding cost.
                shot = plan[0]
                job = Job(
                    session_id=session_id,
                    job_type=JobType.generate,
                    prompt=shot.prompt,
                    prompt_id=shot.prompt_id,
                    template_path=self._resolve_template_path(shot.template),
                    shot_spec=shot.shot_spec,
                )
                self._jobs[job.job_id] = job
                await self._queue.put(job)
                jobs.append(job)

            return jobs

    async def submit_revision(
        self, session_id: str, image_id: str, instruction: str
    ) -> Job:
        """Create a revision job (same conversation)."""
        async with self._lock_for(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(session_id)
            self.generation_consent_gate(state)
            if find_registered_image(state, image_id) is None:
                raise FileNotFoundError(image_id)
            if not image_or_source_passed_final_gate(state, image_id):
                raise PermissionError(image_id)
            constrained_instruction = constrain_revision_instruction(instruction)
            state.status = SessionStatus.generating
            state.revisions_used += 1

            job = Job(
                session_id=session_id,
                job_type=JobType.revise,
                prompt=constrained_instruction,
                instruction=constrained_instruction,
                revised_image_id=image_id,
            )
            # Find the turn number from existing images
            existing_turns = [
                img.turn for img in state.generated_images
            ]
            job.turn = max(existing_turns, default=0) + 1

            self._jobs[job.job_id] = job
            await self._queue.put(job)
            return job

    def get_jobs(self, session_id: str) -> list[Job]:
        return [
            j for j in self._jobs.values()
            if j.session_id == session_id
        ]

    async def retry_failed_jobs(self, session_id: str) -> list[Job]:
        """Replace failed in-memory generation jobs with fresh queued jobs."""
        async with self._lock_for(session_id):
            state = self.get_session(session_id)
            if state is None:
                raise KeyError(session_id)
            failed = [
                job for job in self._jobs.values()
                if job.session_id == session_id
                and job.status == JobStatus.failed
                and job.job_type in {JobType.hero_preview, JobType.full_set}
            ]
            if not failed:
                for row in storage.load_generation_events(session_id):
                    if (
                        row["status"] != JobStatus.failed.value
                        or row["failure_reason"] != "set_quality_gate_failed"
                    ):
                        continue
                    try:
                        metadata = json.loads(row["metadata_json"] or "{}")
                        shot_spec = json.loads(row["shot_spec_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    retry = metadata.get("support_retry") or {}
                    prompt = str(retry.get("prompt") or "").strip()
                    if not prompt:
                        continue
                    recovered = Job(
                        session_id=session_id,
                        job_type=JobType.full_set,
                        prompt=prompt,
                        prompt_id=row["prompt_id"],
                        shot_spec=shot_spec,
                    )
                    recovered.job_id = row["job_id"]
                    recovered.status = JobStatus.failed
                    recovered.error = row["error"]
                    self._jobs[recovered.job_id] = recovered
                    failed.append(recovered)
            if not failed:
                raise ValueError(
                    "No retryable failed jobs are loaded; use a replacement entitlement "
                    "or restart the project from its preserved Library state"
                )
            replacements = []
            for old in failed:
                replacement = Job(
                    session_id=old.session_id,
                    job_type=old.job_type,
                    prompt=old.prompt,
                    prompt_id=old.prompt_id,
                    instruction=old.instruction,
                    revised_image_id=old.revised_image_id,
                    template_path=old.template_path,
                    shot_spec=dict(old.shot_spec) if old.shot_spec else None,
                )
                self._jobs.pop(old.job_id, None)
                self._jobs[replacement.job_id] = replacement
                await self._queue.put(replacement)
                replacements.append(replacement)
            state.status = SessionStatus.generating
            storage.update_session_status(session_id, state.status.value)
            return replacements

    def prepare_set_quality_retry(self, session_id: str) -> list[str]:
        """Turn a failed six-frame delivery gate into retryable paid-set jobs.

        The user-confirmed Hero remains the cover. Existing five paid-set
        outputs are lifecycle-marked so fresh retry results cannot produce an
        eight- or eleven-image delivery candidate on the next sync.
        """
        state = self.get_session(session_id)
        if state is None:
            raise KeyError(session_id)
        retry_shot_ids = []
        events_by_job_id = {
            row["job_id"]: row
            for row in storage.load_generation_events(session_id)
        }
        retryable_jobs = [
            job for job in self._jobs.values()
            if job.session_id == session_id
            and job.job_type == JobType.full_set
            and job.status == JobStatus.completed
        ]
        for job in retryable_jobs:
            job.status = JobStatus.failed
            job.error = "set_quality_retry_required"
            shot_id = str((job.shot_spec or {}).get("shot_id") or "")
            if shot_id:
                retry_shot_ids.append(shot_id)
            previous = events_by_job_id.get(job.job_id)
            try:
                metadata = json.loads(previous["metadata_json"] or "{}") if previous else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
            metadata["support_retry"] = {
                "reason": "set_quality_gate_failed",
                "prompt": job.prompt,
                "shot_id": shot_id or None,
            }
            _save_generation_event(
                job,
                JobStatus.failed,
                failure_reason="set_quality_gate_failed",
                error=job.error,
                result_image_id=(
                    previous["result_image_id"] if previous else None
                ),
                metadata=metadata,
            )

        for image in state.generated_images:
            if (
                image.image_id == state.hero_preview_image_id
                or image.parent_image_id is not None
                or image.operation is not None
            ):
                continue
            image.operation = "set_quality_retry_pending"
            storage.mark_generated_image_operation(
                session_id, image.image_id, image.operation,
            )

        state.status = SessionStatus.failed
        storage.update_session_status(session_id, state.status.value)
        return list(dict.fromkeys(retry_shot_ids))

    async def _schedule_automatic_full_set_retry(
        self,
        session: SessionState,
        failed_job: Job,
        reason: str,
        metadata: dict | None = None,
    ) -> Job | None:
        """Atomically replace one failed paid-set shot with a bounded retry."""
        if (
            failed_job.job_type != JobType.full_set
            or failed_job.automatic_retry_count >= MAX_AUTOMATIC_FULL_SET_RETRIES
            or reason not in {
                "transient_provider_error",
                "duplicate_final_asset",
                "delivery_gate_failed",
            }
        ):
            return None

        replacement_prompt = failed_job.prompt
        replacement_shot_spec = (
            dict(failed_job.shot_spec) if failed_job.shot_spec else None
        )
        diagnosis = None
        if reason == "delivery_gate_failed":
            diagnosis = classify_selected_failure(metadata)
            if diagnosis["failure_class"] == "none":
                diagnosis = {
                    **diagnosis,
                    "failure_class": "unknown_quality",
                    "recovery_action": "REGENERATE_FROM_ORIGINAL",
                    "recovery_strategy": "quality_regeneration",
                }
            replacement_shot_spec = build_recovery_shot_spec(
                replacement_shot_spec,
                failure_class=diagnosis["failure_class"],
                attempt=failed_job.automatic_retry_count + 1,
            )
            replacement_prompt = compose_recovery_shot_prompt(
                replacement_prompt,
                replacement_shot_spec,
            )

        async with self._lock_for(session.session_id):
            if self._jobs.get(failed_job.job_id) is not failed_job:
                return None
            replacement = Job(
                session_id=failed_job.session_id,
                job_type=failed_job.job_type,
                prompt=replacement_prompt,
                prompt_id=failed_job.prompt_id,
                instruction=failed_job.instruction,
                revised_image_id=failed_job.revised_image_id,
                template_path=failed_job.template_path,
                shot_spec=replacement_shot_spec,
                automatic_retry_count=failed_job.automatic_retry_count + 1,
                automatic_retry_reason=reason,
            )
            self._jobs.pop(failed_job.job_id, None)
            self._jobs[replacement.job_id] = replacement
            await self._queue.put(replacement)
            _bump_generation_metric(session, "automatic_full_set_retries")
            if diagnosis is not None:
                replacement.automatic_retry_reason = (
                    f"delivery_gate_failed:{diagnosis['failure_class']}"
                )
            session.status = SessionStatus.generating
            storage.update_session_status(session.session_id, session.status.value)

        await self._broadcast(session.session_id, {
            "type": "job_retrying",
            "failed_job_id": failed_job.job_id,
            "job_id": replacement.job_id,
            "prompt_id": replacement.prompt_id,
            "shot_spec": replacement.shot_spec,
            "reason": reason,
            "attempt": replacement.automatic_retry_count,
            "max_attempts": MAX_AUTOMATIC_FULL_SET_RETRIES,
            "recovery": diagnosis,
        })
        return replacement

    def _persist_generation_batch_status(self, session: SessionState, job: Job) -> None:
        """Persist a terminal batch status after either success or failure."""
        if job.job_type == JobType.hero_preview:
            restored_replacement = (
                bool(job.replaces_image_id)
                and session.status == SessionStatus.hero_preview_ready
                and session.hero_preview_image_id == job.replaces_image_id
            )
            if job.status == JobStatus.failed and not restored_replacement:
                session.status = SessionStatus.failed
            storage.update_session_status(session.session_id, session.status.value)
            return
        if job.job_type != JobType.full_set:
            storage.update_session_status(session.session_id, session.status.value)
            return
        batch = [
            other for other in self._jobs.values()
            if other.session_id == session.session_id
            and other.job_type == job.job_type
        ]
        if batch and all(
            other.status in {JobStatus.completed, JobStatus.failed}
            for other in batch
        ):
            session.status = (
                SessionStatus.done
                if all(other.status == JobStatus.completed for other in batch)
                else SessionStatus.failed
            )
        elif any(
            other.status in {JobStatus.queued, JobStatus.processing}
            for other in batch
        ):
            session.status = SessionStatus.generating
        storage.update_session_status(session.session_id, session.status.value)

    def _full_set_batch_is_complete(self, session_id: str) -> bool:
        batch = [
            other for other in self._jobs.values()
            if other.session_id == session_id
            and other.job_type == JobType.full_set
        ]
        return bool(batch) and all(
            other.status == JobStatus.completed for other in batch
        )

    def _set_visual_review_records(self, session: SessionState) -> list[dict]:
        records = []
        for image in session.generated_images:
            if image.parent_image_id is not None or image.operation is not None:
                continue
            if not image_or_source_passed_final_gate(session, image.image_id):
                continue
            path = self.get_image_path(session.session_id, image.image_id)
            if not path:
                continue
            records.append({
                "image_id": image.image_id,
                "storage_path": str(path),
                "prompt_id": image.prompt_id,
                "resemblance": image.resemblance,
            })
        return records[-6:]

    def get_queue_position(self, job_id: str) -> int:
        """Get position of a job in the queue (0 = currently processing)."""
        pos = 0
        for item in list(self._queue._queue):
            pos += 1
            if item.job_id == job_id:
                return pos
        return 0

    def queue_length(self) -> int:
        return self._queue.qsize()

    @property
    def is_busy(self) -> bool:
        """Whether the worker is currently processing a job.

        Deliberately returns a boolean, NOT the session id — exposing which
        session is mid-generation lets one client enumerate/observe others.
        """
        return bool(self._worker and self._worker.active_session_id)

    # ── Worker loop ───────────────────────────────────

    async def _process_loop(self):
        """Continuously dequeue and process one job at a time."""
        print("  🔄 _process_loop started, waiting for jobs...")
        while True:
            try:
                job = await self._queue.get()
                print(f"  🔄 _process_loop dequeued job {job.job_id}")
                await self._execute_job(job)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker error: {e}")
                import traceback
                traceback.print_exc()

    async def _execute_job(self, job: Job):
        """Run a single job via GeminiWorker."""
        print(f"  🔧 _execute_job called: job={job.job_id} type={job.job_type.value} session={job.session_id}")

        # Lazy-connect: if worker is None (API client not ready at startup, e.g.
        # OPENROUTER_API_KEY was missing), try constructing it again now.
        if not self._worker:
            print("  ⏳ Worker not ready, attempting to construct API client...")
            try:
                self._worker = GeminiWorker(learning_layer=self._learning_layer)
                await asyncio.to_thread(self._worker.connect)
                self._worker_readiness_error = None
                self._provider_readiness_checked_at = time.monotonic()
                print("  ✓ API client ready on retry")
            except Exception as exc:
                self._worker_readiness_error = str(exc)
                print(f"  ❌ Worker construction failed: {exc}")
                self._worker = None
                job.status = JobStatus.failed
                job.error = (
                    "Image generation provider is not ready — verify the API key, "
                    "configured model, region availability, and restart the API"
                )
                failed_session = self._sessions.get(job.session_id)
                if failed_session:
                    if not self._restore_replaced_hero(failed_session, job):
                        self._persist_generation_batch_status(failed_session, job)
                await self._broadcast(job.session_id, {
                    "type": "job_failed",
                    "job_id": job.job_id,
                    "error": job.error,
                })
                return

        if not self._worker:
            job.status = JobStatus.failed
            job.error = (
                "Image generation provider is not ready — verify the API key, "
                "configured model, region availability, and restart the API"
            )
            failed_session = self._sessions.get(job.session_id)
            if failed_session:
                if not self._restore_replaced_hero(failed_session, job):
                    self._persist_generation_batch_status(failed_session, job)
            await self._broadcast(job.session_id, {
                "type": "job_failed",
                "job_id": job.job_id,
                "error": job.error,
            })
            return

        session = self._sessions.get(job.session_id)
        if not session:
            job.status = JobStatus.failed
            job.error = "Session not found"
            return

        provider_readiness = getattr(self._worker, "provider_readiness", None)
        if (
            isinstance(provider_readiness, dict)
            and provider_readiness.get("pass") is False
        ):
            job.status = JobStatus.failed
            job.error = (
                "Image generation is temporarily unavailable; support has been "
                "notified and the paid entitlement remains valid"
            )
            if job.job_type in {
                JobType.generate,
                JobType.hero_preview,
                JobType.full_set,
            }:
                _bump_generation_metric(session, "generation_attempts")
                _record_shot_attempt(session, job)
                _record_generation_failure(session, "provider_unavailable")
                _record_shot_failure(session, job, "provider_unavailable")
                await asyncio.to_thread(
                    _save_generation_event,
                    job,
                    JobStatus.failed,
                    "provider_unavailable",
                    job.error,
                )
            if not self._restore_replaced_hero(session, job):
                session.status = SessionStatus.failed
                self._persist_generation_batch_status(session, job)
            await self._broadcast(job.session_id, {
                "type": "job_failed",
                "job_id": job.job_id,
                "error": job.error,
            })
            return

        job.status = JobStatus.processing
        if job.job_type in {JobType.generate, JobType.hero_preview, JobType.full_set}:
            _bump_generation_metric(session, "generation_attempts")
            _record_shot_attempt(session, job)
            await asyncio.to_thread(_save_generation_event, job, JobStatus.processing)
        await self._broadcast(job.session_id, {
            "type": "job_started",
            "job_id": job.job_id,
            "job_type": job.job_type.value,
            "prompt_id": job.prompt_id,
            "shot_spec": job.shot_spec,
        })

        filepath = None
        resemblance_meta = None
        try:
            if job.job_type in {JobType.generate, JobType.full_set}:
                # Use ALL uploaded photos as reference
                photos = [str(p) for p in session.uploaded_photos]

                # Thread-safe progress callback for resemblance agent loop.
                # Captured once; the worker thread calls this from off-loop.
                ev_loop = asyncio.get_running_loop()

                def progress_cb(iteration, max_iter, phase, detail):
                    fut = asyncio.run_coroutine_threadsafe(
                        self._broadcast(job.session_id, {
                            "type": "generation_progress",
                            "job_id": job.job_id,
                            "prompt_id": job.prompt_id,
                            "shot_spec": job.shot_spec,
                            "iteration": iteration,
                            "max_iterations": max_iter,
                            "phase": phase,
                            "detail": detail,
                        }),
                        ev_loop,
                    )
                    # Surface (but swallow) schedule failures so a closed loop
                    # during shutdown never crashes the worker thread.
                    fut.add_done_callback(self._swallow_future_error)

                filepath, resemblance_meta = await asyncio.to_thread(
                    self._worker.execute_generate_with_quality_pipeline,
                    job.session_id,
                    job.prompt,
                    photos,
                    f"{job.session_id}_{job.prompt_id or job.job_id}",
                    job.template_path,
                    progress_cb,
                    job.shot_spec,
                    session.user_feedback,
                )

            elif job.job_type == JobType.hero_preview:
                # Hero preview: simplified pipeline, single close-up portrait
                photos = [str(p) for p in session.uploaded_photos]

                ev_loop = asyncio.get_running_loop()

                def progress_cb(iteration, max_iter, phase, detail):
                    fut = asyncio.run_coroutine_threadsafe(
                        self._broadcast(job.session_id, {
                            "type": "generation_progress",
                            "job_id": job.job_id,
                            "prompt_id": job.prompt_id,
                            "shot_spec": job.shot_spec,
                            "iteration": iteration,
                            "max_iterations": max_iter,
                            "phase": phase,
                            "detail": detail,
                        }),
                        ev_loop,
                    )
                    fut.add_done_callback(self._swallow_future_error)

                filepath, resemblance_meta = await asyncio.to_thread(
                    self._worker.execute_hero_preview,
                    job.session_id,
                    job.prompt,
                    photos,
                    f"{job.session_id}_{job.prompt_id or job.job_id}",
                    job.template_path,
                    progress_cb,
                    job.shot_spec,
                    session.user_feedback,
                )

            elif job.job_type == JobType.revise:
                # Build explicit revision prompt
                instruction = (
                    f"{job.instruction}。"
                    "请保持人物面部特征不变，只修改上述要求的部分。"
                )
                # Resolve the parent image file the user is revising. The parent
                # was delivered earlier so its file lives in output_dir; we still
                # validate the id (path-traversal) and fall back to the canonical
                # path if the file is momentarily absent (e.g. test fixtures) —
                # local_edit will then surface a clear error if it's truly missing.
                parent_image_id = job.revised_image_id or ""
                source_path = self.get_image_path(session.session_id, parent_image_id)
                if source_path is None:
                    source_path = (
                        session.output_dir
                        / f"{safe_id(parent_image_id, label='image_id')}.png"
                    )
                revise_started_at = time.time()
                filepath = await asyncio.to_thread(
                    self._worker.execute_revise,
                    job.session_id,
                    instruction,
                    f"{job.session_id}_rev_{job.turn}",
                    str(source_path),
                )
                photos = [str(p) for p in session.uploaded_photos]
                parent_img = find_registered_image(
                    session,
                    job.revised_image_id or "",
                )
                parent_meta = (
                    parent_img.resemblance
                    if parent_img and isinstance(parent_img.resemblance, dict)
                    else {}
                )
                shot_spec = (
                    parent_meta.get("shot_spec")
                    if isinstance(parent_meta.get("shot_spec"), dict)
                    else None
                )
                revision_judgement = await asyncio.to_thread(
                    self._worker._judge_current_candidate,
                    filepath,
                    photos,
                )
                revision_gate = EvaluationService._candidate_gate_status(
                    revision_judgement,
                    self._worker.identity_thresholds_for_shot(shot_spec),
                )
                revision_deliverable = bool(revision_gate.get("hard_gates_pass"))
                revision_candidate_id = f"revision_{job.job_id}"
                revision_agent_actions = [{
                    "action": "LOCAL_EDIT",
                    "reason": "user_bounded_revision",
                    "state": "MANUAL_REVISION",
                    "executed": True,
                    "candidate_id": revision_candidate_id,
                    "parent_candidate_id": job.revised_image_id,
                }]
                if not revision_deliverable:
                    revision_agent_actions.append({
                        "action": "DROP_CANDIDATE",
                        "reason": "manual_revision_failed_delivery_gate",
                        "state": "FINAL_EVALUATE",
                        "executed": True,
                        "selected_for_execution": True,
                        "candidate_id": revision_candidate_id,
                        "parent_candidate_id": job.revised_image_id,
                        "hard_gate_failures": revision_gate.get(
                            "hard_gate_failures", []
                        ),
                    })
                resemblance_meta = {
                    "pipeline": "manual_local_edit_v1",
                    "allowed_actions": ["LOCAL_EDIT"],
                    "agent_actions": revision_agent_actions,
                    "provider_invocations": [
                        build_provider_invocation_metadata(
                            invocation_id=f"manual_local_edit_{job.job_id}",
                            operation="LOCAL_EDIT",
                            prompt_version="manual_local_edit_v1",
                            reference_ids=[],
                            parent_candidate_id=job.revised_image_id,
                            latency_ms=int((time.time() - revise_started_at) * 1000),
                            result_status="success",
                        )
                    ],
                    "evaluation_result": revision_judgement,
                    "selected_candidate": {
                        "candidate_id": revision_candidate_id,
                        "parent_candidate_id": job.revised_image_id,
                        "filename": Path(filepath).name,
                        "identity_score": (
                            (revision_judgement.get("scores") or {})
                            .get("identity")
                        ),
                        "deliverable": revision_deliverable,
                        "gate_status": revision_gate,
                    },
                }
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            generation_job_types = {
                JobType.generate,
                JobType.hero_preview,
                JobType.full_set,
            }
            gated_job_types = generation_job_types | {JobType.revise}

            delivery_gate_check = None
            if job.job_type in gated_job_types:
                delivery_gate_check = attach_delivery_gate_check(resemblance_meta)

            if (
                job.job_type in gated_job_types
                and not delivery_gate_check.get("pass")
            ):
                job.status = JobStatus.failed
                job.error = delivery_gate_failure_message(resemblance_meta)
                if job.job_type in generation_job_types:
                    _record_generation_failure(session, "delivery_gate_failed")
                    _record_shot_failure(
                        session,
                        job,
                        "delivery_gate_failed",
                        resemblance_meta,
                    )
                    _record_failed_generation_metadata(session, resemblance_meta)
                    await asyncio.to_thread(
                        _save_generation_event,
                        job,
                        JobStatus.failed,
                        "delivery_gate_failed",
                        job.error,
                        metadata=resemblance_meta,
                    )
                    if await self._schedule_automatic_full_set_retry(
                        session, job, "delivery_gate_failed", resemblance_meta
                    ):
                        return
                has_pending_jobs = any(
                    other.session_id == session.session_id
                    and other.job_id != job.job_id
                    and other.status in {JobStatus.queued, JobStatus.processing}
                    for other in self._jobs.values()
                )
                if self._restore_replaced_hero(session, job):
                    pass
                elif session.generated_images:
                    session.status = SessionStatus.reviewing
                elif has_pending_jobs:
                    session.status = SessionStatus.generating
                else:
                    session.status = SessionStatus.failed
                self._persist_generation_batch_status(session, job)
                await self._broadcast(job.session_id, {
                    "type": "job_failed",
                    "job_id": job.job_id,
                    "error": job.error,
                })
                return

            if job.job_type in generation_job_types:
                existing_paths = [
                    session.output_dir / f"{img.image_id}.png"
                    for img in session.generated_images
                    if img.parent_image_id is None and not img.operation
                ]
                duplicate_check = final_duplicate_check(filepath, existing_paths)
                if isinstance(resemblance_meta, dict):
                    final_eval = resemblance_meta.setdefault("final_evaluate", {})
                    if isinstance(final_eval, dict):
                        final_eval["duplicate_check"] = duplicate_check
                        final_eval["status"] = (
                            "pass" if duplicate_check.get("pass", True) else "fail"
                        )
                if not duplicate_check.get("pass", True):
                    job.status = JobStatus.failed
                    job.error = final_duplicate_failure_message(duplicate_check)
                    _record_generation_failure(session, "duplicate_final_asset")
                    _record_shot_failure(
                        session,
                        job,
                        "duplicate_final_asset",
                        resemblance_meta,
                    )
                    _record_failed_generation_metadata(session, resemblance_meta)
                    await asyncio.to_thread(
                        _save_generation_event,
                        job,
                        JobStatus.failed,
                        "duplicate_final_asset",
                        job.error,
                        metadata=resemblance_meta,
                    )
                    if await self._schedule_automatic_full_set_retry(
                        session, job, "duplicate_final_asset"
                    ):
                        return
                    has_pending_jobs = any(
                        other.session_id == session.session_id
                        and other.job_id != job.job_id
                        and other.status in {JobStatus.queued, JobStatus.processing}
                        for other in self._jobs.values()
                    )
                    if self._restore_replaced_hero(session, job):
                        pass
                    elif session.generated_images:
                        session.status = SessionStatus.reviewing
                    elif has_pending_jobs:
                        session.status = SessionStatus.generating
                    else:
                        session.status = SessionStatus.failed
                    self._persist_generation_batch_status(session, job)
                    await self._broadcast(job.session_id, {
                        "type": "job_failed",
                        "job_id": job.job_id,
                        "error": job.error,
                    })
                    return

            # Record result
            image_id = f"img_{uuid.uuid4().hex[:8]}"
            img = GeneratedImage(
                image_id=image_id,
                url=f"/api/sessions/{session.session_id}/images/{image_id}",
                prompt_id=job.prompt_id or "revision",
                turn=job.turn,
                revised_image_id=job.revised_image_id,
                created_at=storage.utcnow(),
                resemblance=resemblance_meta,
            )
            # Move/copy file to session output dir with image_id name
            dest = session.output_dir / f"{image_id}.png"
            clean_dest = clean_export_path(dest)
            final_render_started_at = time.time()
            clean_ai_label = copy_with_ai_metadata(
                filepath,
                clean_dest,
                operation=(
                    "GENERATE"
                    if job.job_type in generation_job_types
                    else "REVISE"
                ),
                source="openrouter_gemini",
                visible_label=False,
            )
            ai_label = copy_with_ai_metadata(
                filepath,
                dest,
                operation=(
                    "GENERATE"
                    if job.job_type in generation_job_types
                    else "REVISE"
                ),
                source="openrouter_gemini",
            )
            if isinstance(resemblance_meta, dict):
                final_eval = resemblance_meta.setdefault("final_evaluate", {})
                ai_label_check = build_ai_label_check(ai_label, clean_ai_label)
                if isinstance(final_eval, dict):
                    final_eval["ai_label_check"] = ai_label_check
                    final_eval["final_render"] = {
                        "pass": True,
                        "status": "pass",
                        "operation": "FINAL_RENDER",
                        "final_asset_id": image_id,
                    }
                    final_eval_checks_pass = ai_label_check.get("pass")
                    if job.job_type in gated_job_types:
                        final_eval_checks_pass = (
                            final_eval_checks_pass
                            and (final_eval.get("delivery_gate") or {}).get("pass", False)
                        )
                    if job.job_type in generation_job_types:
                        final_eval_checks_pass = (
                            final_eval_checks_pass
                            and (final_eval.get("duplicate_check") or {}).get("pass", True)
                        )
                    final_eval["status"] = (
                        "pass" if final_eval_checks_pass else "fail"
                    )
                resemblance_meta["final_asset"] = {
                    "image_id": image_id,
                    "candidate_id": (
                        (resemblance_meta.get("selected_candidate") or {})
                        .get("candidate_id")
                    ),
                    **ai_label,
                    "clean_export": {
                        "available": clean_dest.is_file(),
                        "metadata_ai_label": bool(
                            clean_ai_label.get("metadata_ai_label")
                        ),
                        "visible_ai_label": bool(
                            clean_ai_label.get("visible_ai_label")
                        ),
                    },
                    "ai_label_operation": ai_label.get("operation"),
                    "operation": "FINAL_RENDER",
                }
                img.resemblance = resemblance_meta
            if job.job_type in gated_job_types:
                append_final_render_invocation(
                    resemblance_meta,
                    image_id,
                    int((time.time() - final_render_started_at) * 1000),
                )

            job.result_image = img
            job.status = JobStatus.completed
            session.generated_images.append(img)
            
            # Hero preview: update session state and set hero_preview_image_id
            if job.job_type == JobType.hero_preview:
                session.hero_preview_image_id = image_id
                session.hero_preview_generated = True
                session.status = SessionStatus.hero_preview_ready
                storage.update_session_hero_preview(
                    session.session_id, image_id, unlocked=session.unlocked,
                )
                storage.update_session_status(
                    session.session_id, session.status.value,
                )
                _record_shot_completion(
                    session,
                    job,
                    resemblance_meta,
                    image_id,
                )
                await asyncio.to_thread(
                    _save_generation_event,
                    job,
                    JobStatus.completed,
                    result_image_id=image_id,
                    metadata=resemblance_meta,
                )
            elif job.job_type in {JobType.generate, JobType.full_set}:
                session.status = SessionStatus.reviewing
                _record_shot_completion(
                    session,
                    job,
                    resemblance_meta,
                    image_id,
                )
                if (
                    job.job_type == JobType.full_set
                    and self._worker is not None
                    and self._full_set_batch_is_complete(session.session_id)
                ):
                    review_records = self._set_visual_review_records(session)
                    if len(review_records) == 6:
                        set_visual_review = await asyncio.to_thread(
                            self._worker.evaluate_portrait_set_visual,
                            review_records,
                            str(
                                session.output_dir
                                / f".{session.session_id}-set-review.jpg"
                            ),
                        )
                        resemblance_meta["set_visual_review"] = set_visual_review
                        img.resemblance = resemblance_meta
                if job.automatic_retry_count:
                    _bump_generation_metric(
                        session, "automatic_full_set_retry_successes"
                    )
                await asyncio.to_thread(
                    _save_generation_event,
                    job,
                    JobStatus.completed,
                    result_image_id=image_id,
                    metadata=resemblance_meta,
                )
            else:
                session.status = SessionStatus.reviewing
            # Persist metadata so the gallery (with resemblance scores) survives
            # a backend restart. Pixels are already on disk in output_dir.
            image_metadata_persisted = False
            try:
                storage.save_generated_image(
                    image_id=image_id,
                    session_id=session.session_id,
                    prompt_id=job.prompt_id or "revision",
                    turn=job.turn,
                    revised_image_id=job.revised_image_id,
                    parent_image_id=None,
                    operation=None,
                    resemblance=resemblance_meta,
                    created_at=img.created_at,
                )
                image_metadata_persisted = True
            except Exception as exc:
                print(f"⚠ Could not persist generated-image metadata ({exc})")
                if job.replaces_image_id:
                    img.operation = "failed_preview_retry"
                    raise RuntimeError(
                        "Could not persist replacement preview metadata"
                    ) from exc
            if image_metadata_persisted:
                self._supersede_replaced_hero(session, job)
            if (
                image_metadata_persisted
                and job.job_type in {JobType.generate, JobType.full_set}
            ):
                # Commit the image row before exposing a terminal batch status so
                # restart hydration can never observe done with only five assets.
                self._persist_generation_batch_status(session, job)

            await self._broadcast(job.session_id, {
                "type": "image_ready",
                "job_id": job.job_id,
                "image": img.model_dump(mode="json"),
            })

        except Exception as e:
            import traceback
            print(f"❌ Job {job.job_id} failed: {e}")
            traceback.print_exc()
            job.status = JobStatus.failed
            job.error = str(e)
            if _is_permanent_provider_error(e):
                self._mark_provider_unavailable(e)
            if job.job_type in {JobType.generate, JobType.hero_preview, JobType.full_set}:
                _record_generation_failure(session, "exception")
                _record_shot_failure(session, job, "exception")
                await asyncio.to_thread(
                    _save_generation_event,
                    job,
                    JobStatus.failed,
                    "exception",
                    job.error,
                )
                if _is_transient_generation_error(e):
                    if await self._schedule_automatic_full_set_retry(
                        session, job, "transient_provider_error"
                    ):
                        return
            if not self._restore_replaced_hero(session, job):
                session.status = SessionStatus.failed
                self._persist_generation_batch_status(session, job)

            await self._broadcast(job.session_id, {
                "type": "job_failed",
                "job_id": job.job_id,
                "error": str(e),
            })
        finally:
            if self._worker is not None:
                release_resources = getattr(
                    self._worker, "release_job_resources", None
                )
                if callable(release_resources):
                    await asyncio.to_thread(release_resources, job.session_id)
                else:
                    end_session = getattr(self._worker, "end_session", None)
                    if callable(end_session):
                        end_session(job.session_id)
            await asyncio.to_thread(
                _delete_session_intermediate_outputs, job.session_id
            )

    # ── WebSocket management ──────────────────────────

    def add_ws(self, session_id: str, ws: WebSocket):
        if session_id not in self._ws_connections:
            self._ws_connections[session_id] = []
        self._ws_connections[session_id].append(ws)

    def remove_ws(self, session_id: str, ws: WebSocket):
        conns = self._ws_connections.get(session_id, [])
        if ws in conns:
            conns.remove(ws)

    async def _broadcast(self, session_id: str, message: dict[str, Any]):
        conns = self._ws_connections.get(session_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.remove(ws)

    @staticmethod
    def _swallow_future_error(fut):
        """done_callback for run_coroutine_threadsafe: log+swallow any error
        so a failing broadcast (e.g. loop closed during shutdown) does not
        raise 'exception was never retrieved' in the worker thread."""
        try:
            fut.result()
        except Exception:
            pass

    # ── Styles API ────────────────────────────────────

    def get_styles(self) -> dict:
        return self._prompts_data

    def get_image_path(self, session_id: str, image_id: str) -> Path | None:
        """Resolve an image_id to its file path, rejecting path traversal.

        ``image_id`` is validated to ``[A-Za-z0-9_-]`` only, so ``..`` / ``/``
        cannot escape ``output_dir``.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        safe = safe_id(image_id, label="image_id")
        path = session.output_dir / f"{safe}.png"
        # Belt-and-suspenders: also confirm the resolved path stays in the dir.
        if not path.resolve().is_relative_to(session.output_dir.resolve()):
            return None
        return path if path.exists() else None


# ── Singleton ─────────────────────────────────────────
queue = JobQueue()
