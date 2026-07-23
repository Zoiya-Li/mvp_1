"""Image provider gateway metadata and cost estimation.

MVP still routes to one primary image model, but business logic should speak in
operations (COMPOSITION_SCAFFOLD / CREATE_FROM_REFERENCES / IDENTITY_BLEND /
LOCAL_EDIT / UPSCALE / FINAL_RENDER) and
capability records rather than scattering provider constants through the agent loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .config import settings
from .models import ProviderInvocation

ImageOperation = Literal[
    "COMPOSITION_SCAFFOLD",
    "CREATE_FROM_REFERENCES",
    "IDENTITY_BLEND",
    "LOCAL_EDIT",
    "IDENTITY_REPAIR",
    "UPSCALE",
    "FINAL_RENDER",
]


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    model: str
    supports_multiple_references: bool
    supports_mask_edit: bool
    supports_high_fidelity: bool
    supports_seed: bool
    supports_portrait_ratio: bool
    max_reference_images: int
    average_latency_ms: int
    estimated_cost: float
    supported_tasks: tuple[str, ...]


OPENROUTER_GEMINI_CAPABILITIES = ProviderCapabilities(
    provider="openrouter",
    model=settings.gemini_model,
    supports_multiple_references=True,
    supports_mask_edit=False,
    supports_high_fidelity=True,
    supports_seed=True,
    supports_portrait_ratio=True,
    max_reference_images=settings.openrouter_max_reference_images,
    average_latency_ms=55_000,
    estimated_cost=settings.openrouter_estimated_image_cost,
    supported_tasks=(
        "hero_face",
        "half_body",
        "full_body",
        "environmental",
        "local_edit",
    ),
)

SILICONFLOW_QWEN_CAPABILITIES = ProviderCapabilities(
    provider="siliconflow",
    model=settings.siliconflow_image_model,
    supports_multiple_references=True,
    supports_mask_edit=False,
    supports_high_fidelity=True,
    supports_seed=True,
    supports_portrait_ratio=True,
    max_reference_images=3,
    average_latency_ms=42_000,
    estimated_cost=settings.siliconflow_estimated_image_cost,
    supported_tasks=(
        "hero_face",
        "half_body",
        "full_body",
        "environmental",
        "local_edit",
    ),
)

SILICONFLOW_QWEN_COMPOSITION_CAPABILITIES = ProviderCapabilities(
    provider="siliconflow",
    model=settings.siliconflow_text_to_image_model,
    supports_multiple_references=False,
    supports_mask_edit=False,
    supports_high_fidelity=True,
    supports_seed=True,
    supports_portrait_ratio=True,
    max_reference_images=0,
    average_latency_ms=42_000,
    estimated_cost=settings.siliconflow_estimated_image_cost,
    supported_tasks=("half_body", "full_body", "environmental", "composition"),
)

CHROME_GEMINI_CAPABILITIES = ProviderCapabilities(
    provider="chrome",
    model="gemini-web-ui",
    supports_multiple_references=True,
    supports_mask_edit=False,
    supports_high_fidelity=True,
    supports_seed=False,
    supports_portrait_ratio=True,
    max_reference_images=4,
    average_latency_ms=45_000,
    estimated_cost=0.0,
    supported_tasks=(
        "hero_face",
        "half_body",
        "full_body",
        "environmental",
        "local_edit",
    ),
)

LOCAL_IDENTITY_REPAIR_CAPABILITIES = ProviderCapabilities(
    provider="local",
    model="inswapper_128",
    supports_multiple_references=True,
    supports_mask_edit=False,
    supports_high_fidelity=False,
    supports_seed=False,
    supports_portrait_ratio=True,
    max_reference_images=8,
    average_latency_ms=2_000,
    estimated_cost=0.0,
    supported_tasks=("identity_repair",),
)

LOCAL_FINAL_RENDER_CAPABILITIES = ProviderCapabilities(
    provider="local",
    model="flashshot_delivery_packager_v1",
    supports_multiple_references=False,
    supports_mask_edit=False,
    supports_high_fidelity=False,
    supports_seed=False,
    supports_portrait_ratio=True,
    max_reference_images=0,
    average_latency_ms=250,
    estimated_cost=0.0,
    supported_tasks=("final_render",),
)

LOCAL_UPSCALE_CAPABILITIES = ProviderCapabilities(
    provider="local",
    model="realesrgan_x2_with_lanczos_fallback",
    supports_multiple_references=False,
    supports_mask_edit=False,
    supports_high_fidelity=True,
    supports_seed=False,
    supports_portrait_ratio=True,
    max_reference_images=0,
    average_latency_ms=5_000,
    estimated_cost=0.0,
    supported_tasks=("upscale",),
)


def provider_for_operation(operation: ImageOperation) -> ProviderCapabilities:
    if operation == "COMPOSITION_SCAFFOLD":
        if getattr(settings, "gemini_backend", "openrouter") == "siliconflow":
            return SILICONFLOW_QWEN_COMPOSITION_CAPABILITIES
        if getattr(settings, "gemini_backend", "openrouter") == "chrome":
            return CHROME_GEMINI_CAPABILITIES
        return OPENROUTER_GEMINI_CAPABILITIES
    if operation == "IDENTITY_REPAIR":
        return LOCAL_IDENTITY_REPAIR_CAPABILITIES
    if operation == "UPSCALE":
        return LOCAL_UPSCALE_CAPABILITIES
    if operation == "FINAL_RENDER":
        return LOCAL_FINAL_RENDER_CAPABILITIES
    if getattr(settings, "gemini_backend", "openrouter") == "chrome":
        return CHROME_GEMINI_CAPABILITIES
    if getattr(settings, "gemini_backend", "openrouter") == "siliconflow":
        return SILICONFLOW_QWEN_CAPABILITIES
    return OPENROUTER_GEMINI_CAPABILITIES


def estimate_cost(operation: ImageOperation, reference_count: int = 1) -> float:
    """Estimate invocation cost, including reference-image input pressure.

    We keep the MVP estimate conservative and deterministic. Reference-heavy
    requests cost slightly more than a one-reference request, so metrics can
    distinguish "more references improved identity but increased cost".
    """
    cap = provider_for_operation(operation)
    if cap.estimated_cost <= 0:
        return 0.0
    extra_refs = max(0, reference_count - 1)
    multiplier = 1.0 + 0.08 * extra_refs
    return round(cap.estimated_cost * multiplier, 4)


def invocation_provider_metadata(operation: ImageOperation) -> dict:
    cap = provider_for_operation(operation)
    return {
        "provider": cap.provider,
        "model": cap.model,
        "provider_capabilities": asdict(cap),
    }


def build_provider_invocation_metadata(
    *,
    invocation_id: str,
    operation: ImageOperation,
    prompt_version: str | None,
    reference_ids: list[str] | None = None,
    reference_roles: list[dict] | None = None,
    candidate_index: int | None = None,
    parent_candidate_id: str | None = None,
    shot_id: str | None = None,
    final_asset_id: str | None = None,
    latency_ms: int | None = None,
    cost: float | None = None,
    result_status: str = "success",
) -> dict:
    """Build a spec-aligned, analytics-ready provider invocation record."""
    cap = provider_for_operation(operation)
    if cost is None:
        cost = estimate_cost(operation, len(reference_ids or []))
    payload = ProviderInvocation(
        invocation_id=invocation_id,
        provider=cap.provider,
        model=cap.model,
        operation=operation,
        prompt_version=prompt_version,
        reference_ids=reference_ids or [],
        reference_roles=reference_roles or [],
        candidate_index=candidate_index,
        parent_candidate_id=parent_candidate_id,
        shot_id=shot_id,
        final_asset_id=final_asset_id,
        latency_ms=latency_ms,
        cost=cost,
        estimated_cost=cost,
        provider_capabilities=asdict(cap),
        result_status=result_status,
    )
    return payload.model_dump(mode="json")
