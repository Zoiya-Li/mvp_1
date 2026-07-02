"""Tests for image provider gateway metadata and cost estimation."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.image_gateway import (  # noqa: E402
    build_provider_invocation_metadata,
    estimate_cost,
    invocation_provider_metadata,
    provider_for_operation,
)


def test_gateway_estimates_reference_heavy_generation_cost():
    assert estimate_cost("CREATE_FROM_REFERENCES", reference_count=1) == 0.12
    assert estimate_cost("CREATE_FROM_REFERENCES", reference_count=4) == 0.1488
    assert estimate_cost("IDENTITY_REPAIR", reference_count=4) == 0.0
    assert estimate_cost("UPSCALE", reference_count=0) == 0.0
    assert estimate_cost("FINAL_RENDER", reference_count=0) == 0.0


def test_gateway_exposes_provider_capabilities():
    cap = provider_for_operation("CREATE_FROM_REFERENCES")
    assert cap.provider == "openrouter"
    assert cap.supports_multiple_references is True
    assert cap.max_reference_images == 4

    meta = invocation_provider_metadata("LOCAL_EDIT")
    assert meta["provider"] == "openrouter"
    assert meta["provider_capabilities"]["supports_portrait_ratio"] is True

    final_render = invocation_provider_metadata("FINAL_RENDER")
    assert final_render["provider"] == "local"
    assert final_render["model"] == "flashshot_delivery_packager_v1"
    assert final_render["provider_capabilities"]["estimated_cost"] == 0.0

    upscale = invocation_provider_metadata("UPSCALE")
    assert upscale["provider"] == "local"
    assert upscale["model"] == "realesrgan_x2_with_lanczos_fallback"
    assert upscale["provider_capabilities"]["supports_high_fidelity"] is True


def test_gateway_builds_spec_aligned_provider_invocation_record():
    record = build_provider_invocation_metadata(
        invocation_id="create_1",
        operation="CREATE_FROM_REFERENCES",
        prompt_version="controlled_candidate_v2",
        reference_ids=["ref_1", "ref_2", "ref_3"],
        reference_roles=[{"reference_id": "ref_1", "role": "front_neutral"}],
        candidate_index=1,
        parent_candidate_id=None,
        shot_id="closeup",
        latency_ms=17830,
        cost=0.14,
        result_status="success",
    )

    assert record["provider"] == "openrouter"
    assert record["operation"] == "CREATE_FROM_REFERENCES"
    assert record["prompt_version"] == "controlled_candidate_v2"
    assert record["reference_ids"] == ["ref_1", "ref_2", "ref_3"]
    assert record["parent_candidate_id"] is None
    assert record["latency_ms"] == 17830
    assert record["cost"] == 0.14
    assert record["estimated_cost"] == 0.14
    assert record["result_status"] == "success"
    assert record["provider_capabilities"]["supports_multiple_references"] is True
