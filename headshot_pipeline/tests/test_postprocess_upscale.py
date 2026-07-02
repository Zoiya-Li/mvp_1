"""Tests for postprocess upscale provenance metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

import server.router_postprocess as router_postprocess  # noqa: E402
from server.models import (  # noqa: E402
    GeneratedImage,
    PricingTier,
    SessionState,
    StyleKey,
    utcnow,
)
from server.router_postprocess import UpscaleRequest, upscale_hd  # noqa: E402


def _deliverable_meta(deliverable: bool = True) -> dict:
    return {
        "selected_candidate": {
            "candidate_id": "cand_1",
            "deliverable": deliverable,
            "gate_status": {
                "hard_gates_pass": deliverable,
                "hard_gate_failures": [] if deliverable else ["identity_fail"],
            },
        }
    }


@pytest.mark.asyncio
async def test_upscale_records_gateway_provider_invocation(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")

    state = SessionState("s_upscale", StyleKey.business, "female", "tok")
    state.tier = PricingTier.premium
    state.output_dir = tmp_path / "outputs"
    state.output_dir.mkdir()
    source = state.output_dir / "img_parent.png"
    Image.new("RGB", (120, 80), color=(30, 80, 130)).save(source, format="PNG")
    state.generated_images.append(GeneratedImage(
        image_id="img_parent",
        url="/api/sessions/s_upscale/images/img_parent",
        prompt_id="closeup",
        turn=1,
        created_at=utcnow(),
        resemblance=_deliverable_meta(True),
    ))

    def fake_upscale(input_path, output_path, scale):
        Image.open(input_path).resize((240, 160)).save(output_path, format="PNG")
        return output_path, "lanczos_x2"

    monkeypatch.setattr(router_postprocess, "upscale_image", fake_upscale)

    response = await upscale_hd(
        "s_upscale",
        UpscaleRequest(image_id="img_parent"),
        state=state,
    )

    assert response.original_image_id == "img_parent"
    assert response.operation == "upscale_x2_lanczos"
    img = state.generated_images[-1]
    assert img.image_id == response.processed_image_id
    meta = img.resemblance
    invocation = meta["provider_invocations"][0]
    assert invocation["operation"] == "UPSCALE"
    assert invocation["provider"] == "local"
    assert invocation["model"] == "realesrgan_x2_with_lanczos_fallback"
    assert invocation["parent_candidate_id"] == "img_parent"
    assert invocation["final_asset_id"] == response.processed_image_id
    assert invocation["upscale_method"] == "lanczos_x2"
    assert invocation["provider_capabilities"]["supports_high_fidelity"] is True
    assert meta["final_asset"]["operation"] == "upscale_x2_lanczos"
    final_eval = meta["final_evaluate"]
    assert final_eval["status"] == "pass"
    assert final_eval["delivery_gate"]["status"] == "pass"
    assert final_eval["delivery_gate"]["source_image_id"] == "img_parent"
    assert final_eval["delivery_gate"]["deliverable_ancestor_image_id"] == "img_parent"
    assert final_eval["delivery_gate"]["inherited_from_source"] is False
    assert final_eval["ai_label_check"]["status"] == "pass"
    assert final_eval["final_render"]["operation"] == "FINAL_RENDER"
    assert final_eval["final_render"]["postprocess_operation"] == "upscale_x2_lanczos"
    assert final_eval["final_render"]["final_asset_id"] == response.processed_image_id
    assert meta["provider_invocations"][-1]["operation"] == "FINAL_RENDER"
    assert meta["provider_invocations"][-1]["final_asset_id"] == response.processed_image_id


@pytest.mark.asyncio
async def test_upscale_rejects_non_deliverable_source_image(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")

    state = SessionState("s_upscale_bad", StyleKey.business, "female", "tok")
    state.tier = PricingTier.premium
    state.output_dir = tmp_path / "outputs"
    state.output_dir.mkdir()
    source = state.output_dir / "img_bad.png"
    Image.new("RGB", (120, 80), color=(30, 80, 130)).save(source, format="PNG")
    state.generated_images.append(GeneratedImage(
        image_id="img_bad",
        url="/api/sessions/s_upscale_bad/images/img_bad",
        prompt_id="closeup",
        turn=1,
        created_at=utcnow(),
        resemblance=_deliverable_meta(False),
    ))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("upscale should not run for non-deliverable sources")

    monkeypatch.setattr(router_postprocess, "upscale_image", fail_if_called)

    with pytest.raises(router_postprocess.HTTPException) as exc:
        await upscale_hd(
            "s_upscale_bad",
            UpscaleRequest(image_id="img_bad"),
            state=state,
        )

    assert exc.value.status_code == 409
    assert state.generated_images[-1].image_id == "img_bad"
