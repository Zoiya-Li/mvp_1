from __future__ import annotations

import base64
import io
import json
import urllib.error

import pytest
from PIL import Image

from server.openrouter_image_client import (
    OpenRouterImageClient,
    OpenRouterImageError,
)


class _Response:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._payload


def _client(tmp_path) -> OpenRouterImageClient:
    return OpenRouterImageClient(
        api_key="sk-or-test",
        output_dir=tmp_path,
        model="bytedance-seed/seedream-4.5",
        base_url="https://openrouter.test/api/v1",
        timeout=10,
        image_size="1728x2304",
        provider_tag="seed",
        estimated_image_cost=0.04,
        minimum_credit_balance=3.0,
        max_reference_images=5,
    )


def test_readiness_checks_reference_support_and_real_balance(tmp_path, monkeypatch):
    client = _client(tmp_path)

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/credits"):
            return _Response({
                "data": {"total_credits": 10.0, "total_usage": 0.45}
            })
        return _Response({
            "id": client.model,
            "endpoints": [{
                "supported_parameters": {"input_references": {"min": 0, "max": 14}}
            }],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = client.check_readiness()

    assert result["pass"] is True
    assert result["remaining_credits"] == 9.55
    assert result["minimum_credit_balance"] == 3.0


def test_readiness_fails_before_generation_when_balance_is_low(tmp_path, monkeypatch):
    client = _client(tmp_path)

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/credits"):
            return _Response({
                "data": {"total_credits": 10.0, "total_usage": 8.0}
            })
        return _Response({
            "id": client.model,
            "endpoints": [{"supported_parameters": {"input_references": {}}}],
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(OpenRouterImageError, match="insufficient credit"):
        client.check_readiness()


def test_generate_uses_dedicated_image_api_and_pinned_provider(tmp_path, monkeypatch):
    client = _client(tmp_path)
    reference = tmp_path / "reference.jpg"
    Image.new("RGB", (640, 800), "#8c725e").save(reference, "JPEG")
    raw = io.BytesIO()
    Image.new("RGB", (48, 64), "#d8c4b2").save(raw, "JPEG")
    encoded = base64.b64encode(raw.getvalue()).decode("ascii")
    captured = {}

    monkeypatch.setattr(
        client,
        "require_credit_balance",
        lambda minimum=None: {"remaining": 9.5},
    )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _Response({
            "data": [{"b64_json": encoded, "media_type": "image/jpeg"}],
            "usage": {"cost": 0.04},
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    output = client.generate(
        prompt="A real editorial portrait",
        reference_paths=[str(reference)],
        title="seedream-test",
    )

    assert captured["url"].endswith("/api/v1/images")
    assert captured["payload"]["model"] == "bytedance-seed/seedream-4.5"
    assert captured["payload"]["size"] == "1728x2304"
    assert captured["payload"]["provider"] == {
        "allow_fallbacks": False,
        "only": ["seed"],
    }
    assert len(captured["payload"]["input_references"]) == 1
    reference_url = captured["payload"]["input_references"][0]["image_url"]["url"]
    assert reference_url.startswith("data:image/jpeg;base64,")
    assert client.last_usage["cost"] == 0.04
    with Image.open(output) as image:
        assert image.format == "PNG"


def test_reference_data_uri_normalizes_progressive_jpeg_and_caps_pixels(tmp_path):
    reference = tmp_path / "large-progressive.jpg"
    Image.new("RGB", (2400, 3200), "#8c725e").save(
        reference,
        "JPEG",
        quality=95,
        progressive=True,
    )

    data_uri = OpenRouterImageClient._data_uri(reference)

    prefix, encoded = data_uri.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as normalized:
        assert normalized.format == "JPEG"
        assert normalized.info.get("progressive") is None
        assert normalized.width * normalized.height <= 950_000
