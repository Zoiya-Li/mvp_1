from __future__ import annotations

import base64

from PIL import Image

from server import openrouter_client


def test_data_uri_downsizes_oversized_png_for_vlm_without_touching_source(
    tmp_path, monkeypatch,
):
    source = tmp_path / "large-judge-input.png"
    Image.effect_noise((512, 768), 80).convert("RGB").save(source, "PNG")
    original = source.read_bytes()
    monkeypatch.setattr(openrouter_client, "MAX_DATA_URI_RAW_BYTES", 1024)

    uri = openrouter_client._b64_data_url(source)

    assert uri.startswith("data:image/jpeg;base64,")
    proxy = base64.b64decode(uri.split(",", 1)[1])
    assert len(proxy) < len(original)
    assert source.read_bytes() == original
