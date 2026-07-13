from __future__ import annotations

import base64
import io
import json

from PIL import Image

from server.generation.providers import SiliconFlowProvider


def _image(path, color):
    Image.new("RGB", (24, 32), color).save(path)
    return str(path)


def _png_b64(color="navy"):
    buffer = io.BytesIO()
    Image.new("RGB", (30, 40), color).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_create_reserves_third_slot_for_template(monkeypatch, tmp_path):
    provider = SiliconFlowProvider("test-key", output_dir=tmp_path)
    refs = [_image(tmp_path / f"ref_{index}.png", "red") for index in range(4)]
    template = _image(tmp_path / "template.png", "blue")
    captured = {}

    def fake_request(endpoint, payload, **kwargs):
        captured.update({"endpoint": endpoint, "payload": payload})
        return {"images": [{"b64_json": _png_b64()}]}

    monkeypatch.setattr(provider, "_request_json", fake_request)

    result = provider.create_from_references(
        "Create a professional headshot.", refs, template, "hero"
    )

    assert captured["endpoint"] == "images/generations"
    assert captured["payload"]["model"] == provider.image_model
    assert set(captured["payload"]) >= {"image", "image2", "image3"}
    assert "image4" not in captured["payload"]
    assert base64.b64decode(captured["payload"]["image3"].split(",", 1)[1]) == (
        tmp_path / "template.png"
    ).read_bytes()
    assert "template only" in captured["payload"]["prompt"]
    assert Image.open(result).size == (30, 40)


def test_local_edit_puts_current_image_first(monkeypatch, tmp_path):
    provider = SiliconFlowProvider("test-key", output_dir=tmp_path)
    current = _image(tmp_path / "current.png", "green")
    refs = [_image(tmp_path / f"identity_{index}.png", "yellow") for index in range(3)]
    captured = {}

    def fake_request(endpoint, payload, **kwargs):
        captured["payload"] = payload
        return {"images": [{"b64_json": _png_b64("white")}]}

    monkeypatch.setattr(provider, "_request_json", fake_request)

    provider.local_edit(current, refs, "Remove the collar artifact.", "fixed")

    first_image = base64.b64decode(captured["payload"]["image"].split(",", 1)[1])
    assert first_image == (tmp_path / "current.png").read_bytes()
    assert "image3" in captured["payload"]
    assert "image4" not in captured["payload"]


def test_judge_disables_thinking_at_top_level(monkeypatch, tmp_path):
    provider = SiliconFlowProvider("test-key", output_dir=tmp_path)
    current = _image(tmp_path / "candidate.png", "green")
    reference = _image(tmp_path / "reference.png", "red")
    captured = {}

    def fake_request(endpoint, payload, **kwargs):
        captured.update({"endpoint": endpoint, "payload": payload})
        return {"choices": [{"message": {"content": json.dumps({"score": 0.9})}}]}

    monkeypatch.setattr(provider, "_request_json", fake_request)

    verdict = provider.judge(current, [reference], "Return JSON only.")

    assert json.loads(verdict) == {"score": 0.9}
    assert captured["endpoint"] == "chat/completions"
    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert "extra_body" not in captured["payload"]
    assert len(captured["payload"]["messages"][0]["content"]) == 3


def test_readiness_requires_both_configured_models(monkeypatch, tmp_path):
    provider = SiliconFlowProvider("test-key", output_dir=tmp_path)
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda *args, **kwargs: {
            "data": [{"id": provider.image_model}, {"id": provider.judge_model}]
        },
    )

    result = provider.check_readiness()

    assert result["pass"] is True
    assert result["provider"] == "siliconflow"
    assert result["judge_model"] == provider.judge_model
