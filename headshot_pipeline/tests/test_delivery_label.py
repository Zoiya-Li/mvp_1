"""Tests for AI-generated content metadata on delivered images."""

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

from server.delivery_label import (  # noqa: E402
    AI_LABEL_KEY,
    AI_LABEL_VALUE,
    VISIBLE_LABEL_TEXT,
    copy_with_ai_metadata,
    read_ai_metadata,
)


def test_copy_with_ai_metadata_labels_delivered_png(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    src = tmp_path / "src.png"
    dest = tmp_path / "dest.png"
    Image.new("RGB", (32, 32), color=(20, 30, 40)).save(src, format="PNG")

    result = copy_with_ai_metadata(src, dest, operation="GENERATE", source="test")

    meta = read_ai_metadata(dest)
    assert result["metadata_ai_label"] is True
    assert result["visible_label_reserved"] is True
    assert result["operation"] == "GENERATE"
    assert result["source"] == "test"
    assert meta[AI_LABEL_KEY] == AI_LABEL_VALUE
    assert meta["AI_Content_Notice"].startswith("Generated or edited")
    assert meta["FlashShotOperation"] == "GENERATE"
    assert meta["FlashShotSource"] == "test"
    assert meta["FlashShotVisibleLabelReserved"] == "true"
    assert meta["FlashShotVisibleLabel"] == VISIBLE_LABEL_TEXT


def test_copy_with_ai_metadata_preserves_existing_text(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    PngImagePlugin = pytest.importorskip("PIL.PngImagePlugin")

    src = tmp_path / "src.png"
    dest = tmp_path / "dest.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("ExistingKey", "ExistingValue")
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(
        src, format="PNG", pnginfo=info
    )

    result = copy_with_ai_metadata(src, dest, operation="crop_1in")

    meta = read_ai_metadata(dest)
    assert result["metadata_ai_label"] is True
    assert result["visible_ai_label"] is False
    assert meta["ExistingKey"] == "ExistingValue"
    assert meta[AI_LABEL_KEY] == AI_LABEL_VALUE
    assert meta["FlashShotVisibleLabelApplied"] == "false"


def test_copy_with_ai_metadata_applies_visible_label_on_delivered_png(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    src = tmp_path / "src.png"
    dest = tmp_path / "dest.png"
    Image.new("RGB", (240, 160), color=(90, 120, 150)).save(src, format="PNG")

    result = copy_with_ai_metadata(src, dest, operation="GENERATE")

    meta = read_ai_metadata(dest)
    assert result["visible_ai_label"] is True
    assert meta["FlashShotVisibleLabelApplied"] == "true"

    with Image.open(src) as before, Image.open(dest) as after:
        label_region = (120, 110, 238, 158)
        before_bytes = before.convert("RGBA").crop(label_region).tobytes()
        after_bytes = after.convert("RGBA").crop(label_region).tobytes()
        assert before_bytes != after_bytes
