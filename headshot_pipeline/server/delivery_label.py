"""AI-generated content labeling for delivered image files.

Delivered artifacts need both an invisible file-level marker and a visible
content marker. The visible mark is intentionally small and placed in the
corner so it satisfies the default delivery contract without dominating the
portrait. A user-requested clean export keeps the invisible marker and omits
the visible mark under the clean-export consent contract.
"""

from __future__ import annotations

import shutil
from pathlib import Path


AI_LABEL_KEY = "AI_Generated"
AI_LABEL_VALUE = "true"
AI_NOTICE_KEY = "AI_Content_Notice"
AI_NOTICE_VALUE = "Generated or edited by FlashShot AI portrait pipeline"
AI_SOFTWARE = "FlashShot"
VISIBLE_LABEL_TEXT = "AI generated"
MIN_VISIBLE_LABEL_DIM = 96
CLEAN_EXPORT_TERMS_VERSION = "cn-ai-label-2025-09-v1"
CLEAN_EXPORT_RETENTION_DAYS = 190


def clean_export_path(delivered_path: str | Path) -> Path:
    """Return the private clean-export sibling for a delivered image."""
    path = Path(delivered_path)
    return path.with_name(f"{path.stem}.clean{path.suffix}")


def _draw_visible_ai_label(img) -> tuple[object, bool]:
    """Return an RGBA image with a subtle visible AI label when size allows."""
    from PIL import Image, ImageDraw, ImageFont

    out = img.convert("RGBA")
    width, height = out.size
    if width < MIN_VISIBLE_LABEL_DIM or height < 64:
        return out, False

    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(11, min(18, width // 44))
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), VISIBLE_LABEL_TEXT, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    pad_x = max(7, width // 120)
    pad_y = max(5, height // 160)
    margin = max(8, min(width, height) // 36)
    left = width - text_width - (pad_x * 2) - margin
    top = height - text_height - (pad_y * 2) - margin
    right = width - margin
    bottom = height - margin

    draw.rectangle((left, top, right, bottom), fill=(0, 0, 0, 135))
    draw.text(
        (left + pad_x, top + pad_y),
        VISIBLE_LABEL_TEXT,
        fill=(255, 255, 255, 230),
        font=font,
    )
    return Image.alpha_composite(out, overlay), True


def copy_with_ai_metadata(
    src: str | Path,
    dest: str | Path,
    *,
    operation: str,
    source: str = "flashshot_ai_pipeline",
    visible_label: bool = True,
) -> dict[str, object]:
    """Copy an image and attach PNG metadata plus an optional visible label."""
    src_path = Path(src)
    dest_path = Path(dest)
    try:
        from PIL import Image, PngImagePlugin

        with Image.open(src_path) as img:
            pnginfo = PngImagePlugin.PngInfo()
            existing = getattr(img, "text", {}) or {}
            for key, value in existing.items():
                if isinstance(key, str) and isinstance(value, str):
                    pnginfo.add_text(key, value)
            pnginfo.add_text(AI_LABEL_KEY, AI_LABEL_VALUE)
            pnginfo.add_text(AI_NOTICE_KEY, AI_NOTICE_VALUE)
            pnginfo.add_text("Software", AI_SOFTWARE)
            pnginfo.add_text("FlashShotOperation", operation)
            pnginfo.add_text("FlashShotSource", source)
            pnginfo.add_text(
                "FlashShotVisibleLabelReserved",
                "true" if visible_label else "false",
            )
            pnginfo.add_text("FlashShotVisibleLabel", VISIBLE_LABEL_TEXT)
            pnginfo.add_text(
                "FlashShotCleanExport",
                "false" if visible_label else "true",
            )

            if visible_label:
                out, visible_applied = _draw_visible_ai_label(img)
            else:
                out = img.convert("RGBA")
                visible_applied = False
            pnginfo.add_text(
                "FlashShotVisibleLabelApplied",
                "true" if visible_applied else "false",
            )
            out.save(dest_path, format="PNG", pnginfo=pnginfo)
            return {
                "metadata_ai_label": True,
                "visible_ai_label": visible_applied,
                "visible_label_reserved": visible_label,
                "clean_export": not visible_label,
                "operation": operation,
                "source": source,
            }
    except Exception:
        shutil.copy2(src_path, dest_path)
        return {
            "metadata_ai_label": False,
            "visible_ai_label": False,
            "visible_label_reserved": False,
            "clean_export": not visible_label,
            "operation": operation,
            "source": source,
        }


def read_ai_metadata(path: str | Path) -> dict[str, str]:
    """Read PNG text metadata for tests and audits."""
    from PIL import Image

    with Image.open(path) as img:
        return dict(getattr(img, "text", {}) or {})
