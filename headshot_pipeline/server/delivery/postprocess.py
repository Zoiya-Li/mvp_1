"""Delivery post-processing — upscale, final render, AI label, download policy."""

from __future__ import annotations

from pathlib import Path


def no_op_upscale(image_path: str) -> str:
    """Placeholder upscale: returns the same path.

    Future versions will wire to RealESRGAN or a dedicated cloud upscale model.
    """
    return image_path


def final_render_path(image_path: str, suffix: str = "_final") -> str:
    """Return a canonical final-render path next to the source image."""
    p = Path(image_path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))
