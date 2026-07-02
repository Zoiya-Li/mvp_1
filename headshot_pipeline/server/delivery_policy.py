"""Delivery-policy helpers shared by image serving and post-processing."""

from __future__ import annotations

from .models import GeneratedImage


def find_registered_image(state, image_id: str) -> GeneratedImage | None:
    """Return the registered session image for ``image_id`` if it exists."""
    return next(
        (img for img in state.generated_images if img.image_id == image_id),
        None,
    )


def image_passed_final_gate(img: GeneratedImage) -> bool:
    """Return whether an image has explicit final-QA delivery evidence."""
    meta = img.resemblance if isinstance(img.resemblance, dict) else {}
    selected = meta.get("selected_candidate") if isinstance(meta, dict) else None
    if not isinstance(selected, dict):
        return False
    gate = selected.get("gate_status")
    return bool(
        selected.get("deliverable")
        and isinstance(gate, dict)
        and gate.get("hard_gates_pass")
    )


def image_or_source_passed_final_gate(state, image_id: str) -> bool:
    """Follow post-process/revision ancestry until final-QA evidence is found."""
    seen: set[str] = set()
    current_id = image_id
    for _ in range(12):
        if current_id in seen:
            return False
        seen.add(current_id)
        img = find_registered_image(state, current_id)
        if img is None:
            return False
        if image_passed_final_gate(img):
            return True
        current_id = img.parent_image_id or img.revised_image_id or ""
        if not current_id:
            return False
    return False
