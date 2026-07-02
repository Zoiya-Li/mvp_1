"""Template-based portrait shot planner.

The planner is deliberately code-driven instead of agentic. It turns the
curated style/template library into bounded ShotSpec jobs so generation creates
a small portrait set from the original identity pack, not an open-ended prompt
rewrite loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlannedShot:
    prompt_id: str
    prompt: str
    template: dict[str, Any]
    shot_spec: dict[str, Any]


DEFAULT_PORTRAIT_SHOTS: list[dict[str, str]] = [
    {
        "shot_id": "closeup",
        "label": "Close-up portrait",
        "framing": "close-up portrait, head and shoulders",
        "pose": "natural expression, looking near camera",
        "lighting": "soft flattering portrait light",
        "lens": "85mm portrait lens",
    },
    {
        "shot_id": "half_body",
        "label": "Half-body portrait",
        "framing": "medium half-body portrait",
        "pose": "relaxed shoulders, slight three-quarter angle",
        "lighting": "balanced studio or environmental key light",
        "lens": "50mm portrait lens",
    },
    {
        "shot_id": "environmental",
        "label": "Environmental portrait",
        "framing": "wider environmental portrait with visible scene context",
        "pose": "natural editorial pose, calm confident presence",
        "lighting": "cinematic natural ambience matching the template",
        "lens": "35mm environmental portrait lens",
    },
]

ID_PHOTO_SHOTS: list[dict[str, str]] = [
    {
        "shot_id": "standard",
        "label": "Standard ID photo",
        "framing": "front-facing head and shoulders ID photo",
        "pose": "neutral expression, straight posture, looking at camera",
        "lighting": "flat even ID-photo studio lighting",
        "lens": "70mm portrait lens with minimal distortion",
    }
]


def build_style_shot_plan(
    style_key: str,
    gender: str,
    style_data: dict[str, Any],
    max_shots: int | None = None,
    hero_only: bool = False,
) -> list[PlannedShot]:
    """Build a fixed, auditable shot plan from curated style templates."""
    template_list = style_data.get("templates", style_data.get("prompts", []))
    matching = [t for t in template_list if t.get("gender") == gender]
    if not matching:
        return []

    default_shots = ID_PHOTO_SHOTS if style_key == "id_photo" else DEFAULT_PORTRAIT_SHOTS
    # Hero preview: force closeup for highest identity-preservation success rate
    if hero_only:
        default_shots = [DEFAULT_PORTRAIT_SHOTS[0]]  # closeup only
    limit = max_shots or len(default_shots)
    planned: list[PlannedShot] = []

    explicit_shots = matching[0].get("shots")
    if isinstance(explicit_shots, list) and explicit_shots:
        source = [
            _normalize_shot(shot, fallback=default_shots[min(i, len(default_shots) - 1)])
            for i, shot in enumerate(explicit_shots)
            if isinstance(shot, dict)
        ]
        templates_for_shots = [matching[0]] * len(source)
    else:
        source = default_shots
        templates_for_shots = [matching[min(i, len(matching) - 1)] for i in range(len(source))]

    for idx, shot in enumerate(source[:limit]):
        template = templates_for_shots[idx]
        prompt_id = f"{template['id']}_{shot['shot_id']}"
        shot_spec = _build_shot_spec(style_key, style_data, template, shot, idx + 1, hero_only=hero_only)
        planned.append(
            PlannedShot(
                prompt_id=prompt_id,
                prompt=_compose_shot_prompt(template, shot, shot_spec),
                template=template,
                shot_spec=shot_spec,
            )
        )
    return planned


def _normalize_shot(shot: dict[str, Any], fallback: dict[str, str]) -> dict[str, str]:
    return {
        "shot_id": str(shot.get("shot_id") or fallback["shot_id"]),
        "label": str(shot.get("label") or fallback["label"]),
        "framing": str(shot.get("framing") or fallback["framing"]),
        "pose": str(shot.get("pose") or fallback["pose"]),
        "lighting": str(shot.get("lighting") or fallback["lighting"]),
        "lens": str(shot.get("lens") or fallback["lens"]),
    }


def _build_shot_spec(
    style_key: str,
    style_data: dict[str, Any],
    template: dict[str, Any],
    shot: dict[str, str],
    sequence: int,
    hero_only: bool = False,
) -> dict[str, Any]:
    base_prompt = template.get("prompt") or template.get("gen_prompt") or ""
    
    # Hero preview: use a constrained, high-success-rate prompt
    if hero_only:
        # Force closeup framing with identity-first constraints
        scene_block = (
            "close-up portrait, head and shoulders; "
            "natural expression, looking near camera; "
            "soft flattering portrait light; "
            "85mm portrait lens"
        )
        # Override shot for hero-only mode
        shot = {
            "shot_id": "closeup",
            "label": "Close-up portrait",
            "framing": "close-up portrait, head and shoulders",
            "pose": "natural expression, looking near camera",
            "lighting": "soft flattering portrait light",
            "lens": "85mm portrait lens",
        }
    else:
        scene_block = (
            f"{shot['framing']}; {shot['pose']}; "
            f"{shot['lighting']}; {shot['lens']}"
        )
    
    return {
        "style_id": style_key,
        "style_label": style_data.get("label_en") or style_data.get("label") or style_key,
        "template_id": template.get("id"),
        "template_label": template.get("label"),
        "shot_id": shot["shot_id"],
        "shot_label": shot["label"],
        "sequence": sequence,
        "framing": shot["framing"],
        "pose": shot["pose"],
        "lighting": shot["lighting"],
        "lens": shot["lens"],
        "prompt_blocks": {
            "identity_block": "derived_from_task_identity_pack",
            "scene_block": scene_block,
            "style_block": base_prompt,
            "preservation_block": (
                "preserve the user's real identity, apparent age, face shape, "
                "eyes, nose, mouth, jawline, skin texture, hairline, glasses, "
                "facial hair, moles, and other identity markers"
            ),
        },
    }


def _compose_shot_prompt(
    template: dict[str, Any],
    shot: dict[str, str],
    shot_spec: dict[str, Any],
) -> str:
    base_prompt = template.get("prompt") or template.get("gen_prompt") or ""
    
    # Hero preview constraints: identity-first, conservative composition
    hero_constraints = ""
    if shot_spec.get("shot_id") == "closeup":
        hero_constraints = """
Hero Preview constraints:
- This is the FIRST impression image. It must be a clear, flattering close-up portrait.
- Prioritize facial identity accuracy above all else. Do not trade identity for beauty.
- Use a soft, even studio lighting that flatters the face without harsh shadows.
- Keep the background clean and simple (solid color or subtle gradient).
- Natural, relaxed expression. No exaggerated smiles or forced poses.
- Do NOT include hands, arms, or complex body poses that could introduce artifacts.
- The face should fill approximately 40-50% of the frame for optimal identity recognition."""
    
    return f"""\
{base_prompt}

ShotSpec:
- shot_id: {shot['shot_id']}
- framing: {shot['framing']}
- pose: {shot['pose']}
- lighting: {shot['lighting']}
- lens: {shot['lens']}

Planner constraints:
- This is one shot in a portrait set. Keep the style consistent with the chosen template while varying the composition for this shot.
- Start from the original user reference photos for this shot. Do not inherit face, texture, pose, or composition from any previously generated result.
- Identity preservation is a hard constraint; do not trade identity for beauty or stylization.
- Do not over-beautify. Keep realistic skin texture and the user's apparent age.
- Preservation block: {shot_spec['prompt_blocks']['preservation_block']}.{hero_constraints}"""
