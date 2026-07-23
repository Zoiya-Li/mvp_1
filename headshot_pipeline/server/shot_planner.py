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
        "label": "Editorial close portrait",
        "framing": "natural chest-up editorial portrait with breathing room",
        "pose": "subtle three-quarter turn, relaxed expression, gaze near camera",
        "environment": "a window table in a quiet neighborhood cafe, with street reflections, a timber or metal window frame, and one practical lamp",
        "lighting": "believable window light or open-shade daylight",
        "lens": "50mm to 70mm lens at f/2.8 to f/4",
        "wardrobe": "Look A: the style's primary polished outfit, with one coordinated outer layer and a simple base garment",
        "narrative": "arrival; a warm, quietly confident first encounter",
    },
    {
        "shot_id": "half_body",
        "label": "Half-body portrait",
        "framing": "medium half-body portrait from complete head to waist or upper thighs",
        "pose": "relaxed shoulders, slight three-quarter angle, arms resting naturally",
        "environment": "a bright gallery or hotel lobby with a readable doorway, seating, stone or tile floor, and architectural depth",
        "lighting": "directional window light with ordinary local contrast",
        "lens": "50mm lens at f/3.2 to f/4",
        "wardrobe": "Look A: exactly the same polished outfit family, garment colors, and materials as the close portrait",
        "narrative": "composure; poised in a public interior before stepping into the city",
    },
    {
        "shot_id": "environmental",
        "label": "Environmental portrait",
        "framing": "medium-to-wide environmental portrait from complete head to waist or upper thighs, with the location occupying at least half the canvas",
        "pose": "standing slightly off-center with a calm, unforced presence",
        "environment": "a covered city arcade or broad sidewalk edge with storefronts, paving lines, passing daylight, and deep street perspective",
        "lighting": "soft daylight with a readable background and no abstract blur field",
        "lens": "35mm to 50mm lens at f/4 with moderate depth of field",
        "wardrobe": "Look A: the same polished outfit family as the first two frames, shown continuously through the wider crop",
        "narrative": "context; the person belongs naturally to a living city rather than a studio",
    },
    {
        "shot_id": "seated",
        "label": "Seated portrait",
        "framing": (
            "medium seated portrait from complete head to waist, with the chair "
            "back and seated posture unmistakably visible"
        ),
        "pose": "relaxed seated posture, open shoulders, subtle turn toward camera; include hands only when complete, natural, and artifact-free",
        "environment": "a book-lined library lounge with a clearly visible chair, side table, reading lamp, and window",
        "lighting": "directional window light consistent with the set",
        "lens": "50mm lens at f/3.2 to f/4",
        "wardrobe": "Look B: a softer off-duty variation in the same palette, using one textured layer over a simple base garment",
        "narrative": "pause; an intimate, thoughtful chapter with relaxed confidence",
    },
    {
        "shot_id": "profile",
        "label": "Turned portrait",
        "framing": "chest-up three-quarter turned portrait with asymmetric shoulders",
        "pose": "shoulders and face turned 25 to 45 degrees; gaze may meet the camera or pass just beside it",
        "environment": "an open-air architectural passage or rooftop terrace edge with repeating lines and a distant urban layer",
        "lighting": "side window light that gives the face dimensionality without a halo",
        "lens": "50mm to 70mm lens at f/3.2 to f/4",
        "wardrobe": "Look B: exactly the same off-duty outfit family, garment colors, and materials as the seated frame",
        "narrative": "movement; attention briefly drawn toward something beyond the frame",
    },
    {
        "shot_id": "candid",
        "label": "Candid portrait",
        "framing": "chest-up to half-body candid portrait with enough surroundings to identify the location",
        "pose": "quiet in-between moment, eyes looking toward a window or activity outside the frame, expression unperformed",
        "environment": "an independent bookshop or maker studio with shelves, books or materials, a work surface, and a lived-in foreground object",
        "lighting": "available light with ordinary contrast and no generic studio blur",
        "lens": "50mm documentary lens at f/4",
        "wardrobe": "Look B: the same off-duty outfit family as the seated and turned frames",
        "narrative": "afterglow; an unperformed in-between moment that closes the story",
    },
]

ID_PHOTO_SHOTS: list[dict[str, str]] = [
    {
        "shot_id": "standard",
        "label": "Standard ID photo",
        "framing": "front-facing head and shoulders ID photo",
        "pose": "neutral expression, straight posture, looking at camera",
        "environment": "plain compliant ID-photo background",
        "lighting": "flat even ID-photo studio lighting",
        "lens": "70mm portrait lens with minimal distortion",
        "wardrobe": "plain compliant clothing",
        "narrative": "documentary identification",
    }
]


RECOVERY_SHOT_VARIANTS: dict[str, dict[str, str]] = {
    "half_body": {
        "variant": "waist_up_relaxed",
        "framing": "waist-up editorial portrait with complete head and arms",
        "pose": "gentle three-quarter stance with arms relaxed and hands outside the crop",
        "lens": "60mm portrait lens at f/4",
    },
    "environmental": {
        "variant": "medium_environmental",
        "framing": "medium environmental portrait from complete head to upper thighs, with the real location occupying at least half the canvas",
        "pose": "standing off-center with a small natural weight shift",
        "lens": "50mm documentary lens at f/4.5",
    },
    "seated": {
        "variant": "seated_clean_hands",
        "framing": "waist-up seated portrait with chair back and seat edge clearly visible",
        "pose": "relaxed seated posture turned slightly toward camera; hands resting below the crop",
        "lens": "60mm portrait lens at f/4",
    },
    "profile": {
        "variant": "soft_turned_portrait",
        "framing": "chest-up turned portrait with breathing room and asymmetric shoulders",
        "pose": "head and shoulders turned 20 to 35 degrees, with both eyes naturally readable and gaze just beside camera",
        "lens": "65mm portrait lens at f/4",
    },
    "candid": {
        "variant": "observational_glance",
        "framing": "waist-up observational portrait with a clearly readable real environment",
        "pose": "a quiet glance toward nearby activity, body mostly still, expression unperformed",
        "lens": "50mm documentary lens at f/4.5",
    },
}


def build_recovery_shot_spec(
    shot_spec: dict[str, Any] | None,
    *,
    failure_class: str,
    attempt: int,
) -> dict[str, Any]:
    """Return a lower-risk variant for the same narrative delivery slot."""
    recovered = dict(shot_spec or {})
    shot_id = str(recovered.get("shot_id") or "portrait")
    variant = RECOVERY_SHOT_VARIANTS.get(shot_id, {
        "variant": "conservative_portrait",
        "framing": "natural chest-up editorial portrait with breathing room",
        "pose": "subtle three-quarter turn with a relaxed, unforced expression",
        "lens": "60mm portrait lens at f/4",
    })
    recovered.update({key: value for key, value in variant.items() if key != "variant"})
    recovered.update({
        "canonical_shot_id": shot_id,
        "shot_variant": variant["variant"],
        "recovery_attempt": max(1, int(attempt)),
        "recovery_failure_class": failure_class or "unknown_quality",
    })
    prompt_blocks = dict(recovered.get("prompt_blocks") or {})
    prompt_blocks["scene_block"] = "; ".join(
        str(recovered.get(key) or "")
        for key in ("framing", "pose", "environment", "lighting", "lens")
    )
    recovered["prompt_blocks"] = prompt_blocks
    return recovered


def compose_recovery_shot_prompt(
    original_prompt: str,
    shot_spec: dict[str, Any],
) -> str:
    """Append an authoritative replacement ShotSpec to a failed paid-set job."""
    return f"""{original_prompt}

RECOVERY SHOTSPEC (this block supersedes earlier crop, pose, and lens instructions):
- canonical delivery slot: {shot_spec.get('canonical_shot_id') or shot_spec.get('shot_id')}
- replacement variant: {shot_spec.get('shot_variant')}
- framing: {shot_spec.get('framing')}
- pose: {shot_spec.get('pose')}
- environment: {shot_spec.get('environment')}
- lighting: {shot_spec.get('lighting')}
- lens: {shot_spec.get('lens')}

This is a new photographic solution for the same story beat. Do not recreate
the failed composition. Preserve the person's identity and the set's wardrobe,
palette, location family, and emotional continuity."""


def build_style_shot_plan(
    style_key: str,
    gender: str,
    style_data: dict[str, Any],
    max_shots: int | None = None,
    hero_only: bool = False,
    template_id: str | None = None,
    shot_overrides: list[dict[str, Any]] | None = None,
) -> list[PlannedShot]:
    """Build a fixed, auditable shot plan from curated style templates."""
    template_list = style_data.get("templates", style_data.get("prompts", []))
    matching = [t for t in template_list if t.get("gender") == gender]
    if template_id:
        matching = [t for t in matching if t.get("id") == template_id]
    if not matching:
        return []

    default_shots = ID_PHOTO_SHOTS if style_key == "id_photo" else DEFAULT_PORTRAIT_SHOTS
    # Hero preview: force closeup for highest identity-preservation success rate
    if hero_only:
        default_shots = [DEFAULT_PORTRAIT_SHOTS[0]]  # closeup only
    limit = max_shots or len(default_shots)
    planned: list[PlannedShot] = []

    explicit_shots = shot_overrides or matching[0].get("shots")
    if isinstance(explicit_shots, list) and explicit_shots:
        source = [
            _normalize_shot(shot, fallback=default_shots[min(i, len(default_shots) - 1)])
            for i, shot in enumerate(explicit_shots)
            if isinstance(shot, dict)
        ]
        templates_for_shots = [matching[0]] * len(source)
    else:
        source = default_shots
        look_b_template = matching[min(1, len(matching) - 1)]
        templates_for_shots = [
            matching[0] if i < 3 else look_b_template
            for i in range(len(source))
        ]

    for idx, shot in enumerate(source[:limit]):
        template = templates_for_shots[idx]
        prompt_id = f"{template['id']}_{shot['shot_id']}"
        shot_spec = _build_shot_spec(
            style_key,
            style_data,
            template,
            shot,
            idx + 1,
            hero_only=hero_only,
            curated_series=bool(shot_overrides),
        )
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
        "environment": str(shot.get("environment") or fallback["environment"]),
        "lighting": str(shot.get("lighting") or fallback["lighting"]),
        "lens": str(shot.get("lens") or fallback["lens"]),
        "wardrobe": str(shot.get("wardrobe") or fallback["wardrobe"]),
        "narrative": str(shot.get("narrative") or fallback["narrative"]),
        "style_prompt": str(shot.get("style_prompt") or ""),
    }


def _build_shot_spec(
    style_key: str,
    style_data: dict[str, Any],
    template: dict[str, Any],
    shot: dict[str, str],
    sequence: int,
    hero_only: bool = False,
    curated_series: bool = False,
) -> dict[str, Any]:
    if shot.get("style_prompt"):
        base_prompt = shot["style_prompt"]
    elif hero_only:
        base_prompt = _hero_style_direction(style_data, template)
    elif style_key == "id_photo":
        base_prompt = template.get("gen_prompt") or template.get("prompt") or ""
    else:
        base_prompt = _shot_style_direction(style_data, template)
    
    # Hero preview: use a constrained, high-success-rate prompt
    if hero_only:
        scene_block = (
            f"{shot['framing']}; {shot['pose']}; {shot['environment']}; "
            f"{shot['lighting']}; {shot['lens']}"
        )
    else:
        scene_block = (
            f"{shot['framing']}; {shot['pose']}; {shot['environment']}; "
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
        "environment": shot["environment"],
        "lighting": shot["lighting"],
        "lens": shot["lens"],
        "wardrobe": shot["wardrobe"],
        "narrative": shot["narrative"],
        "prompt_blocks": {
            "identity_block": "derived_from_task_identity_pack",
            "scene_block": scene_block,
            "style_block": base_prompt,
            "preservation_block": (
                "preserve the user's real identity, apparent age, face shape, "
                "eyes, nose, mouth, jawline, skin texture, hairline, glasses, "
                "facial hair, moles, and other identity markers"
            ),
            "set_continuity_block": (
                "follow the selected shoot blueprint exactly across all six frames; "
                "keep the named wardrobe, materials, time of day, color response, "
                "and adjacent-location details continuous; do not invent another "
                "outfit, season, city, or unrelated setting"
                if curated_series else
                "build one six-frame editorial story with two controlled wardrobe "
                "looks; Look A spans frames 1-3 and Look B spans frames 4-6; "
                "every frame uses its own named, visibly different real location"
            ),
        },
    }


def _hero_style_direction(
    style_data: dict[str, Any],
    template: dict[str, Any],
) -> str:
    """Carry aesthetic intent into Hero without inheriting template geometry."""
    explicit = str(template.get("hero_prompt") or "").strip()
    if explicit:
        return explicit
    return (
        "Use the selected wardrobe materials, restrained color palette, and calm "
        "emotional tone in a physically readable real editorial location. Do not "
        "name or imitate a regional beauty trend, celebrity, beauty campaign, or "
        "social-media filter. Do not inherit the template subject, backdrop, "
        "framing, camera angle, studio-lighting diagram, or lens instructions; "
        "the Hero ShotSpec controls all geometry and lighting."
    )


def _shot_style_direction(
    style_data: dict[str, Any],
    template: dict[str, Any],
) -> str:
    """Return style intent without geometry that can conflict with ShotSpec."""
    explicit = str(template.get("style_prompt") or "").strip()
    if explicit:
        return explicit
    style_label = str(
        style_data.get("label_en") or style_data.get("label") or "Editorial portrait"
    ).strip()
    template_label = str(template.get("label") or "").strip()
    labels = " · ".join(value for value in (style_label, template_label) if value)
    return (
        f"{labels}. Carry only this style's wardrobe, palette, material texture, "
        "and emotional tone into the photograph. ShotSpec is the sole authority "
        "for crop, pose, camera angle, environment, lighting, and lens. Do not "
        "default to a studio headshot, seamless or gradient backdrop, centered "
        "ID-photo framing, or instructions from the legacy template prompt."
    )


def _compose_shot_prompt(
    template: dict[str, Any],
    shot: dict[str, str],
    shot_spec: dict[str, Any],
) -> str:
    base_prompt = str(
        ((shot_spec.get("prompt_blocks") or {}).get("style_block"))
        or template.get("gen_prompt")
        or template.get("prompt")
        or ""
    )
    
    # Hero preview constraints: identity-first, conservative composition
    hero_constraints = ""
    if shot_spec.get("shot_id") == "closeup":
        hero_constraints = """
Hero Preview constraints:
- This is the FIRST impression image. It must look like a real editorial photograph, not an AI beauty portrait or ID photo.
- Prioritize facial identity accuracy above all else. Do not trade identity for beauty.
- Preserve natural facial asymmetry, moles, under-eye structure, pores, flyaway hairs, and small skin-tone variations.
- Use believable directional window light or open-shade daylight with ordinary local contrast.
- Keep a real environment readable behind the person. No solid or gradient backdrop, abstract blur, halo, lens flare, or fake bokeh.
- Use a subtle three-quarter body and face turn with a natural in-between expression. Avoid perfectly centered bilateral symmetry.
- Do NOT include hands, arms, or complex body poses that could introduce artifacts.
- The portrait should be chest-up with breathing room; the face should occupy roughly 28-38% of frame height."""

    shot_constraints = ""
    if shot_spec.get("shot_id") == "seated":
        shot_constraints = """
Seated portrait constraints:
- Show enough chair back and posture to read clearly as seated.
- Hands may appear only when complete, naturally posed, and anatomically correct; never crop through fingers.
- Keep the face large, unobstructed, and naturally proportioned."""
    
    return f"""\
{base_prompt}

ShotSpec:
- shot_id: {shot['shot_id']}
- framing: {shot['framing']}
- pose: {shot['pose']}
- environment: {shot['environment']}
- lighting: {shot['lighting']}
- lens: {shot['lens']}
- wardrobe chapter: {shot['wardrobe']}
- narrative beat: {shot['narrative']}

Planner constraints:
- This is one shot in a portrait set. Keep the chosen shoot direction consistent and follow this ShotSpec's exact composition and setting.
- Set continuity: {shot_spec['prompt_blocks']['set_continuity_block']}.
- Obey this frame's wardrobe chapter and narrative beat. Do not improvise an unplanned outfit family or unrelated location.
- The environment must contain physically coherent, recognizable details; do not replace it with a smooth wall, abstract blur, or generic studio backdrop.
- Start from the original user reference photos for this shot. Do not inherit face, texture, pose, or composition from any previously generated result.
- Identity preservation is a hard constraint; do not trade identity for beauty or stylization.
- Do not over-beautify. Keep realistic skin texture and the user's apparent age.
- Render like an unretouched real camera photograph: retain pores, fine lines, subtle uneven skin tone, flyaway hairs, fabric wrinkles, and ordinary local contrast.
- No beauty filter, plastic or waxy skin, CGI polish, perfect bilateral symmetry, pristine showroom surfaces, or globally softened facial detail.
- Keep small physically plausible imperfections in the location, materials, and mixed practical light so the setting feels inhabited rather than generated.
- Preservation block: {shot_spec['prompt_blocks']['preservation_block']}.{hero_constraints}{shot_constraints}"""
