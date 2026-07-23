"""Turn a private inspiration image into a safe, identity-free shoot spec."""

from __future__ import annotations

import json
import re
from typing import Any


INSPIRATION_ANALYSIS_PROMPT = """You are a photography art director.
Analyze the supplied inspiration image and return one JSON object only.
Describe reusable photographic attributes, never the source person's identity.
Do not identify a real person, infer sensitive traits, or reproduce text/logos.

Required keys:
{
  "scene": "short scene description",
  "wardrobe": "generic wardrobe silhouette and colors",
  "lighting": "lighting direction and quality",
  "composition": "framing, camera angle and subject placement",
  "pose": "generic pose",
  "palette": ["3-5 colors"],
  "mood": "short mood",
  "camera": "lens/depth-of-field character",
  "complexity": "low|medium|high",
  "safety": {"pass": true, "reasons": []},
  "forbidden_transfer": [
    "source_person_identity", "logos", "watermarks", "unique_text",
    "exact_background_copy"
  ]
}

Reject with safety.pass=false when the image is sexual, exploitative, appears
to involve a minor in an adult context, or is intended for impersonation.
"""


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1)
    else:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start:end + 1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Inspiration analysis must be a JSON object")
    return parsed


def analyze_with_provider(provider, image_path: str) -> dict[str, Any]:
    raw = provider.judge(
        current_image_path=image_path,
        reference_paths=[],
        judge_prompt=INSPIRATION_ANALYSIS_PROMPT,
    )
    result = parse_json_object(raw)
    safety = result.get("safety") or {}
    if safety.get("pass") is not True:
        reasons = safety.get("reasons") or ["inspiration_not_allowed"]
        raise ValueError("Inspiration rejected: " + ", ".join(map(str, reasons)))
    required = {"scene", "wardrobe", "lighting", "composition", "pose", "mood"}
    if not required.issubset(result):
        raise ValueError("Inspiration analysis is incomplete")
    result["forbidden_transfer"] = sorted(set(
        list(result.get("forbidden_transfer") or []) + [
            "source_person_identity", "logos", "watermarks", "unique_text",
        ]
    ))
    return result


def inspiration_generation_prompt(
    spec: dict[str, Any], *, hero_only: bool = True,
) -> str:
    """Compose an identity-safe prompt from an approved inspiration spec."""
    palette = ", ".join(map(str, spec.get("palette") or []))
    hero_instruction = (
        "This is a clear, flattering close-up hero portrait with no hands or complex pose."
        if hero_only else
        "Follow the supplied ShotSpec for this image while keeping the complete set visually consistent."
    )
    return f"""Create a new portrait of the identity shown only in the user's identity references.

Use the separate inspiration image only as photographic art direction:
- scene: {spec.get('scene', 'editorial portrait setting')}
- wardrobe: {spec.get('wardrobe', 'coordinated portrait wardrobe')}
- lighting: {spec.get('lighting', 'flattering portrait light')}
- composition: {spec.get('composition', 'close portrait composition')}
- pose: {spec.get('pose', 'natural portrait pose')}
- palette: {palette or 'cohesive restrained colors'}
- mood: {spec.get('mood', 'editorial')}
- camera: {spec.get('camera', 'portrait lens with natural depth of field')}

Never copy the source person's face, body identity, distinctive marks, logos,
watermarks, text, or exact background. Preserve the user's real facial identity,
apparent age, face shape, skin texture, hairline, and other identity markers.
{hero_instruction}"""
