"""Set-level portrait delivery checks.

Per-image QA cannot prove that six acceptable frames form one coherent set.
This module keeps deterministic delivery failures separate from visual-review
warnings so uncalibrated aesthetic opinions never strand a paid order.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw


EXPECTED_SHOT_IDS = (
    "closeup",
    "half_body",
    "environmental",
    "seated",
    "profile",
    "candid",
)

LOOK_A_SHOTS = frozenset(EXPECTED_SHOT_IDS[:3])
LOOK_B_SHOTS = frozenset(EXPECTED_SHOT_IDS[3:])

VISUAL_SET_JUDGE_PROMPT = """You are reviewing one six-frame editorial portrait set.
The contact sheet is ordered left-to-right, top-to-bottom and each frame is
labeled with its expected shot ID.

Judge the SET, not isolated beauty. A successful set must:
- visibly depict the same person in all six frames;
- keep one coherent Look A across frames 1-3 and one coherent Look B across 4-6;
- progress through close, medium, environmental, seated, turned, and candid
  compositions instead of repeating centered phone-selfie framing;
- show readable environments and an intentional emotional sequence;
- retain ordinary skin texture, asymmetry, camera depth, and believable light;
- avoid recurring AI artifacts, generic beauty-face normalization, or six
  unrelated fashion images.

Return ONLY valid JSON in this schema:
{
  "scores": {
    "identity_consistency": 0,
    "wardrobe_continuity": 0,
    "composition_variety": 0,
    "narrative_coherence": 0,
    "realism": 0,
    "commercial_readiness": 0
  },
  "hard_failures": [],
  "retry_shot_ids": [],
  "notes": ""
}

Allowed hard_failures: identity_drift, look_a_discontinuity,
look_b_discontinuity, repeated_composition, unreadable_environment,
synthetic_set_appearance, broken_story_sequence. Scores are integers 0-10.
retry_shot_ids may only contain closeup, half_body, environmental, seated,
profile, or candid. Be strict and concise."""


def _image_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("resemblance")
    return metadata if isinstance(metadata, dict) else {}


def _shot_spec(item: dict[str, Any]) -> dict[str, Any]:
    metadata = _image_metadata(item)
    spec = metadata.get("shot_spec")
    return spec if isinstance(spec, dict) else {}


def _shot_id(item: dict[str, Any]) -> str:
    shot_id = _shot_spec(item).get("shot_id")
    if shot_id:
        return str(shot_id)
    prompt_id = str(item.get("prompt_id") or "")
    for expected in EXPECTED_SHOT_IDS:
        if prompt_id == expected or prompt_id.endswith(f"_{expected}"):
            return expected
    return prompt_id


def _selected_candidate(item: dict[str, Any]) -> dict[str, Any]:
    selected = _image_metadata(item).get("selected_candidate")
    return selected if isinstance(selected, dict) else {}


def _final_judgement(item: dict[str, Any]) -> dict[str, Any]:
    judgement = _selected_candidate(item).get("final_judgement")
    return judgement if isinstance(judgement, dict) else {}


def _dhash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        pixels = image.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        ).tobytes()
    value = 0
    width = hash_size + 1
    for row in range(hash_size):
        offset = row * width
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_set_contact_sheet(
    images: list[dict[str, Any]], output_path: str | Path,
) -> Path:
    """Build a numbered 2x3 review sheet without altering final assets."""
    output = Path(output_path)
    panel_width, panel_height, label_height = 384, 512, 34
    sheet = Image.new("RGB", (panel_width * 2, panel_height * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(images[:6]):
        path = Path(str(item.get("storage_path") or ""))
        with Image.open(path) as source:
            frame = source.convert("RGB")
            frame.thumbnail(
                (panel_width, panel_height - label_height),
                Image.Resampling.LANCZOS,
            )
        x = (index % 2) * panel_width
        y = (index // 2) * panel_height
        paste_x = x + (panel_width - frame.width) // 2
        paste_y = y + label_height + (panel_height - label_height - frame.height) // 2
        sheet.paste(frame, (paste_x, paste_y))
        label = f"{index + 1}. {_shot_id(item) or 'unknown'}"
        draw.text((x + 12, y + 10), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=88, optimize=True)
    return output


def _parse_visual_set_review(text: str) -> dict[str, Any]:
    allowed_scores = {
        "identity_consistency",
        "wardrobe_continuity",
        "composition_variety",
        "narrative_coherence",
        "realism",
        "commercial_readiness",
    }
    allowed_failures = {
        "identity_drift",
        "look_a_discontinuity",
        "look_b_discontinuity",
        "repeated_composition",
        "unreadable_environment",
        "synthetic_set_appearance",
        "broken_story_sequence",
    }
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    data = json.loads(match.group(0) if match else text)
    raw_scores = data.get("scores") if isinstance(data, dict) else {}
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    scores = {}
    for key in allowed_scores:
        value = raw_scores.get(key)
        if not isinstance(value, (int, float)):
            scores[key] = None
        else:
            scores[key] = int(max(0, min(10, round(float(value)))))
    failures = data.get("hard_failures") if isinstance(data, dict) else []
    failures = failures if isinstance(failures, list) else []
    retry = data.get("retry_shot_ids") if isinstance(data, dict) else []
    retry = retry if isinstance(retry, list) else []
    return {
        "status": "reviewed",
        "scores": scores,
        "hard_failures": [
            str(item) for item in failures if str(item) in allowed_failures
        ],
        "retry_shot_ids": [
            str(item) for item in retry if str(item) in EXPECTED_SHOT_IDS
        ],
        "notes": str(data.get("notes") or "")[:1_000],
        "blocking": False,
        "policy_version": "portrait_set_visual_review_v1_diagnostic",
    }


def judge_visual_portrait_set(
    gateway,
    images: list[dict[str, Any]],
    *,
    contact_sheet_path: str | Path,
) -> dict[str, Any]:
    """Run one diagnostic VLM review of the six-frame contact sheet."""
    sheet = build_set_contact_sheet(images, contact_sheet_path)
    try:
        response = gateway.judge(
            current_image_path=str(sheet),
            reference_paths=[],
            judge_prompt=VISUAL_SET_JUDGE_PROMPT,
            timeout=180,
        )
        review = _parse_visual_set_review(response)
        review["raw_response"] = str(response or "")[:4_000]
        return review
    except Exception as exc:
        return {
            "status": "judge_failed",
            "scores": {},
            "hard_failures": ["visual_set_judge_failed"],
            "retry_shot_ids": [],
            "notes": str(exc)[:1_000],
            "blocking": False,
            "policy_version": "portrait_set_visual_review_v1_diagnostic",
        }


def _visual_review_payload(
    *,
    shot_ids: list[str],
    identity_scores: list[float],
    identity_cosines: list[float],
    face_area_ratios: list[float],
    face_center_offsets: list[float],
) -> dict[str, Any]:
    warnings: list[str] = []
    if len(identity_scores) == 6 and max(identity_scores) - min(identity_scores) > 2:
        warnings.append("identity_score_drift")
    if len(identity_cosines) == 6 and max(identity_cosines) - min(identity_cosines) > 0.20:
        warnings.append("identity_cosine_drift")
    if len(face_area_ratios) == 6 and max(face_area_ratios) - min(face_area_ratios) < 0.03:
        warnings.append("weak_framing_variety")
    if (
        len(face_center_offsets) == 6
        and sum(offset < 0.04 for offset in face_center_offsets) >= 5
    ):
        warnings.append("over_centered_sequence")
    return {
        "required": True,
        "status": "pending_calibrated_visual_review",
        "warnings": warnings,
        "criteria": [
            "same_visible_person_across_all_frames",
            "look_a_continuity_frames_1_to_3",
            "look_b_continuity_frames_4_to_6",
            "readable_location_and_pose_progression",
            "natural_skin_and_non_synthetic_camera_rendering",
            "emotional_arc_without_repeated_composition",
        ],
        "shot_ids": shot_ids,
    }


def evaluate_portrait_set(images: list[dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic hard failures plus non-blocking visual warnings.

    Each item needs ``image_id``, ``storage_path``, ``prompt_id`` and the
    generation ``resemblance`` metadata. The function is side-effect free so
    it can be used by delivery, support tooling, and production E2E tests.
    """
    hard_failures: list[str] = []
    diagnostics: dict[str, Any] = {}
    if len(images) != 6:
        hard_failures.append("set_must_contain_exactly_six_images")

    image_ids = [str(item.get("image_id") or "") for item in images]
    if not all(image_ids) or len(set(image_ids)) != len(image_ids):
        hard_failures.append("set_image_ids_not_unique")

    shot_ids = [_shot_id(item) for item in images]
    diagnostics["shot_ids"] = shot_ids
    if set(shot_ids) != set(EXPECTED_SHOT_IDS) or len(set(shot_ids)) != 6:
        hard_failures.append("set_shot_coverage_incomplete")

    file_records: list[dict[str, Any]] = []
    exact_hashes: dict[str, str] = {}
    perceptual_hashes: dict[str, int] = {}
    for item, image_id in zip(images, image_ids):
        path = Path(str(item.get("storage_path") or ""))
        if not path.is_file():
            hard_failures.append("set_asset_missing")
            continue
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            perceptual_hash = _dhash(path)
        except Exception:
            hard_failures.append("set_asset_unreadable")
            continue
        exact_hashes[image_id] = content_hash
        perceptual_hashes[image_id] = perceptual_hash
        file_records.append({
            "image_id": image_id,
            "sha256": content_hash,
            "dhash": f"{perceptual_hash:016x}",
        })
    diagnostics["assets"] = file_records
    if len(set(exact_hashes.values())) != len(exact_hashes):
        hard_failures.append("set_contains_exact_duplicate")

    near_duplicates = []
    for left, right in combinations(perceptual_hashes, 2):
        distance = _hamming(perceptual_hashes[left], perceptual_hashes[right])
        if distance <= 2:
            near_duplicates.append({
                "image_ids": [left, right],
                "dhash_distance": distance,
            })
    diagnostics["near_duplicates"] = near_duplicates
    if near_duplicates:
        hard_failures.append("set_contains_near_duplicate")

    identity_scores: list[float] = []
    identity_cosines: list[float] = []
    face_area_ratios: list[float] = []
    face_center_offsets: list[float] = []
    geometry_profiles: list[str] = []
    per_shot: dict[str, Any] = {}
    for item, shot_id in zip(images, shot_ids):
        spec = _shot_spec(item)
        selected = _selected_candidate(item)
        gate = selected.get("gate_status")
        gate = gate if isinstance(gate, dict) else {}
        judgement = _final_judgement(item)
        scores = judgement.get("scores")
        scores = scores if isinstance(scores, dict) else {}
        identity = judgement.get("identity_quality")
        identity = identity if isinstance(identity, dict) else {}
        local = judgement.get("local_quality")
        local = local if isinstance(local, dict) else {}
        measurements = local.get("measurements")
        measurements = measurements if isinstance(measurements, dict) else {}

        if not selected or not judgement:
            hard_failures.append("set_image_missing_final_qa")
        if not gate.get("hard_gates_pass") or not selected.get("deliverable"):
            hard_failures.append("set_image_failed_delivery_gate")

        score = scores.get("identity")
        cosine = identity.get("cosine_similarity")
        face_area = measurements.get("face_area_ratio")
        center_dx = measurements.get("face_center_dx")
        geometry = measurements.get("geometry_profile")
        if isinstance(score, (int, float)):
            identity_scores.append(float(score))
        if isinstance(cosine, (int, float)):
            identity_cosines.append(float(cosine))
        if isinstance(face_area, (int, float)):
            face_area_ratios.append(float(face_area))
        if isinstance(center_dx, (int, float)):
            face_center_offsets.append(float(center_dx))
        if geometry:
            geometry_profiles.append(str(geometry))

        wardrobe = str(spec.get("wardrobe") or "").lower()
        narrative = str(spec.get("narrative") or "").strip()
        expected_look = "look a" if shot_id in LOOK_A_SHOTS else "look b"
        if not narrative:
            hard_failures.append("set_narrative_metadata_missing")
        if shot_id in EXPECTED_SHOT_IDS and expected_look not in wardrobe:
            hard_failures.append("set_look_assignment_incomplete")
        per_shot[shot_id] = {
            "identity_score": score,
            "identity_cosine": cosine,
            "face_area_ratio": face_area,
            "face_center_dx": center_dx,
            "geometry_profile": geometry,
            "expected_look": expected_look,
            "narrative": narrative,
            "hard_gates_pass": bool(gate.get("hard_gates_pass")),
        }

    diagnostics["per_shot"] = per_shot
    diagnostics["identity_score_range"] = (
        [min(identity_scores), max(identity_scores)] if identity_scores else []
    )
    diagnostics["identity_cosine_range"] = (
        [min(identity_cosines), max(identity_cosines)] if identity_cosines else []
    )
    diagnostics["face_area_ratio_range"] = (
        [min(face_area_ratios), max(face_area_ratios)] if face_area_ratios else []
    )
    diagnostics["geometry_profiles"] = geometry_profiles

    if len(face_area_ratios) != 6 or len(face_center_offsets) != 6:
        hard_failures.append("set_geometry_measurements_incomplete")
    elif (
        sum(area >= 0.16 for area in face_area_ratios) >= 5
        and sum(offset <= 0.08 for offset in face_center_offsets) >= 5
    ):
        hard_failures.append("set_is_selfie_dominated")
    if len(set(geometry_profiles)) < 3:
        hard_failures.append("set_framing_profiles_not_varied")

    hard_failures = list(dict.fromkeys(hard_failures))
    visual_review = next(
        (
            review for item in reversed(images)
            if isinstance(
                review := _image_metadata(item).get("set_visual_review"), dict
            )
        ),
        None,
    )
    if visual_review is None:
        visual_review = _visual_review_payload(
            shot_ids=shot_ids,
            identity_scores=identity_scores,
            identity_cosines=identity_cosines,
            face_area_ratios=face_area_ratios,
            face_center_offsets=face_center_offsets,
        )
    return {
        "pass": not hard_failures,
        "hard_failures": hard_failures,
        "diagnostics": diagnostics,
        "visual_review": visual_review,
        "policy_version": "portrait_set_delivery_v1",
    }
