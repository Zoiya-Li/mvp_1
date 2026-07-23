"""Gemini worker — adapts the Gemini client for the job queue.

Tracks which session currently "owns" the generation pipeline and handles
switching between sessions (end old → start new).

Generation backend is now abstracted through ImageGateway; business code
no longer directly calls OpenRouter or Chrome client methods.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .evaluation import (
    EvaluationService,
    AgentRouter,
    PolicyEngine,
    EpisodeRecoveryPlanner,
    classify_failure,
    select_best_variant,
)
from .evaluation.set_evaluator import judge_visual_portrait_set
from .generation import ImageGateway
from .image_gateway import (
    build_provider_invocation_metadata,
    estimate_cost,
    provider_for_operation,
)
from .learning import LearningLayer
from .models import IdentityPack, ShotSpec
from .repair import (
    FaceSwapRepair,
    public_repair_metadata,
    reframe_small_face_region,
    sharpen_face_region,
)

# ── Resemblance Agent Constants ───────────────────────────

MAX_RESEMBLANCE_ITERATIONS = 3
RESEMBLANCE_THRESHOLD = 8

# Controlled production pipeline: generate a small candidate set, judge each
# candidate with structured criteria, then repair/select instead of letting a
# free-form revision loop wander.
PIPELINE_CANDIDATE_COUNT = 3
QUALITY_ACCEPT_THRESHOLD = 8
IDENTITY_PASS_THRESHOLD = 8
IDENTITY_REPAIR_THRESHOLD = 7
IDENTITY_THRESHOLD_PROFILES = {
    "closeup": {
        "identity_pass_threshold": 8.0,
        "identity_repair_threshold": 7.0,
        "rationale": "large visible face; strict identity preservation",
    },
    "medium": {
        "identity_pass_threshold": 7.5,
        "identity_repair_threshold": 6.5,
        "rationale": "medium shot has less facial detail than close-up",
    },
    "small_face": {
        "identity_pass_threshold": 7.0,
        "identity_repair_threshold": 6.0,
        "rationale": "full-body or environmental portrait has smaller face area",
    },
    "side_profile": {
        "identity_pass_threshold": 7.5,
        "identity_repair_threshold": 6.5,
        "rationale": "profile/angled faces need a calibrated non-frontal threshold",
    },
}
IDENTITY_COSINE_ACCEPT_THRESHOLD = 0.45
MAX_PIPELINE_REGENERATIONS = 1
MAX_PIPELINE_LOCAL_EDITS = 2
MAX_PIPELINE_IDENTITY_REPAIRS = 1
MAX_PIPELINE_TOTAL_API_COST = 0.4
PIPELINE_PROMPT_VERSION = "controlled_candidate_v3"
IDENTITY_TEMPLATE_VERSION = "identity_pack_v2"
COMPOSITION_SCAFFOLD_PROMPT_VERSION = "composition_scaffold_v1"
COMPOSITION_FIRST_SHOT_IDS = {
    "half_body",
    "full_body",
    "environmental",
    "seated",
    "profile",
    "candid",
}
PIPELINE_ALLOWED_ACTIONS = [
    "ACCEPT",
    "LOCAL_EDIT",
    "IDENTITY_REPAIR",
    "REGENERATE_FROM_ORIGINAL",
    "REGENERATE_WITH_POSE_REFERENCE",
    "DROP_CANDIDATE",
    "REQUEST_BETTER_REFERENCE",
]

# Nano Banana 2 (google/gemini-3.1-flash-image) supports up to 4
# character references via the API. Sending more confuses identity anchoring
# and wastes input tokens; cap reference photos to the first 4 uploads.
MAX_CHARACTER_REFERENCES = 4
MAX_IDENTITY_PACK_REFERENCES = 6

REFERENCE_ROLE_SPECS = [
    {
        "role": "front_neutral",
        "angle_hint": "front",
        "expression_hint": "neutral",
        "usage": "primary_identity",
        "priority": 1,
    },
    {
        "role": "front_smile",
        "angle_hint": "front",
        "expression_hint": "smile",
        "usage": "primary_identity",
        "priority": 2,
    },
    {
        "role": "left_45",
        "angle_hint": "left_45",
        "expression_hint": "natural",
        "usage": "identity_and_pose_coverage",
        "priority": 3,
    },
    {
        "role": "right_45",
        "angle_hint": "right_45",
        "expression_hint": "natural",
        "usage": "identity_and_pose_coverage",
        "priority": 4,
    },
    {
        "role": "lifestyle",
        "angle_hint": "unconstrained",
        "expression_hint": "natural",
        "usage": "quality_evaluation_only",
        "priority": 5,
    },
    {
        "role": "side_profile",
        "angle_hint": "profile",
        "expression_hint": "natural",
        "usage": "quality_evaluation_only",
        "priority": 6,
    },
]

RESEMBLANCE_JUDGE_PROMPT = """\
你现在是严格的人脸身份审核系统。请像证件照比对或人脸识别一样，严格判断第一张图片（AI生成）中的人是否是第二张图片（用户本人参考照片）中的同一个人。

不要宽容、不要给面子。AI生成的图片常常会有"看起来像但仔细一看不是本人"的问题，你必须抓出来。

请逐项核对以下面部特征，任何一项有明显差异，评分就不能超过6分：
1. 脸型轮廓（圆脸/瓜子脸/方脸/长脸，下颌线角度）
2. 眼睛（大小、形状、单双眼皮、眼距、眼角形状）
3. 鼻子（鼻梁高度、鼻翼宽度、鼻头形状、鼻孔露出程度）
4. 嘴巴（嘴唇厚度、嘴角上扬/下垂、笑容弧度、牙齿露出情况）
5. 眉毛（粗细、弧度、眉峰位置、眉头间距）
6. 发型与发际线（头发长度、卷曲度、分线方向、发际线高低）
7. 肤色与肤质
8. 明显标记（痣、疤痕、眼镜款式、胡须、酒窝等）

评分标准（非常严格）：
- 10分：完全一致，像双胞胎
- 8-9分：明显是同一个人，只是角度/表情/光线不同
- 6-7分：有点像，但熟人可能需要多看几眼才能确认
- 5分及以下：不像，或者明显不是同一个人

输出格式必须如下：
评分：X/10
判断理由：...

如果评分低于8分，请在"判断理由"中具体列出差异最大的2-3个面部特征，并给出修改建议。"""

RESEMBLANCE_REVISION_PROMPT_TEMPLATE = """\
根据你刚才的分析，请重新生成一张图片，必须让生成的人脸与参考照片中的本人一致：

{feedback}

这是修订要求，必须遵守：
- 修改第一张图片（你刚才生成的图）的面部特征，使其与第二张参考照片中的人是同一个人
- 只修改脸，保持风格、构图、服装、背景、灯光、姿势完全不变
- 这是身份修正，不是风格修正。必须让熟悉参考照片的人一眼认出是同一个人
- 自然皮肤质感，不要美颜滤镜，不要磨皮
- 如果参考照片戴眼镜，生成图也必须戴同款眼镜；如果参考照片没戴，生成图也不能凭空添加
- 如果参考照片有痣、疤痕等明显标记，必须保留在相同位置"""

GENERIC_REVISION_PROMPT = (
    "请重新生成，更加注意保持与参考照片中人物的面部特征一致。"
    "保持风格、构图、背景不变。"
)


def build_generation_prompt(base_prompt: str, num_user_photos: int) -> str:
    """Wrap a style prompt with explicit identity-preservation instructions.

    The message sent to the model contains images in this order:
      - Image 1: the style template
      - Images 2..N: user reference photos (all the same person)

    We explicitly label each image so the model does not confuse the style
    reference with the identity reference, and we repeatedly anchor it on the
    user's face rather than the template's face.
    """
    return f"""\
The supplied identity reference images all show the SAME PERSON. This is the
only subject whose face and identity you may use. A style reference may also be
supplied; use it only for visual language such as color, light, wardrobe, and
environment. The written ShotSpec controls framing, camera angle, and pose.

Instruction:
{base_prompt}

Critical requirements:
- Generate the person from the identity references, never the style-reference person.
- Preserve the user's exact facial features, face shape, eyes, nose, mouth, eyebrows, skin tone, and overall identity.
- Do not copy framing or pose from a style reference; follow the written ShotSpec.
- Natural skin texture, no beauty filter, no plastic skin.
- Photorealistic, professional portrait quality.
- If the user is wearing different clothing in the identity references, follow the requested wardrobe in the written style details."""


def build_editing_prompt(base_prompt: str, num_user_photos: int) -> str:
    """Edit-framing wrapper: user selfie first, style template last.

    Experiments (experiments/compare_identity_preservation.py) showed that
    telling the model to "edit the user photo while preserving identity"
    produces significantly better resemblance scores than template-first
    generation on Google Gemini image models.
    """
    return f"""\
The supplied identity reference images all show the user. Preserve this
person's face, expression, and identity EXACTLY. A separate style reference may
also be supplied; it is visual guidance only, never an identity or pose source.

Instruction:
Apply the requested visual style to the person in the identity references while
keeping the face and identity unchanged.

Style details:
{base_prompt}

Critical requirements:
- Keep the person's facial features, face shape, eyes, nose, mouth, eyebrows, skin tone, expression, and overall identity exactly the same as the identity references.
- Preserve natural facial asymmetry, moles, under-eye structure, pores, flyaway hairs, and small skin-tone variations instead of normalizing them.
- Follow the written ShotSpec for framing, camera angle, and pose even when a style reference has different composition.
- Use a style reference only for palette, lighting, wardrobe, environment, and texture. Do not copy its person's identity, framing, or pose.
- Do NOT generate a different person.
- Do not produce a generic fashion-model face, enlarged eyes, narrowed jaw, glass skin, plastic skin, beauty filter, or age reduction.
- Render believable camera texture, directional light, moderate depth of field, and a physically readable environment. Avoid a centered ID-photo composition, abstract fake bokeh, halo, lens flare, dreamy haze, and synthetic symmetry unless the ShotSpec explicitly requires them.
- Photorealistic editorial portrait quality that remains visibly human on close inspection."""


def build_candidate_prompt(base_prompt: str, num_user_photos: int,
                           candidate_index: int, total_candidates: int) -> str:
    """Variant of the edit prompt for candidate-set generation.

    We keep the template constraints fixed and only vary the production intent a
    little, so diversity comes from sampling without drifting into a new style.
    """
    wrapped = build_editing_prompt(base_prompt, num_user_photos)
    variant_notes = [
        "Candidate strategy: conservative identity-first result. Keep the user's face as close as possible to the reference photos while placing them beside a real window with identifiable room details and ordinary local contrast.",
        "Candidate strategy: observational editorial result. Keep identity exact, compose slightly off-center, and retain architecture, furniture, or street detail instead of a studio gradient.",
        "Candidate strategy: candid photographic result. Prioritize an unforced expression, natural asymmetry, moderate depth of field, and a physically coherent location with foreground-to-background depth.",
    ]
    note = variant_notes[(candidate_index - 1) % len(variant_notes)]
    return f"""{wrapped}

Production candidate {candidate_index}/{total_candidates}.
{note}

Do not over-beautify. Do not change age, face shape, hairstyle, glasses, facial hair, moles, or other identity markers. Reject the visual language of AI beauty portraits: no glass skin, enlarged eyes, narrow V-line jaw, empty gaze, perfect bilateral symmetry, halo, fake bokeh, or overly centered ID-photo framing."""


def order_references_for_recovery(
    reference_paths: list[str],
    action: dict | None,
    shot_spec: dict | None,
) -> tuple[list[str], list[int]]:
    """Put the most useful pose anchor first for a targeted regeneration."""
    indexes = list(range(len(reference_paths)))
    if not indexes or (action or {}).get("action") != "REGENERATE_WITH_POSE_REFERENCE":
        return list(reference_paths), indexes

    shot_id = str((shot_spec or {}).get("shot_id") or "").lower()
    # Reference intake orders front-neutral, front-expression, then angled.
    # Profile/turned shots therefore lead with the angled reference. Other
    # geometry failures still retain the complete pack but avoid crop-derived
    # duplicates being promoted over real pose evidence.
    primary = 2 if shot_id == "profile" and len(indexes) >= 3 else 0
    ordered_indexes = [primary, *[index for index in indexes if index != primary]]
    return [reference_paths[index] for index in ordered_indexes], ordered_indexes


def append_recovery_constraint(prompt: str, action: dict | None) -> str:
    """Add the diagnosed recovery target without rewriting the whole prompt."""
    constraint = str((action or {}).get("targeted_constraint") or "").strip()
    if not constraint:
        return prompt
    return (
        f"{prompt}\n\nRecovery diagnosis: "
        f"{(action or {}).get('failure_class') or 'unknown_quality'}.\n"
        f"Recovery target: {constraint}"
    )


def should_use_composition_first(
    shot_spec: dict | None,
    *,
    force_closeup: bool = False,
    backend: str | None = None,
) -> bool:
    """Use a reference-free scaffold where image-edit models resist reframing."""
    active_backend = backend or settings.gemini_backend
    if force_closeup:
        # Hero must remain identity-first. Production stage-by-stage inspection
        # showed that scaffold -> inswapper -> identity blend creates pasted
        # facial geometry and then smooths it into a generic beauty portrait.
        return False
    shot_id = str((shot_spec or {}).get("shot_id") or "").strip().lower()
    return active_backend == "siliconflow" and shot_id in COMPOSITION_FIRST_SHOT_IDS


def _composition_safe_style_direction(base_prompt: str, shot_spec: dict) -> str:
    """Keep visual styling while removing prose that biases camera geometry."""
    style_block = str(
        ((shot_spec.get("prompt_blocks") or {}).get("style_block"))
        or base_prompt
        or ""
    )
    geometry_bias_terms = (
        "headshot",
        "close-up",
        "closeup",
        "portrait of",
        "front-facing",
        "centered head",
        "head and shoulders",
        "shot on",
        " lens",
        "sharp focus",
        "makeup",
        "skin texture",
        "skin pores",
        "expression",
        "looking at",
        "looking directly",
        "gazing",
    )
    sentences = re.split(r"(?<=[.!?])\s+", style_block.strip())
    safe_sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and not any(term in sentence.lower() for term in geometry_bias_terms)
    ]
    details = " ".join(safe_sentences[:4]).strip()[:600]
    return details or "Restrained wardrobe, natural materials, and observational editorial styling."


def build_composition_scaffold_prompt(
    base_prompt: str,
    shot_spec: dict,
    candidate_index: int,
    total_candidates: int,
) -> str:
    """Build a short photographic instruction for geometry before identity write-back."""
    shot_id = str(shot_spec.get("shot_id") or "portrait").lower()
    shot_headers = {
        "closeup": "OBSERVATIONAL CHEST-UP EDITORIAL PHOTOGRAPH, vertical 3:4 photograph.",
        "half_body": "WIDE THREE-QUARTER FASHION PORTRAIT, vertical 3:4 photograph.",
        "full_body": "FULL-BODY FASHION EDITORIAL, vertical 3:4 photograph.",
        "environmental": "FULL-LENGTH ENVIRONMENTAL FASHION EDITORIAL, vertical 3:4 photograph.",
        "seated": "WIDE SEATED FASHION PORTRAIT, vertical 3:4 photograph.",
        "profile": "THREE-QUARTER PROFILE BEAUTY PORTRAIT, vertical 3:4 photograph.",
        "candid": "WIDE CANDID ENVIRONMENTAL PORTRAIT, vertical 3:4 photograph.",
    }
    framing_guards = {
        "closeup": (
            "Photograph from about two metres away at eye level. Show the complete "
            "head, shoulders, upper chest, and a small amount of torso with breathing "
            "room around the hair. Place the face slightly off-centre and turn the "
            "shoulders and face 15 to 30 degrees. The face should occupy roughly 28% "
            "to 38% of frame height. Keep recognizable architecture or furniture in "
            "focus behind the subject. This must not look like an arm's-length selfie, "
            "passport photo, beauty campaign, or centered studio headshot."
        ),
        "half_body": (
            "Intentionally compose wider than the final crop: show the complete head, "
            "shoulders, torso, both full arms and hands, and the body down to mid-thigh. "
            "Leave at least 12% clear margin below the fingertips because the identity "
            "blend stage may tighten the crop. This must not be a headshot or chest crop."
        ),
        "full_body": (
            "Show the complete person from head through both feet with breathing room "
            "above and below. Do not crop limbs."
        ),
        "environmental": (
            "Show the subject's entire body continuously from the complete head through "
            "both feet, including both arms and hands. The person should occupy about "
            "45% to 65% of the frame height while the location remains clearly readable. "
            "Use a 35mm environmental perspective from six metres away. Never crop the "
            "head, hands, legs or feet; this must not be a close-up or chest portrait."
        ),
        "seated": (
            "Make the seated posture unmistakable: show the full seat, torso, waist, both "
            "hands, knees, and generous margin below the knees. Do not crop into a shoulder-up portrait."
        ),
        "profile": (
            "Create a shoulder-up three-quarter profile with the head turned 45 to 70 "
            "degrees away from the camera. The nose must sit visibly off the facial "
            "centreline, one cheek must be more prominent than the other, and the gaze "
            "must point away from camera. This must not be a frontal or near-frontal face."
        ),
        "candid": (
            "Show a wide three-quarter environmental composition down to mid-thigh, with "
            "both hands and generous surrounding context "
            "visible. Avoid a centered headshot crop."
        ),
    }
    style_direction = _composition_safe_style_direction(base_prompt, shot_spec)
    wardrobe_continuity = {
        "half_body": "Show coordinated lower clothing continuously through mid-thigh.",
        "full_body": "Show a coordinated full-length lower garment and both shoes.",
        "environmental": "Show a coordinated full-length lower garment and both shoes.",
        "seated": "Show coordinated lower clothing continuously through both knees.",
        "candid": "Show coordinated lower clothing continuously through mid-thigh.",
    }.get(shot_id, "")
    return f"""\
{shot_headers.get(shot_id, "EDITORIAL PORTRAIT, vertical 3:4 photograph.")}
{framing_guards.get(shot_id, "Follow the requested framing and pose exactly.")}

Style: {style_direction}
Wardrobe continuity: {wardrobe_continuity or "Keep the requested wardrobe coherent."}
Pose: {shot_spec.get("pose") or "natural editorial pose"}.
Environment: {shot_spec.get("environment") or "a physically coherent real location"}.
Lighting: {shot_spec.get("lighting") or "professional portrait light"}.
Camera: {shot_spec.get("lens") or "natural portrait perspective"}.

Exactly one clearly visible adult subject. Natural anatomy, realistic hands,
clothing and background, photorealistic commercial quality. No text, logos,
borders, collage or extra people. Inspect the whole canvas before returning.

FINAL FRAMING CHECK:
{framing_guards.get(shot_id, "Follow the requested framing and pose exactly.")}"""


def build_identity_blend_prompt(
    shot_spec: dict,
    identity_attributes: dict | None = None,
) -> str:
    """Refine a low-resolution local identity write without losing geometry."""
    target = {
        key: shot_spec.get(key)
        for key in ("shot_id", "framing", "pose", "environment", "lighting", "lens")
        if shot_spec.get(key)
    }
    attribute_contract = EvaluationService.identity_attribute_contract(
        identity_attributes
    )
    return f"""\
Perform a face-region identity blend on Image 1. Images 2 and 3 show the same
real person whose identity must appear in Image 1. Refine only the face so a
close friend would immediately recognize that person. Restore natural
high-resolution eyes, nose, lips, brows, skin pores, and face contours. Blend
the face cleanly into Image 1's existing hair, neck, lighting, and color.

Approved ShotSpec that must remain unchanged:
{json.dumps(target, ensure_ascii=False, sort_keys=True)}

{attribute_contract}

Hard preservation constraints:
- Keep Image 1's canvas, framing, crop, pose, body, hands, wardrobe, background,
  camera angle, lighting, and every non-face region as close as possible.
- Preserve every visible hand completely and retain the empty margin below it.
- Do not turn this portrait into a headshot or tighter crop.
- Do not change hairstyle, body proportions, clothing, or scene.
- Preserve the reference person's real eye size, face width, jaw width, nose-to-mouth
  spacing, facial asymmetry, under-eye structure, moles, and apparent age. Do not
  normalize the face toward a beauty-model average.
- Preserve ordinary skin micro-contrast and uneven tone. Do not synthesize repeated
  pore noise or erase texture with a smooth skin layer.
- No beauty filter, plastic skin, blur, text, borders, or extra people."""


def build_local_edit_prompt(judgement: dict) -> str:
    """Prompt for fixing only local artifacts on an otherwise viable candidate."""
    failures = ", ".join(judgement.get("hard_failures") or [])
    notes = str(judgement.get("notes") or "")
    return f"""\
Perform a LOCAL_EDIT only. Fix the visible local artifact(s) in the current generated portrait without changing the person's identity, face shape, age, expression, pose, camera angle, clothing style, background, lighting, framing, or overall composition.

Detected local issues: {failures or "minor local artifact"}
QA notes: {notes[:300]}

Allowed fixes:
- restore naturally sharp eyes, lashes, brows, lips, and skin detail when the face alone is soft
- hand or finger artifact cleanup
- collar / hair / accessory glitch cleanup
- small background stray-object cleanup
- minor expression or gaze correction
- subtle local color cleanup

Do not regenerate a new person. Do not change the scene. Do not make a full-body/pose/background/style change. Preserve realistic skin texture."""


def should_prefer_local_sharpness_edit(
    judgement: dict,
    identity_thresholds: dict,
) -> bool:
    """Keep a correct composition when its only defect is local face softness."""
    failures = set(judgement.get("hard_failures") or [])
    local_quality = judgement.get("local_quality") or {}
    local_failures = set(local_quality.get("hard_failures") or [])
    local_scores = local_quality.get("scores") or {}
    measurements = local_quality.get("measurements") or {}
    scores = judgement.get("scores") or {}
    quality_accept_threshold = float(
        identity_thresholds.get("quality_accept_threshold")
        or QUALITY_ACCEPT_THRESHOLD
    )
    return (
        failures <= {"too_blurry"}
        and local_failures <= {"too_blurry"}
        and float(scores.get("identity") or 0)
        >= float(identity_thresholds.get("identity_pass_threshold") or 8)
        and float(scores.get("style_match") or 0) >= 8
        and float(local_scores.get("face_quality") or 10)
        < quality_accept_threshold
        and measurements.get("sharpness_metric_source") in {
            "face_crop", "face_crop_256",
        }
        and float(measurements.get("sharpness_value") or 0) < 100
    )


def should_prefer_local_reframe(
    judgement: dict,
    identity_thresholds: dict,
    shot_spec: dict,
) -> bool:
    """Crop an otherwise approved close portrait when its face is too small."""
    failures = set(judgement.get("hard_failures") or [])
    local_quality = judgement.get("local_quality") or {}
    local_failures = set(local_quality.get("hard_failures") or [])
    measurements = local_quality.get("measurements") or {}
    scores = judgement.get("scores") or {}
    face_range = measurements.get("face_area_range") or []
    face_ratio = measurements.get("face_area_ratio")
    return (
        shot_spec.get("shot_id") in {"closeup", "profile"}
        and failures == {"face_scale_unusual"}
        and local_failures == {"face_scale_unusual"}
        and float(scores.get("identity") or 0)
        >= float(identity_thresholds.get("identity_pass_threshold") or 8)
        and float(scores.get("style_match") or 0) >= 8
        and len(face_range) == 2
        and face_ratio is not None
        and float(face_ratio) < float(face_range[0])
        and int(measurements.get("face_count") or 0) == 1
    )


def build_identity_pack_metadata(photo_paths: list[str]) -> dict:
    """Create task-local identity-pack metadata without storing embeddings."""
    references = []
    for idx, path in enumerate(photo_paths[:MAX_IDENTITY_PACK_REFERENCES]):
        spec = REFERENCE_ROLE_SPECS[idx] if idx < len(REFERENCE_ROLE_SPECS) else {
            "role": f"reference_{idx + 1}",
            "angle_hint": "unknown",
            "expression_hint": "unknown",
            "usage": "quality_evaluation_only",
            "priority": idx + 1,
        }
        references.append({
            "reference_id": f"ref_{idx + 1}",
            # Keep slot for older frontend/debug consumers, but role is the
            # canonical product meaning used by the pipeline.
            "slot": spec["role"],
            "role": spec["role"],
            "angle_hint": spec["angle_hint"],
            "expression_hint": spec["expression_hint"],
            "usage": spec["usage"],
            "priority": spec["priority"],
            "filename": Path(path).name,
            "input_quality_status": "checked_before_generation",
        })
    primary_reference_ids = [
        ref["reference_id"]
        for ref in references
        if ref["usage"] in {"primary_identity", "identity_and_pose_coverage"}
    ]
    payload = {
        "reference_images": references,
        "primary_reference_ids": primary_reference_ids,
        "reference_role_order": [spec["role"] for spec in REFERENCE_ROLE_SPECS],
        "appearance_constraints": {
            "apparent_age": "preserve",
            "face_shape": "preserve",
            "eye_shape": "preserve",
            "nose_shape": "preserve",
            "jawline": "preserve",
            "beautification_strength": "low",
        },
        "temporary_face_template": {
            "storage": "in_memory_task_scope",
            "stores_embedding_in_metadata": False,
            "built_from_reference_ids": primary_reference_ids,
        },
        "minimum_reference_count": 4,
        "max_identity_pack_references": MAX_IDENTITY_PACK_REFERENCES,
        "max_generation_references": MAX_CHARACTER_REFERENCES,
        "template_scope": "task_local_only",
        "cross_user_search": False,
        "persistent_face_library": False,
        "expires_at": "job_finished_plus_retention_ttl",
        "version": IDENTITY_TEMPLATE_VERSION,
    }
    return IdentityPack(**payload).model_dump(mode="json")


def identity_pack_reference_manifest(
    identity_pack: dict,
    reference_ids: set[str] | None = None,
) -> list[dict]:
    """Return invocation-safe reference metadata with no paths or embeddings."""
    manifest = []
    for ref in identity_pack.get("reference_images", []):
        if reference_ids is not None and ref.get("reference_id") not in reference_ids:
            continue
        manifest.append({
            "reference_id": ref.get("reference_id"),
            "role": ref.get("role") or ref.get("slot"),
            "angle_hint": ref.get("angle_hint"),
            "usage": ref.get("usage"),
            "priority": ref.get("priority"),
        })
    return manifest


def identity_threshold_profile(shot_spec: dict | None = None) -> dict:
    """Select initial identity thresholds for the current shot geometry."""
    shot_spec = shot_spec or {}
    shot_id = str(shot_spec.get("shot_id") or "").lower()
    framing = str(shot_spec.get("framing") or "").lower()
    pose = str(shot_spec.get("pose") or "").lower()
    text = " ".join([shot_id, framing, pose])

    profile_id = "closeup"
    if any(token in text for token in ("profile", "side", "45")):
        profile_id = "side_profile"
    if any(token in text for token in ("half_body", "medium", "waist", "street_medium")):
        profile_id = "medium"
    if any(token in text for token in ("full_body", "full body", "environmental", "small face")):
        profile_id = "small_face"

    profile = dict(IDENTITY_THRESHOLD_PROFILES[profile_id])
    profile["profile"] = profile_id
    profile["shot_id"] = shot_spec.get("shot_id")
    return profile


def model_task_type_for_shot(
    shot_spec: dict | None = None,
    *,
    force_closeup: bool = False,
) -> str:
    """Map a planned shot to the SmartModelRouter task vocabulary."""
    if force_closeup:
        return "hero_face"
    shot_spec = shot_spec or {}
    shot_id = str(shot_spec.get("shot_id") or "").lower()
    framing = str(shot_spec.get("framing") or "").lower()
    text = f"{shot_id} {framing}"
    if "closeup" in text or "close-up" in text or "head and shoulders" in text:
        return "hero_face"
    if "environmental" in text or "wide" in text:
        return "environmental"
    if "full_body" in text or "full body" in text:
        return "full_body"
    return "half_body"


def build_shot_spec_metadata(prompt: str, title: str, template_path: str | None) -> dict:
    """Represent the current job as a single planned shot.

    The MVP still creates one job per selected template/prompt. Keeping this
    shape in metadata makes the next step (multiple ShotSpec entries per job)
    additive instead of a rewrite.
    """
    payload = {
        "shot_id": title,
        "framing": "template_defined",
        "pose": "template_defined",
        "lighting": "template_defined",
        "lens": "template_defined",
        "prompt_blocks": {
            "identity_block": "derived_from_identity_pack",
            "scene_block": prompt,
            "style_block": Path(template_path).name if template_path else None,
            "preservation_block": (
                "preserve facial identity, age, face shape, skin texture, "
                "hairline, glasses/facial hair/moles when present"
            ),
        },
    }
    return ShotSpec(**payload).model_dump(mode="json")


@dataclass(frozen=True)
class _PipelineProfile:
    """Parameterization of the shared candidate pipeline.

    Captures every behavioural difference between the hero preview and the
    full-set quality pipeline so both can share one skeleton
    (``_execute_candidate_pipeline``). Tunable knobs + labelling only; the
    control flow (sample → judge → budgeted regenerate / local_edit /
    identity_repair → select → delivery gate) is identical and lives once.
    """

    # ── Identity / naming ───────────────────────────────────────
    prompt_version: str                       # metadata["pipeline"] + strategy.prompt_version

    # ── Budget knobs ────────────────────────────────────────────
    candidate_count: int
    max_regenerations: int
    max_local_edits: int
    max_identity_repairs: int
    max_total_api_cost: float

    # ── Shot selection ──────────────────────────────────────────
    force_closeup: bool                       # hero forces shot_id="closeup"

    # ── Per-candidate identifier / title prefixes ───────────────
    candidate_id_prefix: str                  # "cand_" | "hero_cand_"   → f"{prefix}{idx}"
    create_inv_id_prefix: str                 # "create_" | "hero_create_"
    regenerate_inv_id_prefix: str             # "regenerate_" | "hero_regenerate_"
    local_edit_inv_id_prefix: str             # "local_edit_" | "hero_local_edit_"
    identity_repair_inv_id_prefix: str        # "identity_repair_" | "hero_identity_repair_"
    candidate_title_suffix: str               # "_cand" | "_hero_cand"  → f"{title}{suffix}{idx}"
    regenerate_title_suffix: str              # "_regen" | "_hero_regen"

    # ── Behavioural switches ────────────────────────────────────
    drop_on_max_regenerations: bool           # full-set drops; hero does not
    # When the local edit blows the budget, the full-set pipeline annotates the
    # action with skip_reason; hero skips silently. Net effect (edit not applied)
    # is identical either way.
    mark_local_edit_skip_reason: bool

    # ── Messages ────────────────────────────────────────────────
    budget_exhausted_message: str
    no_repair_message: str

    # ── Progress-callback wording (human-facing only) ───────────
    progress_candidate_word: str              # "candidate" | "hero preview"
    progress_regenerate_word: str             # "from original references" | "hero preview"
    progress_repair_repair: str
    progress_repair_local_edit: str
    progress_repair_accept: str
    final_detail_prefix: str                  # "Selected candidate" | "Hero preview selected candidate"


class GeminiWorker:
    """Wraps the image-generation gateway with session-aware pipeline management."""

    def __init__(self, learning_layer: LearningLayer | None = None):
        self.active_session_id: str | None = None
        self._turn_counts: dict[str, int] = {}
        self._identity_attribute_profiles: dict[str, dict] = {}
        self._learning_layer = learning_layer or LearningLayer()
        self._eval_service = EvaluationService(learning_layer=self._learning_layer)
        self._agent_router = PolicyEngine(
            agent_router=AgentRouter(identity_threshold_profile),
            learning_layer=self._learning_layer,
        )
        self._episode_recovery = EpisodeRecoveryPlanner()
        self._face_swap_repair = FaceSwapRepair(
            analysis_app_factory=self._eval_service._get_identity_app,
            analysis_app_release=self._eval_service.release_identity_app,
        )
        self._gateway = ImageGateway()
        self.provider_readiness: dict | None = None

    def _plan_episode_recovery(
        self,
        action: dict,
        agent_actions: list[dict],
        *,
        allow_alternate: bool = True,
    ) -> dict:
        planner = getattr(self, "_episode_recovery", None)
        if planner is None:
            planner = EpisodeRecoveryPlanner()
            self._episode_recovery = planner
        has_alternate = getattr(self._gateway, "has_recovery_route", None)
        alternate_available = (
            allow_alternate
            and bool(has_alternate())
            if callable(has_alternate)
            else False
        )
        return planner.plan(
            action,
            agent_actions,
            alternate_route_available=alternate_available,
        )

    def _replan_recovery_before_execution(
        self,
        selected: dict,
        agent_actions: list[dict],
        *,
        state: str,
        allow_alternate: bool = True,
    ) -> dict:
        """Prevent a re-selected candidate from repeating a spent strategy."""
        current = selected.get("agent_action", {}) or {}
        planned = self._plan_episode_recovery(
            current,
            agent_actions,
            allow_alternate=allow_alternate,
        )
        selected["agent_action"] = planned

        signature_fields = ("action", "route_mode", "recovery_strategy")
        current_signature = tuple(current.get(field) for field in signature_fields)
        planned_signature = tuple(planned.get(field) for field in signature_fields)
        if planned_signature != current_signature:
            is_terminal = planned.get("action") not in {
                "REGENERATE_FROM_ORIGINAL",
                "REGENERATE_WITH_POSE_REFERENCE",
            }
            agent_actions.append({
                **planned,
                "candidate_id": selected.get("candidate_id"),
                "candidate_index": selected.get("index"),
                "state": state,
                "executed": is_terminal,
                "selected_for_execution": True,
            })
        return planned

    def _record_recovery_outcome(
        self,
        *,
        trigger_action: dict,
        judgement: dict,
        identity_thresholds: dict,
        shot_spec: dict,
        model: str | None,
        cost: float | None,
    ) -> bool:
        learning = getattr(self, "_learning_layer", None)
        record = getattr(learning, "record_pipeline_outcome", None)
        if not callable(record):
            return False
        try:
            gate = self._eval_service._candidate_gate_status(
                judgement,
                identity_thresholds,
            )
            scores = judgement.get("scores", {}) or {}
            record(
                failure_class=str(
                    trigger_action.get("failure_class") or "unknown_quality"
                ),
                action=str(trigger_action.get("action") or "UNKNOWN"),
                strategy=str(
                    trigger_action.get("recovery_strategy") or "unspecified"
                ),
                route_mode=str(trigger_action.get("route_mode") or "primary"),
                model=model,
                shot_profile=str(
                    shot_spec.get("shot_type")
                    or shot_spec.get("shot_id")
                    or "default"
                ),
                passed=bool(gate.get("hard_gates_pass")),
                identity_score=scores.get("identity"),
                quality_score=self._eval_service._aggregate_quality_score(judgement),
                cost=cost,
            )
            return True
        except Exception as exc:
            print(f"⚠ Recovery outcome recording skipped: {exc}")
            return False

    def connect(self):
        """Validate the provider is ready. Called during startup."""
        self._gateway.end_session()
        self.provider_readiness = self._gateway.check_readiness()

    def disconnect(self):
        """Clean up provider state."""
        self._gateway.end_session()

    # ── Face-swap post-processing (delegated to repair module) ──

    def _apply_face_swap(
        self,
        generated_path: str,
        photo_paths: list[str],
        title: str,
    ):
        """Swap the user's face onto the generated portrait."""
        return self._face_swap_repair.apply(generated_path, photo_paths, title)

    # ── Single-shot generation (legacy, kept for fallback) ────

    def execute_generate(
        self,
        session_id: str,
        prompt: str,
        photo_path: str,
        title: str,
        template_path: str | None = None,
    ) -> str:
        """Run a generation job. Returns path to saved image."""
        self._ensure_session(session_id, photo_path)

        self._turn_counts[session_id] = 1
        wrapped_prompt = build_editing_prompt(prompt, num_user_photos=1)
        filepath = self._gateway.create_from_references(
            prompt=wrapped_prompt,
            reference_paths=[photo_path],
            template_path=template_path,
            title=title,
            editing_mode=True,
        )
        return filepath

    # ── Hero Preview pipeline (simplified, fast, identity-first) ──

    HERO_PREVIEW_CANDIDATE_COUNT = 1
    # FLUX identity quality is high but sample variance is material. Keep one
    # initial call, then allow two failure-only rescue samples from the same
    # complete identity pack; accepted candidates stop the loop immediately.
    HERO_PREVIEW_MAX_REGENERATIONS = 2
    HERO_PREVIEW_MAX_LOCAL_EDITS = 1
    # Identity repair is conditional and every repaired stage is re-judged.
    # Keeping one repair in the action budget lets the Agent execute its own
    # diagnosis while variant selection can still retain the more-real source.
    HERO_PREVIEW_MAX_IDENTITY_REPAIRS = 1
    HERO_PREVIEW_MAX_TOTAL_API_COST = 0.6
    HERO_PREVIEW_PROMPT_VERSION = "hero_preview_v3_identity_first_flux"

    def execute_hero_preview(
        self,
        session_id: str,
        prompt: str,
        photo_paths: list[str],
        title: str,
        template_path: str | None = None,
        progress_callback=None,
        shot_spec_metadata: dict | None = None,
        session_feedback: list[dict] | None = None,
    ) -> tuple[str, dict]:
        """Generate a single high-quality hero preview with a simplified pipeline.

        Hero preview is the Aha Moment: one close-up portrait optimized for
        identity preservation and immediate user satisfaction. It uses a
        stripped-down pipeline (fewer candidates, 1 regen/repair max) to
        minimize cost while maximizing first-impression quality.

        ``session_feedback`` is the session's prior user feedback (event-tagged
        records) used to condition the policy engine — see Task 2 wiring.
        """
        profile = _PipelineProfile(
            prompt_version=self.HERO_PREVIEW_PROMPT_VERSION,
            candidate_count=self.HERO_PREVIEW_CANDIDATE_COUNT,
            max_regenerations=self.HERO_PREVIEW_MAX_REGENERATIONS,
            max_local_edits=self.HERO_PREVIEW_MAX_LOCAL_EDITS,
            max_identity_repairs=self.HERO_PREVIEW_MAX_IDENTITY_REPAIRS,
            max_total_api_cost=self.HERO_PREVIEW_MAX_TOTAL_API_COST,
            force_closeup=True,
            candidate_id_prefix="hero_cand_",
            create_inv_id_prefix="hero_create_",
            regenerate_inv_id_prefix="hero_regenerate_",
            local_edit_inv_id_prefix="hero_local_edit_",
            identity_repair_inv_id_prefix="hero_identity_repair_",
            candidate_title_suffix="_hero_cand",
            regenerate_title_suffix="_hero_regen",
            drop_on_max_regenerations=False,
            mark_local_edit_skip_reason=False,
            budget_exhausted_message="Hero preview candidate budget exhausted before generation",
            no_repair_message="Hero preview QA accepted; no identity repair needed",
            progress_candidate_word="hero preview",
            progress_regenerate_word="hero preview",
            progress_repair_repair="Applying identity repair to hero preview…",
            progress_repair_local_edit="Applying local artifact edit…",
            progress_repair_accept="Hero preview passed QA…",
            final_detail_prefix="Hero preview selected candidate",
        )
        return self._execute_candidate_pipeline(
            profile=profile,
            session_id=session_id,
            prompt=prompt,
            photo_paths=photo_paths,
            title=title,
            template_path=template_path,
            progress_callback=progress_callback,
            shot_spec_metadata=shot_spec_metadata,
            session_feedback=session_feedback,
        )

    # ── Shared candidate pipeline (hero preview + full set) ─────

    def _judge_stage_variant(
        self,
        stage: str,
        filepath: str,
        eval_ref_photos: list[str],
        shot_spec: dict,
        identity_attributes: dict,
        identity_thresholds: dict,
        variant_filepaths: dict[str, str],
    ) -> dict:
        """Score one intermediate pipeline stage as a selectable variant.

        Every stage (scaffold, face swap, blend, edits) is judged
        independently so the final delivery can pick the most real version
        instead of the last one. A stage judge failure must never kill the
        pipeline: record a judgement-less variant that variant selection
        skips, and keep going.
        """
        try:
            judgement = self._eval_service.judge_current_candidate(
                self._gateway,
                filepath,
                eval_ref_photos,
                shot_spec=shot_spec,
                identity_attributes=identity_attributes,
            )
        except Exception as exc:
            print(f"⚠ {stage} stage judgement skipped: {exc}")
            judgement = None
        variant = {
            "stage": stage,
            "filename": Path(filepath).name,
            "judgement": judgement,
            "aggregate_score": (
                self._eval_service._aggregate_quality_score(judgement)
                if judgement is not None else None
            ),
            "gate_status": (
                self._eval_service._candidate_gate_status(
                    judgement, identity_thresholds
                )
                if judgement is not None else None
            ),
        }
        variant_filepaths[variant["filename"]] = filepath
        return variant

    def _execute_candidate_pipeline(
        self,
        profile: _PipelineProfile,
        session_id: str,
        prompt: str,
        photo_paths: list[str],
        title: str,
        template_path: str | None = None,
        progress_callback=None,
        shot_spec_metadata: dict | None = None,
        session_feedback: list[dict] | None = None,
    ) -> tuple[str, dict]:
        """Shared candidate-generation skeleton.

        Flow: sample N candidates → judge each → budgeted regenerate /
        local_edit / identity_repair on the selected candidate → final delivery
        gate. Both execute_hero_preview and execute_generate_with_quality_pipeline
        delegate here; ``profile`` carries every difference between them.

        ``session_feedback`` (event-tagged user-feedback records, e.g.
        ``{"event": "not_like_me"}``) conditions the policy engine's ACCEPT vs
        regenerate lean — this is the intra-session feedback signal that was
        previously inert (the old call passed judge-iteration history, which
        carries no ``event`` field, so the modifier was always 0).
        """
        self._ensure_session(session_id, photo_paths[0] if photo_paths else "")
        self._turn_counts[session_id] = 1

        shot_spec = shot_spec_metadata or build_shot_spec_metadata(
            prompt, title, template_path
        )
        if profile.force_closeup:
            shot_spec["shot_id"] = "closeup"
            shot_spec["hero_preview"] = True
        composition_first = should_use_composition_first(
            shot_spec,
            force_closeup=profile.force_closeup,
        )
        quality_generation = (
            settings.gemini_backend == "openrouter"
            and not composition_first
            and callable(getattr(self._gateway, "create_quality_from_references", None))
        )
        generation_operation = (
            "COMPOSITION_SCAFFOLD" if composition_first else "CREATE_FROM_REFERENCES"
        )
        generation_cap = provider_for_operation("CREATE_FROM_REFERENCES")
        # A person-containing style template can contaminate the highest-value
        # first image. The curated text prompt carries the style for Hero while
        # every available image slot remains anchored to the user's identity.
        generation_template_path = None if profile.force_closeup else template_path
        reserved_template_slots = 1 if generation_template_path else 0
        generation_reference_limit = max(
            1,
            generation_cap.max_reference_images - reserved_template_slots,
        )
        ref_photos = photo_paths[
            :min(MAX_CHARACTER_REFERENCES, generation_reference_limit)
        ]
        eval_ref_photos = photo_paths[:MAX_IDENTITY_PACK_REFERENCES]
        attribute_profiles = getattr(self, "_identity_attribute_profiles", None)
        if attribute_profiles is None:
            attribute_profiles = {}
            self._identity_attribute_profiles = attribute_profiles
        if session_id not in attribute_profiles:
            attribute_profiles[session_id] = (
                self._eval_service.extract_identity_attributes(
                    self._gateway, eval_ref_photos
                )
            )
        identity_attributes = attribute_profiles.get(session_id) or {}
        identity_attribute_contract = (
            self._eval_service.identity_attribute_contract(identity_attributes)
        )
        # Composition-first already spends two provider calls per candidate
        # (scaffold + identity blend). Start with one and allow two bounded
        # retries: a paid six-shot set must not fail because one provider sample
        # ignored its framing instruction, while the profile cost cap still
        # prevents an open-ended loop.
        candidate_count = 1 if composition_first else profile.candidate_count
        max_regenerations = (
            min(2, profile.max_regenerations)
            if composition_first else profile.max_regenerations
        )
        candidates: list[dict] = []
        provider_invocations: list[dict] = []
        agent_actions: list[dict] = []
        # filename → absolute path for every recorded variant, so variant
        # selection can point delivery back at an earlier stage's file.
        variant_filepaths: dict[str, str] = {}
        identity_pack = build_identity_pack_metadata(eval_ref_photos)
        identity_thresholds = self._eval_service._get_identity_thresholds(shot_spec)
        if profile.force_closeup:
            identity_thresholds.update({
                "quality_accept_threshold": 9.0,
                "realism_accept_threshold": 9.0,
                "commercial_accept_threshold": 9.0,
            })
        generation_task_type = model_task_type_for_shot(
            shot_spec,
            force_closeup=profile.force_closeup,
        )
        generation_routing = self._gateway.route_by_task(
            generation_task_type,
            shot_spec=shot_spec,
            budget_remaining=profile.max_total_api_cost,
        )
        if quality_generation:
            quality_route = getattr(self._gateway, "quality_route", None)
            if callable(quality_route):
                generation_routing = quality_route()
        cand_word = profile.progress_candidate_word

        metadata = {
            "pipeline": profile.prompt_version,
            # Compatibility with the old frontend/storage wording.
            "iterations": 0,
            "final_score": None,
            "history": [],
            "identity_pack": identity_pack,
            "shot_spec": shot_spec,
            "allowed_actions": PIPELINE_ALLOWED_ACTIONS,
            "budget": {
                "initial_candidates": candidate_count,
                "max_regenerations": max_regenerations,
                "max_local_edits": profile.max_local_edits,
                "max_identity_repairs": profile.max_identity_repairs,
                "max_total_api_cost": profile.max_total_api_cost,
                "initial_candidates_generated": 0,
                "regenerations_used": 0,
                "local_edits_used": 0,
                "identity_repairs_used": 0,
                "composition_identity_writes_used": 0,
                "identity_blends_used": 0,
                "estimated_cost_used": 0.0,
            },
            "strategy": {
                "candidate_count": candidate_count,
                "quality_accept_threshold": QUALITY_ACCEPT_THRESHOLD,
                "identity_pass_threshold": identity_thresholds[
                    "identity_pass_threshold"
                ],
                "identity_repair_threshold": identity_thresholds[
                    "identity_repair_threshold"
                ],
                "identity_threshold_profile": identity_thresholds,
                "reference_count": len(ref_photos),
                "prompt_version": profile.prompt_version,
                "generation_mode": (
                    "composition_first" if composition_first else "identity_first"
                ),
                "generation_operation": generation_operation,
                "template_reference_used": bool(generation_template_path),
                "identity_attribute_contract_applied": bool(identity_attributes),
                "composition_prompt_version": (
                    COMPOSITION_SCAFFOLD_PROMPT_VERSION
                    if composition_first else None
                ),
                "generation_task_type": generation_task_type,
                "generation_routing": generation_routing,
            },
            "candidates": candidates,
            "agent_actions": agent_actions,
            "provider_invocations": provider_invocations,
            "selected_candidate": None,
            "shortlist": [],
            "learning": {"strategy_outcomes_recorded": 0},
        }
        reference_ids = [
            f"ref_{idx + 1}"
            for idx in range(len(ref_photos))
        ]
        reference_manifest = identity_pack_reference_manifest(
            identity_pack,
            set(reference_ids),
        )
        identity_reference_ids = [
            f"ref_{idx + 1}" for idx in range(len(eval_ref_photos))
        ]
        identity_reference_manifest = identity_pack_reference_manifest(
            identity_pack,
            set(identity_reference_ids),
        )
        if len(eval_ref_photos) >= 3:
            # Angled/profile shots should present the 45-degree identity view
            # first. Front-first edit requests tend to straighten a correct
            # profile pose even when the prompt says to preserve geometry.
            blend_reference_indexes = (
                [2, 0] if shot_spec.get("shot_id") == "profile" else [0, 2]
            )
        else:
            blend_reference_indexes = list(range(min(2, len(eval_ref_photos))))
        blend_ref_photos = [eval_ref_photos[index] for index in blend_reference_indexes]
        blend_reference_ids = [f"ref_{index + 1}" for index in blend_reference_indexes]
        blend_reference_manifest = identity_pack_reference_manifest(
            identity_pack,
            set(blend_reference_ids),
        )

        # ── 1. Initial candidate sampling ───────────────────────────
        for idx in range(1, candidate_count + 1):
            invocation_cost = (
                float(generation_routing.get("estimated_cost") or 0)
                if quality_generation
                else estimate_cost(
                    generation_operation,
                    0 if composition_first else len(ref_photos),
                )
            )
            identity_blend_cost = (
                estimate_cost("IDENTITY_BLEND", len(blend_ref_photos))
                if composition_first else 0.0
            )
            if (
                metadata["budget"]["estimated_cost_used"]
                + invocation_cost
                + identity_blend_cost
                > metadata["budget"]["max_total_api_cost"]
            ):
                agent_actions.append({
                    "action": "DROP_CANDIDATE",
                    "reason": "max_total_api_cost_reached",
                    "candidate_id": None,
                    "candidate_index": idx,
                    "state": "BUDGET_CHECK",
                    "executed": True,
                })
                break

            if progress_callback:
                progress_callback(
                    idx, candidate_count, "candidate_generating",
                    f"Generating {cand_word} {idx}/{candidate_count}…",
                )

            candidate_prompt = (
                build_composition_scaffold_prompt(
                    prompt, shot_spec, idx, candidate_count
                )
                if composition_first
                else build_candidate_prompt(
                    (
                        prompt + "\n\n" + identity_attribute_contract
                        if identity_attribute_contract else prompt
                    ),
                    len(ref_photos), idx, candidate_count
                )
            )
            started_at = time.time()
            candidate_title = f"{title}{profile.candidate_title_suffix}{idx}"
            if composition_first:
                filepath = self._gateway.create_composition_scaffold(
                    prompt=candidate_prompt,
                    title=f"{candidate_title}_scaffold",
                )
            elif quality_generation:
                filepath = self._gateway.create_quality_from_references(
                    prompt=candidate_prompt,
                    reference_paths=ref_photos,
                    template_path=generation_template_path,
                    title=candidate_title,
                )
            else:
                filepath = self._gateway.create_from_references(
                    prompt=candidate_prompt,
                    reference_paths=ref_photos,
                    template_path=generation_template_path,
                    title=candidate_title,
                    editing_mode=True,
                )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.create_inv_id_prefix}{idx}",
                operation=generation_operation,
                prompt_version=(
                    COMPOSITION_SCAFFOLD_PROMPT_VERSION
                    if composition_first else profile.prompt_version
                ),
                reference_ids=[] if composition_first else reference_ids,
                reference_roles=[] if composition_first else reference_manifest,
                candidate_index=idx,
                parent_candidate_id=None,
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=invocation_cost,
                result_status="success",
            )
            invocation["routing_decision"] = generation_routing
            if quality_generation:
                invocation["provider"] = generation_routing.get(
                    "provider", invocation["provider"]
                )
                invocation["model"] = generation_routing.get(
                    "model", invocation["model"]
                )
                invocation["provider_capabilities"]["provider"] = invocation["provider"]
                invocation["provider_capabilities"]["model"] = invocation["model"]
                invocation["provider_capabilities"]["estimated_cost"] = invocation_cost
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + invocation_cost, 4
            )

            composition_repair = None
            scaffold_filename = None
            stage_variants: list[dict] = []
            if composition_first:
                scaffold_filename = Path(filepath).name
                # Independent per-stage scoring: the raw scaffold and the
                # post-face-swap frame join the variant pool so delivery can
                # fall back to an earlier, more real version.
                stage_variants.append(self._judge_stage_variant(
                    "composition_scaffold",
                    filepath,
                    eval_ref_photos,
                    shot_spec,
                    identity_attributes,
                    identity_thresholds,
                    variant_filepaths,
                ))
                if progress_callback:
                    progress_callback(
                        idx, candidate_count, "identity_writing",
                        f"Writing your identity into {cand_word} {idx}/{candidate_count}…",
                    )
                repair_started_at = time.time()
                swap_result = self._apply_face_swap(
                    filepath,
                    photo_paths,
                    f"{candidate_title}_identity",
                )
                composition_repair = self._public_repair_metadata(
                    "composition_identity_write", swap_result
                )
                repair_invocation = build_provider_invocation_metadata(
                    invocation_id=f"composition_identity_{idx}",
                    operation="IDENTITY_REPAIR",
                    prompt_version=None,
                    reference_ids=identity_reference_ids,
                    reference_roles=identity_reference_manifest,
                    candidate_index=idx,
                    parent_candidate_id=f"{profile.candidate_id_prefix}{idx}",
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=int((time.time() - repair_started_at) * 1000),
                    cost=estimate_cost("IDENTITY_REPAIR", len(eval_ref_photos)),
                    result_status="success" if swap_result.swapped else "failed",
                )
                provider_invocations.append(repair_invocation)
                if not swap_result.swapped:
                    raise RuntimeError(
                        "Composition scaffold was created but identity write-back failed: "
                        + swap_result.message
                    )
                filepath = str(swap_result.output_path)
                metadata["budget"]["composition_identity_writes_used"] += 1
                stage_variants.append(self._judge_stage_variant(
                    "composition_face_swap",
                    filepath,
                    eval_ref_photos,
                    shot_spec,
                    identity_attributes,
                    identity_thresholds,
                    variant_filepaths,
                ))
                if progress_callback:
                    progress_callback(
                        idx, candidate_count, "identity_blending",
                        f"Refining identity detail in {cand_word} {idx}/{candidate_count}…",
                    )
                blend_started_at = time.time()
                filepath = self._gateway.identity_blend(
                    current_image_path=filepath,
                    reference_paths=blend_ref_photos,
                    blend_prompt=build_identity_blend_prompt(
                        shot_spec, identity_attributes
                    ),
                    title=f"{candidate_title}_identity_blend",
                )
                blend_invocation = build_provider_invocation_metadata(
                    invocation_id=f"identity_blend_{idx}",
                    operation="IDENTITY_BLEND",
                    prompt_version="identity_blend_v1",
                    reference_ids=blend_reference_ids,
                    reference_roles=blend_reference_manifest,
                    candidate_index=idx,
                    parent_candidate_id=f"{profile.candidate_id_prefix}{idx}",
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=int((time.time() - blend_started_at) * 1000),
                    cost=identity_blend_cost,
                    result_status="success",
                )
                provider_invocations.append(blend_invocation)
                metadata["budget"]["estimated_cost_used"] = round(
                    metadata["budget"]["estimated_cost_used"] + identity_blend_cost,
                    4,
                )
                metadata["budget"]["identity_blends_used"] += 1
                composition_repair["identity_blend"] = {
                    "applied": True,
                    "output_filename": Path(filepath).name,
                    "prompt_version": "identity_blend_v1",
                }

            if progress_callback:
                progress_callback(
                    idx, candidate_count, "candidate_judging",
                    f"Quality scoring {cand_word} {idx}/{candidate_count}…",
                )

            judgement = self._eval_service.judge_current_candidate(
                self._gateway,
                filepath,
                eval_ref_photos,
                shot_spec=shot_spec,
                identity_attributes=identity_attributes,
            )
            candidate = {
                "index": idx,
                "candidate_id": f"{profile.candidate_id_prefix}{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._eval_service._aggregate_quality_score(judgement),
                "gate_status": self._eval_service._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "selection_profile": (
                    "hero_identity" if profile.force_closeup else "balanced"
                ),
                "repair": composition_repair,
                "composition_scaffold_filename": scaffold_filename,
                "variants": [*stage_variants, {
                    "stage": (
                        "composition_identity_blend"
                        if composition_first else "generated"
                    ),
                    "filename": Path(filepath).name,
                    "judgement": judgement,
                    "aggregate_score": self._eval_service._aggregate_quality_score(
                        judgement
                    ),
                    "gate_status": self._eval_service._candidate_gate_status(
                        judgement, identity_thresholds
                    ),
                }],
            }
            variant_filepaths[Path(filepath).name] = filepath
            action = self._agent_router.decide(
                judgement,
                budget=metadata["budget"],
                shot_spec=shot_spec,
                session_feedback=session_feedback,
                edit_count=0,
                identity_repairs=0,
                identity_thresholds=identity_thresholds,
            )
            if composition_first and should_prefer_local_sharpness_edit(
                judgement, identity_thresholds
            ):
                action = {
                    **action,
                    "action": "LOCAL_EDIT",
                    "reason": "preserve_composition_fix_face_sharpness",
                    "repair_mode": "local_face_unsharp_v1",
                }
            elif should_prefer_local_reframe(
                judgement,
                identity_thresholds,
                shot_spec,
            ):
                action = {
                    **action,
                    "action": "LOCAL_EDIT",
                    "reason": "preserve_portrait_reframe_small_face",
                    "repair_mode": "local_face_reframe_v1",
                }
            elif composition_first and action.get("action") == "IDENTITY_REPAIR":
                action = {
                    **action,
                    "action": "REGENERATE_FROM_ORIGINAL",
                    "reason": (
                        "composition_identity_write_below_gate"
                    ),
                }
            elif (
                profile.force_closeup
                and action.get("action") == "LOCAL_EDIT"
                and not action.get("repair_mode")
            ):
                action = {
                    **action,
                    "action": "REGENERATE_FROM_ORIGINAL",
                    "reason": "hero_avoids_generative_local_edit",
                }
            action = self._plan_episode_recovery(
                action,
                agent_actions,
                allow_alternate=not composition_first,
            )
            candidate["agent_action"] = action
            agent_actions.append({
                **action,
                "candidate_id": candidate["candidate_id"],
                "candidate_index": idx,
                "state": "EVALUATE",
                "executed": False,
            })
            candidates.append(candidate)
            metadata["budget"]["initial_candidates_generated"] = len(candidates)
            metadata["history"].append({
                "iteration": idx,
                "score": judgement.get("scores", {}).get("identity"),
                "feedback": judgement.get("notes"),
                "raw_response": judgement.get("raw_response", "")[:500],
                "accepted": False,
            })

        if not candidates:
            raise RuntimeError(profile.budget_exhausted_message)

        selected = self._agent_router.select_candidate(candidates)
        if selected is None:
            # Extremely defensive fallback: start_conversation succeeded at least
            # once if candidates is non-empty, so use the last candidate rather
            # than failing the paid flow after a judge-format problem.
            selected = candidates[-1]

        # ── 2. Bounded regeneration from original references ─────────
        while (
            selected.get("agent_action", {}).get("action") in {
                "REGENERATE_FROM_ORIGINAL",
                "REGENERATE_WITH_POSE_REFERENCE",
            }
            and metadata["budget"]["regenerations_used"]
            < metadata["budget"]["max_regenerations"]
        ):
            recovery_action = self._replan_recovery_before_execution(
                selected,
                agent_actions,
                state="REPLAN_BEFORE_REGENERATION",
                allow_alternate=not composition_first,
            )
            if recovery_action.get("action") not in {
                "REGENERATE_FROM_ORIGINAL",
                "REGENERATE_WITH_POSE_REFERENCE",
            }:
                break
            regen_ref_photos, regen_reference_indexes = order_references_for_recovery(
                ref_photos,
                recovery_action,
                shot_spec,
            )
            regen_reference_ids = [
                reference_ids[index] for index in regen_reference_indexes
            ]
            regen_reference_manifest = identity_pack_reference_manifest(
                identity_pack,
                set(regen_reference_ids),
            )
            use_alternate_route = (
                recovery_action.get("route_mode") == "alternate"
                and self._gateway.has_recovery_route()
            )
            recovery_routing = (
                self._gateway.recovery_route()
                if use_alternate_route
                else generation_routing
            ) or generation_routing
            regen_cost = (
                float(recovery_routing.get("estimated_cost") or 0)
                if quality_generation or use_alternate_route
                else estimate_cost(
                    generation_operation,
                    0 if composition_first else len(ref_photos),
                )
            )
            identity_blend_cost = (
                estimate_cost("IDENTITY_BLEND", len(blend_ref_photos))
                if composition_first else 0.0
            )
            if (
                metadata["budget"]["estimated_cost_used"]
                + regen_cost
                + identity_blend_cost
                > metadata["budget"]["max_total_api_cost"]
            ):
                agent_actions.append({
                    "action": "DROP_CANDIDATE",
                    "reason": "max_total_api_cost_reached",
                    "candidate_id": selected.get("candidate_id"),
                    "candidate_index": selected.get("index"),
                    "state": "BUDGET_CHECK",
                    "executed": True,
                })
                break

            for action_record in agent_actions:
                if action_record.get("candidate_id") == selected.get("candidate_id"):
                    action_record["executed"] = True
                    action_record["selected_for_execution"] = True

            metadata["budget"]["regenerations_used"] += 1
            regen_no = metadata["budget"]["regenerations_used"]
            idx = len(candidates) + 1
            max_steps = candidate_count + metadata["budget"]["max_regenerations"]
            if progress_callback:
                progress_callback(
                    idx, max_steps, "regenerating_from_original",
                    f"Regenerating {profile.progress_regenerate_word} {regen_no}/"
                    f"{metadata['budget']['max_regenerations']}…",
                )

            candidate_prompt = (
                build_composition_scaffold_prompt(
                    prompt, shot_spec, idx, max_steps
                )
                if composition_first
                else build_candidate_prompt(
                    (
                        prompt + "\n\n" + identity_attribute_contract
                        if identity_attribute_contract else prompt
                    ),
                    len(ref_photos), idx, max_steps
                )
            )
            candidate_prompt = append_recovery_constraint(
                candidate_prompt,
                recovery_action,
            )
            started_at = time.time()
            regen_title = f"{title}{profile.regenerate_title_suffix}{regen_no}"
            if composition_first:
                filepath = self._gateway.create_composition_scaffold(
                    prompt=candidate_prompt,
                    title=f"{regen_title}_scaffold",
                )
            elif use_alternate_route:
                filepath = self._gateway.create_recovery_from_references(
                    prompt=candidate_prompt,
                    reference_paths=regen_ref_photos,
                    template_path=generation_template_path,
                    title=regen_title,
                )
            elif quality_generation:
                filepath = self._gateway.create_quality_from_references(
                    prompt=candidate_prompt,
                    reference_paths=regen_ref_photos,
                    template_path=generation_template_path,
                    title=regen_title,
                )
            else:
                filepath = self._gateway.create_from_references(
                    prompt=candidate_prompt,
                    reference_paths=regen_ref_photos,
                    template_path=generation_template_path,
                    title=regen_title,
                    editing_mode=True,
                )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.regenerate_inv_id_prefix}{regen_no}",
                operation=generation_operation,
                prompt_version=(
                    COMPOSITION_SCAFFOLD_PROMPT_VERSION
                    if composition_first else profile.prompt_version
                ),
                reference_ids=[] if composition_first else regen_reference_ids,
                reference_roles=(
                    [] if composition_first else regen_reference_manifest
                ),
                candidate_index=idx,
                parent_candidate_id=selected.get("candidate_id"),
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=regen_cost,
                result_status="success",
            )
            invocation["routing_decision"] = recovery_routing
            invocation["recovery"] = {
                "failure_class": recovery_action.get("failure_class"),
                "strategy": recovery_action.get("recovery_strategy"),
                "action": recovery_action.get("action"),
                "route_mode": recovery_action.get("route_mode", "primary"),
                "reference_order": regen_reference_ids,
            }
            if quality_generation or use_alternate_route:
                invocation["provider"] = recovery_routing.get(
                    "provider", invocation["provider"]
                )
                invocation["model"] = recovery_routing.get(
                    "model", invocation["model"]
                )
                invocation["provider_capabilities"]["provider"] = invocation["provider"]
                invocation["provider_capabilities"]["model"] = invocation["model"]
                invocation["provider_capabilities"]["estimated_cost"] = regen_cost
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + regen_cost, 4
            )

            composition_repair = None
            scaffold_filename = None
            stage_variants: list[dict] = []
            if composition_first:
                scaffold_filename = Path(filepath).name
                stage_variants.append(self._judge_stage_variant(
                    "composition_scaffold",
                    filepath,
                    eval_ref_photos,
                    shot_spec,
                    identity_attributes,
                    identity_thresholds,
                    variant_filepaths,
                ))
                repair_started_at = time.time()
                swap_result = self._apply_face_swap(
                    filepath,
                    photo_paths,
                    f"{regen_title}_identity",
                )
                composition_repair = self._public_repair_metadata(
                    "composition_identity_write", swap_result
                )
                repair_invocation = build_provider_invocation_metadata(
                    invocation_id=f"composition_identity_regen_{regen_no}",
                    operation="IDENTITY_REPAIR",
                    prompt_version=None,
                    reference_ids=identity_reference_ids,
                    reference_roles=identity_reference_manifest,
                    candidate_index=idx,
                    parent_candidate_id=f"{profile.candidate_id_prefix}{idx}",
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=int((time.time() - repair_started_at) * 1000),
                    cost=estimate_cost("IDENTITY_REPAIR", len(eval_ref_photos)),
                    result_status="success" if swap_result.swapped else "failed",
                )
                provider_invocations.append(repair_invocation)
                if not swap_result.swapped:
                    raise RuntimeError(
                        "Composition scaffold was regenerated but identity write-back failed: "
                        + swap_result.message
                    )
                filepath = str(swap_result.output_path)
                metadata["budget"]["composition_identity_writes_used"] += 1
                stage_variants.append(self._judge_stage_variant(
                    "composition_face_swap",
                    filepath,
                    eval_ref_photos,
                    shot_spec,
                    identity_attributes,
                    identity_thresholds,
                    variant_filepaths,
                ))
                blend_started_at = time.time()
                filepath = self._gateway.identity_blend(
                    current_image_path=filepath,
                    reference_paths=blend_ref_photos,
                    blend_prompt=build_identity_blend_prompt(
                        shot_spec, identity_attributes
                    ),
                    title=f"{regen_title}_identity_blend",
                )
                blend_invocation = build_provider_invocation_metadata(
                    invocation_id=f"identity_blend_regen_{regen_no}",
                    operation="IDENTITY_BLEND",
                    prompt_version="identity_blend_v1",
                    reference_ids=blend_reference_ids,
                    reference_roles=blend_reference_manifest,
                    candidate_index=idx,
                    parent_candidate_id=f"{profile.candidate_id_prefix}{idx}",
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=int((time.time() - blend_started_at) * 1000),
                    cost=identity_blend_cost,
                    result_status="success",
                )
                provider_invocations.append(blend_invocation)
                metadata["budget"]["estimated_cost_used"] = round(
                    metadata["budget"]["estimated_cost_used"] + identity_blend_cost,
                    4,
                )
                metadata["budget"]["identity_blends_used"] += 1
                composition_repair["identity_blend"] = {
                    "applied": True,
                    "output_filename": Path(filepath).name,
                    "prompt_version": "identity_blend_v1",
                }

            judgement = self._eval_service.judge_current_candidate(
                self._gateway,
                filepath,
                eval_ref_photos,
                shot_spec=shot_spec,
                identity_attributes=identity_attributes,
            )
            if self._record_recovery_outcome(
                trigger_action=recovery_action,
                judgement=judgement,
                identity_thresholds=identity_thresholds,
                shot_spec=shot_spec,
                model=invocation.get("model"),
                cost=regen_cost,
            ):
                metadata["learning"]["strategy_outcomes_recorded"] += 1
            candidate = {
                "index": idx,
                "candidate_id": f"{profile.candidate_id_prefix}{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._eval_service._aggregate_quality_score(judgement),
                "gate_status": self._eval_service._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "selection_profile": (
                    "hero_identity" if profile.force_closeup else "balanced"
                ),
                "repair": composition_repair,
                "composition_scaffold_filename": scaffold_filename,
                "regenerated_from_candidate_id": selected.get("candidate_id"),
                "recovery": invocation["recovery"],
                "variants": [*stage_variants, {
                    "stage": (
                        "composition_identity_blend"
                        if composition_first else "regenerated"
                    ),
                    "filename": Path(filepath).name,
                    "judgement": judgement,
                    "aggregate_score": self._eval_service._aggregate_quality_score(
                        judgement
                    ),
                    "gate_status": self._eval_service._candidate_gate_status(
                        judgement, identity_thresholds
                    ),
                }],
            }
            variant_filepaths[Path(filepath).name] = filepath
            action = self._agent_router.decide(
                judgement,
                budget=metadata["budget"],
                shot_spec=shot_spec,
                session_feedback=session_feedback,
                edit_count=0,
                identity_repairs=0,
                identity_thresholds=identity_thresholds,
            )
            if composition_first and should_prefer_local_sharpness_edit(
                judgement, identity_thresholds
            ):
                action = {
                    **action,
                    "action": "LOCAL_EDIT",
                    "reason": "preserve_composition_fix_face_sharpness",
                    "repair_mode": "local_face_unsharp_v1",
                }
            elif should_prefer_local_reframe(
                judgement,
                identity_thresholds,
                shot_spec,
            ):
                action = {
                    **action,
                    "action": "LOCAL_EDIT",
                    "reason": "preserve_portrait_reframe_small_face",
                    "repair_mode": "local_face_reframe_v1",
                }
            elif composition_first and action.get("action") == "IDENTITY_REPAIR":
                action = {
                    **action,
                    "action": "REGENERATE_FROM_ORIGINAL",
                    "reason": (
                        "composition_identity_write_below_gate"
                    ),
                }
            elif (
                profile.force_closeup
                and action.get("action") == "LOCAL_EDIT"
                and not action.get("repair_mode")
            ):
                action = {
                    **action,
                    "action": "REGENERATE_FROM_ORIGINAL",
                    "reason": "hero_avoids_generative_local_edit",
                }
            action = self._plan_episode_recovery(
                action,
                agent_actions,
                allow_alternate=not composition_first,
            )
            candidate["agent_action"] = action
            agent_actions.append({
                **action,
                "candidate_id": candidate["candidate_id"],
                "candidate_index": idx,
                "state": "EVALUATE",
                "executed": False,
            })
            candidates.append(candidate)
            metadata["history"].append({
                "iteration": idx,
                "score": judgement.get("scores", {}).get("identity"),
                "feedback": judgement.get("notes"),
                "raw_response": judgement.get("raw_response", "")[:500],
                "accepted": False,
                "regenerated_from_candidate_id": selected.get("candidate_id"),
            })

            selected = self._agent_router.select_candidate(candidates) or candidate

        if (
            profile.drop_on_max_regenerations
            and selected.get("agent_action", {}).get("action") in {
                "REGENERATE_FROM_ORIGINAL",
                "REGENERATE_WITH_POSE_REFERENCE",
            }
            and metadata["budget"]["regenerations_used"]
            >= metadata["budget"]["max_regenerations"]
        ):
            agent_actions.append({
                "action": "DROP_CANDIDATE",
                "reason": "max_regenerations_reached",
                "candidate_id": selected.get("candidate_id"),
                "candidate_index": selected.get("index"),
                "state": "BUDGET_CHECK",
                "executed": True,
                "selected_for_execution": True,
            })

        # ── 3. Local edit / identity repair of the selected candidate ─
        selected["selected"] = True
        filepath = selected["path"]
        selected_scores = selected.get("judgement", {}).get("scores", {})
        selected_identity = selected_scores.get("identity")
        selected_action = selected.get("agent_action", {})
        local_edit_needed = selected_action.get("action") == "LOCAL_EDIT"
        identity_repair_budget_available = (
            metadata["budget"]["identity_repairs_used"]
            < metadata["budget"]["max_identity_repairs"]
        )
        repair_needed = (
            not composition_first
            and identity_repair_budget_available
            and (
                selected_action.get("action") == "IDENTITY_REPAIR"
                or self._agent_router.should_apply_identity_repair(
                    selected.get("judgement", {}),
                    identity_thresholds,
                )
            )
        )
        for action_record in agent_actions:
            if action_record.get("candidate_id") == selected.get("candidate_id"):
                action_record["selected_for_execution"] = True

        if progress_callback:
            progress_callback(
                selected["index"], candidate_count, "repairing",
                (
                    profile.progress_repair_repair
                    if repair_needed
                    else profile.progress_repair_local_edit
                    if local_edit_needed
                    else profile.progress_repair_accept
                ),
            )

        if local_edit_needed:
            repair_mode = selected_action.get("repair_mode")
            deterministic_sharpen = repair_mode == "local_face_unsharp_v1"
            deterministic_reframe = repair_mode == "local_face_reframe_v1"
            deterministic_edit = deterministic_sharpen or deterministic_reframe
            local_edit_refs = (
                []
                if deterministic_edit
                else blend_ref_photos if composition_first else ref_photos
            )
            local_edit_reference_ids = (
                []
                if deterministic_edit
                else blend_reference_ids if composition_first else reference_ids
            )
            local_edit_reference_manifest = (
                []
                if deterministic_edit
                else blend_reference_manifest if composition_first else reference_manifest
            )
            local_edit_cost = (
                0.0
                if deterministic_edit
                else estimate_cost("LOCAL_EDIT", len(local_edit_refs))
            )
            if (
                metadata["budget"]["estimated_cost_used"] + local_edit_cost
                > metadata["budget"]["max_total_api_cost"]
            ):
                local_edit_needed = False
                if profile.mark_local_edit_skip_reason:
                    for action_record in agent_actions:
                        if action_record.get("candidate_id") == selected.get("candidate_id"):
                            action_record["executed"] = False
                            action_record["skip_reason"] = "max_total_api_cost_reached"

        if local_edit_needed:
            started_at = time.time()
            if deterministic_sharpen:
                source_path = Path(filepath)
                edited_path = source_path.with_name(
                    f"{source_path.stem}_face_sharp{source_path.suffix}"
                )
                filepath = str(
                    sharpen_face_region(source_path, edited_path)
                )
                local_edit_routing = {
                    "provider": "local",
                    "model": "opencv_face_unsharp_v1",
                    "reason": "preserve_geometry_fix_face_sharpness",
                    "estimated_cost": 0.0,
                    "estimated_latency_ms": int((time.time() - started_at) * 1000),
                    "confidence": 1.0,
                }
            elif deterministic_reframe:
                source_path = Path(filepath)
                edited_path = source_path.with_name(
                    f"{source_path.stem}_reframed{source_path.suffix}"
                )
                filepath = str(
                    reframe_small_face_region(source_path, edited_path)
                )
                local_edit_routing = {
                    "provider": "local",
                    "model": "opencv_face_reframe_v1",
                    "reason": "preserve_subject_fix_face_scale",
                    "estimated_cost": 0.0,
                    "estimated_latency_ms": int((time.time() - started_at) * 1000),
                    "confidence": 1.0,
                }
            else:
                local_edit_routing = self._gateway.route_by_task(
                    "local_edit",
                    shot_spec=shot_spec,
                    budget_remaining=(
                        metadata["budget"]["max_total_api_cost"]
                        - metadata["budget"]["estimated_cost_used"]
                    ),
                )
                edit_prompt = build_local_edit_prompt(selected.get("judgement", {}))
                filepath = self._gateway.local_edit(
                    current_image_path=filepath,
                    reference_paths=local_edit_refs,
                    edit_prompt=edit_prompt,
                    title=f"{title}{profile.candidate_title_suffix}{selected['index']}_local_edit",
                )
            metadata["budget"]["local_edits_used"] += 1
            local_invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.local_edit_inv_id_prefix}{selected['index']}",
                operation="LOCAL_EDIT",
                prompt_version=profile.prompt_version,
                reference_ids=local_edit_reference_ids,
                reference_roles=local_edit_reference_manifest,
                candidate_index=selected["index"],
                parent_candidate_id=selected["candidate_id"],
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=local_edit_cost,
                result_status="success",
            )
            if deterministic_edit:
                local_model = (
                    "opencv_face_unsharp_v1"
                    if deterministic_sharpen
                    else "opencv_face_reframe_v1"
                )
                local_invocation["provider"] = "local"
                local_invocation["model"] = local_model
                local_invocation["provider_capabilities"] = {
                    "provider": "local",
                    "model": local_model,
                    "supports_multiple_references": False,
                    "supports_mask_edit": True,
                    "supports_high_fidelity": True,
                    "supports_seed": False,
                    "supports_portrait_ratio": True,
                    "max_reference_images": 0,
                    "average_latency_ms": 250,
                    "estimated_cost": 0.0,
                    "supported_tasks": [
                        "local_edit",
                        "face_sharpness" if deterministic_sharpen else "face_reframe",
                    ],
                }
            local_invocation["routing_decision"] = local_edit_routing
            provider_invocations.append(local_invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + local_edit_cost, 4
            )
            edited_judgement = self._eval_service.judge_current_candidate(self._gateway,
                filepath,
                eval_ref_photos,
                shot_spec=shot_spec,
                identity_attributes=identity_attributes,
            )
            if self._record_recovery_outcome(
                trigger_action=selected_action,
                judgement=edited_judgement,
                identity_thresholds=identity_thresholds,
                shot_spec=shot_spec,
                model=local_invocation.get("model"),
                cost=local_edit_cost,
            ):
                metadata["learning"]["strategy_outcomes_recorded"] += 1
            selected["local_edit"] = {
                "action": "local_edit",
                "applied": True,
                "output_filename": Path(filepath).name,
                "post_edit_judgement": edited_judgement,
            }
            selected.setdefault("variants", []).append({
                "stage": "local_edit",
                "filename": Path(filepath).name,
                "judgement": edited_judgement,
                "aggregate_score": self._eval_service._aggregate_quality_score(
                    edited_judgement
                ),
                "gate_status": self._eval_service._candidate_gate_status(
                    edited_judgement, identity_thresholds
                ),
            })
            variant_filepaths[Path(filepath).name] = filepath
            selected["judgement"] = edited_judgement
            selected["aggregate_score"] = self._eval_service._aggregate_quality_score(
                edited_judgement
            )
            selected["gate_status"] = self._eval_service._candidate_gate_status(
                edited_judgement,
                identity_thresholds,
            )
            selected_scores = edited_judgement.get("scores", {})
            selected_identity = selected_scores.get("identity")
            if (
                deterministic_reframe
                and metadata["budget"]["local_edits_used"]
                < metadata["budget"]["max_local_edits"]
                and should_prefer_local_sharpness_edit(
                    edited_judgement,
                    identity_thresholds,
                )
            ):
                sharpen_started_at = time.time()
                reframed_path = Path(filepath)
                sharpened_path = reframed_path.with_name(
                    f"{reframed_path.stem}_face_sharp{reframed_path.suffix}"
                )
                filepath = str(
                    sharpen_face_region(reframed_path, sharpened_path)
                )
                metadata["budget"]["local_edits_used"] += 1
                sharpen_invocation = build_provider_invocation_metadata(
                    invocation_id=(
                        f"{profile.local_edit_inv_id_prefix}"
                        f"{selected['index']}_sharpness"
                    ),
                    operation="LOCAL_EDIT",
                    prompt_version=None,
                    reference_ids=[],
                    reference_roles=[],
                    candidate_index=selected["index"],
                    parent_candidate_id=selected["candidate_id"],
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=int((time.time() - sharpen_started_at) * 1000),
                    cost=0.0,
                    result_status="success",
                )
                sharpen_invocation.update({
                    "provider": "local",
                    "model": "opencv_face_unsharp_v1",
                    "provider_capabilities": {
                        "provider": "local",
                        "model": "opencv_face_unsharp_v1",
                        "supports_multiple_references": False,
                        "supports_mask_edit": True,
                        "supports_high_fidelity": True,
                        "supports_seed": False,
                        "supports_portrait_ratio": True,
                        "max_reference_images": 0,
                        "average_latency_ms": 250,
                        "estimated_cost": 0.0,
                        "supported_tasks": ["local_edit", "face_sharpness"],
                    },
                    "routing_decision": {
                        "provider": "local",
                        "model": "opencv_face_unsharp_v1",
                        "reason": "post_reframe_face_softness",
                        "estimated_cost": 0.0,
                        "estimated_latency_ms": int(
                            (time.time() - sharpen_started_at) * 1000
                        ),
                        "confidence": 1.0,
                    },
                })
                provider_invocations.append(sharpen_invocation)
                agent_actions.append({
                    "action": "LOCAL_EDIT",
                    "reason": "post_reframe_face_softness",
                    "repair_mode": "local_face_unsharp_v1",
                    "candidate_id": selected["candidate_id"],
                    "candidate_index": selected["index"],
                    "state": "REPAIR",
                    "executed": True,
                    "selected_for_execution": True,
                })
                edited_judgement = self._eval_service.judge_current_candidate(
                    self._gateway,
                    filepath,
                    eval_ref_photos,
                    shot_spec=shot_spec,
                    identity_attributes=identity_attributes,
                )
                selected["local_edit"]["followup_sharpness"] = {
                    "action": "local_face_unsharp",
                    "applied": True,
                    "output_filename": Path(filepath).name,
                    "post_edit_judgement": edited_judgement,
                }
                selected.setdefault("variants", []).append({
                    "stage": "post_reframe_sharpness",
                    "filename": Path(filepath).name,
                    "judgement": edited_judgement,
                    "aggregate_score": self._eval_service._aggregate_quality_score(
                        edited_judgement
                    ),
                    "gate_status": self._eval_service._candidate_gate_status(
                        edited_judgement, identity_thresholds
                    ),
                })
                variant_filepaths[Path(filepath).name] = filepath
                selected["judgement"] = edited_judgement
                selected["aggregate_score"] = (
                    self._eval_service._aggregate_quality_score(
                        edited_judgement
                    )
                )
                selected["gate_status"] = (
                    self._eval_service._candidate_gate_status(
                        edited_judgement,
                        identity_thresholds,
                    )
                )
                selected_scores = edited_judgement.get("scores", {})
                selected_identity = selected_scores.get("identity")
            repair_needed = (
                not composition_first
                and metadata["budget"]["identity_repairs_used"]
                < metadata["budget"]["max_identity_repairs"]
                and self._should_apply_identity_repair(
                    edited_judgement,
                    identity_thresholds,
                )
            )

        # Deterministic identity hardening. Use it as a conditional repair, not
        # a mandatory final pass: swapping every already-good candidate can
        # degrade expression, skin texture, or lighting.
        if repair_needed:
            swap_result = self._apply_face_swap(filepath, photo_paths, title)
            if swap_result.swapped:
                metadata["budget"]["identity_repairs_used"] += 1
                filepath = str(swap_result.output_path)
                selected["repair"] = self._public_repair_metadata(
                    "face_swap", swap_result
                )
                repair_invocation = build_provider_invocation_metadata(
                    invocation_id=f"{profile.identity_repair_inv_id_prefix}{selected['index']}",
                    operation="IDENTITY_REPAIR",
                    prompt_version=None,
                    reference_ids=reference_ids,
                    reference_roles=reference_manifest,
                    candidate_index=selected["index"],
                    parent_candidate_id=selected["candidate_id"],
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=None,
                    cost=estimate_cost("IDENTITY_REPAIR", len(ref_photos)),
                    result_status="success",
                )
                provider_invocations.append(repair_invocation)
                metadata["budget"]["estimated_cost_used"] = round(
                    metadata["budget"]["estimated_cost_used"]
                    + repair_invocation["estimated_cost"],
                    4,
                )
                # Re-score the repaired image when possible.
                repaired_judgement = self._eval_service.judge_current_candidate(self._gateway,
                    filepath,
                    eval_ref_photos,
                    shot_spec=shot_spec,
                    identity_attributes=identity_attributes,
                )
                if self._record_recovery_outcome(
                    trigger_action=selected_action,
                    judgement=repaired_judgement,
                    identity_thresholds=identity_thresholds,
                    shot_spec=shot_spec,
                    model=repair_invocation.get("model"),
                    cost=repair_invocation.get("estimated_cost"),
                ):
                    metadata["learning"]["strategy_outcomes_recorded"] += 1
                selected["repair"]["post_repair_judgement"] = repaired_judgement
                selected.setdefault("variants", []).append({
                    "stage": "identity_repair",
                    "filename": Path(filepath).name,
                    "judgement": repaired_judgement,
                    "aggregate_score": self._eval_service._aggregate_quality_score(
                        repaired_judgement
                    ),
                    "gate_status": self._eval_service._candidate_gate_status(
                        repaired_judgement, identity_thresholds
                    ),
                })
                variant_filepaths[Path(filepath).name] = filepath
                selected["judgement"] = repaired_judgement
                selected["aggregate_score"] = self._eval_service._aggregate_quality_score(
                    repaired_judgement
                )
                selected["gate_status"] = self._eval_service._candidate_gate_status(
                    repaired_judgement,
                    identity_thresholds,
                )
                selected_scores = repaired_judgement.get("scores", {})
                selected_identity = selected_scores.get("identity")
                for action_record in agent_actions:
                    if (
                        action_record.get("candidate_id") == selected.get("candidate_id")
                        and action_record.get("action") == "IDENTITY_REPAIR"
                        and action_record.get("selected_for_execution") is True
                    ):
                        action_record["executed"] = True
                if (
                    metadata["budget"]["local_edits_used"]
                    < metadata["budget"]["max_local_edits"]
                    and should_prefer_local_sharpness_edit(
                        repaired_judgement,
                        identity_thresholds,
                    )
                ):
                    sharpen_started_at = time.time()
                    repaired_path = Path(filepath)
                    sharpened_path = repaired_path.with_name(
                        f"{repaired_path.stem}_face_sharp{repaired_path.suffix}"
                    )
                    filepath = str(
                        sharpen_face_region(repaired_path, sharpened_path)
                    )
                    metadata["budget"]["local_edits_used"] += 1
                    sharpen_invocation = build_provider_invocation_metadata(
                        invocation_id=(
                            f"{profile.local_edit_inv_id_prefix}"
                            f"{selected['index']}_post_identity"
                        ),
                        operation="LOCAL_EDIT",
                        prompt_version=None,
                        reference_ids=[],
                        reference_roles=[],
                        candidate_index=selected["index"],
                        parent_candidate_id=selected["candidate_id"],
                        shot_id=shot_spec.get("shot_id"),
                        latency_ms=int((time.time() - sharpen_started_at) * 1000),
                        cost=0.0,
                        result_status="success",
                    )
                    sharpen_invocation.update({
                        "provider": "local",
                        "model": "opencv_face_unsharp_v1",
                        "provider_capabilities": {
                            "provider": "local",
                            "model": "opencv_face_unsharp_v1",
                            "supports_multiple_references": False,
                            "supports_mask_edit": True,
                            "supports_high_fidelity": True,
                            "supports_seed": False,
                            "supports_portrait_ratio": True,
                            "max_reference_images": 0,
                            "average_latency_ms": 250,
                            "estimated_cost": 0.0,
                            "supported_tasks": ["local_edit", "face_sharpness"],
                        },
                        "routing_decision": {
                            "provider": "local",
                            "model": "opencv_face_unsharp_v1",
                            "reason": "post_identity_repair_face_softness",
                            "estimated_cost": 0.0,
                            "estimated_latency_ms": int(
                                (time.time() - sharpen_started_at) * 1000
                            ),
                            "confidence": 1.0,
                        },
                    })
                    provider_invocations.append(sharpen_invocation)
                    agent_actions.append({
                        "action": "LOCAL_EDIT",
                        "reason": "post_identity_repair_face_softness",
                        "repair_mode": "local_face_unsharp_v1",
                        "candidate_id": selected["candidate_id"],
                        "candidate_index": selected["index"],
                        "state": "REPAIR",
                        "executed": True,
                        "selected_for_execution": True,
                    })
                    sharpened_judgement = self._eval_service.judge_current_candidate(
                        self._gateway,
                        filepath,
                        eval_ref_photos,
                        shot_spec=shot_spec,
                        identity_attributes=identity_attributes,
                    )
                    selected["repair"]["post_repair_sharpness"] = {
                        "action": "local_face_unsharp",
                        "applied": True,
                        "output_filename": Path(filepath).name,
                        "post_edit_judgement": sharpened_judgement,
                    }
                    selected.setdefault("variants", []).append({
                        "stage": "post_identity_sharpness",
                        "filename": Path(filepath).name,
                        "judgement": sharpened_judgement,
                        "aggregate_score": self._eval_service._aggregate_quality_score(
                            sharpened_judgement
                        ),
                        "gate_status": self._eval_service._candidate_gate_status(
                            sharpened_judgement, identity_thresholds
                        ),
                    })
                    variant_filepaths[Path(filepath).name] = filepath
                    selected["judgement"] = sharpened_judgement
                    selected["aggregate_score"] = (
                        self._eval_service._aggregate_quality_score(
                            sharpened_judgement
                        )
                    )
                    selected["gate_status"] = (
                        self._eval_service._candidate_gate_status(
                            sharpened_judgement,
                            identity_thresholds,
                        )
                    )
                    selected_scores = sharpened_judgement.get("scores", {})
                    selected_identity = selected_scores.get("identity")
            else:
                selected["repair"] = self._public_repair_metadata(
                    "face_swap", swap_result
                )
        elif not selected.get("repair"):
            selected["repair"] = {
                "action": "none",
                "applied": False,
                "message": profile.no_repair_message,
            }

        # A repair is an action, not a terminal verdict. Re-evaluate its
        # outcome and, when it introduced a new global defect (for example
        # synthetic skin), let the policy take a different bounded action.
        # This closes the old dead end where a successful identity write could
        # still fail realism and the pipeline stopped despite unused retry
        # budget. Composition-first shots already have their own staged loop;
        # this branch is the identity-first counterpart.
        repair_applied = bool(selected.get("repair", {}).get("applied"))
        if repair_applied and not selected.get("gate_status", {}).get(
            "hard_gates_pass"
        ):
            post_repair_action = self._agent_router.decide(
                selected.get("judgement", {}),
                budget=metadata["budget"],
                shot_spec=shot_spec,
                session_feedback=session_feedback,
                edit_count=metadata["budget"]["local_edits_used"],
                identity_repairs=metadata["budget"]["identity_repairs_used"],
                identity_thresholds=identity_thresholds,
            )
            post_repair_action = self._plan_episode_recovery(
                post_repair_action,
                agent_actions,
            )
            selected["agent_action"] = post_repair_action
            agent_actions.append({
                **post_repair_action,
                "candidate_id": selected["candidate_id"],
                "candidate_index": selected["index"],
                "state": "POST_REPAIR_EVALUATE",
                "executed": False,
                "selected_for_execution": True,
            })

        while (
            repair_applied
            and selected.get("agent_action", {}).get("action") in {
                "REGENERATE_FROM_ORIGINAL",
                "REGENERATE_WITH_POSE_REFERENCE",
            }
            and metadata["budget"]["regenerations_used"]
            < metadata["budget"]["max_regenerations"]
        ):
            recovery_action = self._replan_recovery_before_execution(
                selected,
                agent_actions,
                state="POST_REPAIR_REPLAN_BEFORE_REGENERATION",
            )
            if recovery_action.get("action") not in {
                "REGENERATE_FROM_ORIGINAL",
                "REGENERATE_WITH_POSE_REFERENCE",
            }:
                break
            regen_ref_photos, regen_reference_indexes = order_references_for_recovery(
                ref_photos,
                recovery_action,
                shot_spec,
            )
            regen_reference_ids = [
                reference_ids[index] for index in regen_reference_indexes
            ]
            regen_reference_manifest = identity_pack_reference_manifest(
                identity_pack,
                set(regen_reference_ids),
            )
            use_alternate_route = (
                recovery_action.get("route_mode") == "alternate"
                and self._gateway.has_recovery_route()
            )
            recovery_routing = (
                self._gateway.recovery_route()
                if use_alternate_route
                else generation_routing
            ) or generation_routing
            regen_cost = (
                float(recovery_routing.get("estimated_cost") or 0)
                if quality_generation or use_alternate_route
                else estimate_cost("CREATE_FROM_REFERENCES", len(regen_ref_photos))
            )
            if (
                metadata["budget"]["estimated_cost_used"] + regen_cost
                > metadata["budget"]["max_total_api_cost"]
            ):
                agent_actions.append({
                    "action": "DROP_CANDIDATE",
                    "reason": "max_total_api_cost_reached",
                    "candidate_id": selected.get("candidate_id"),
                    "candidate_index": selected.get("index"),
                    "state": "POST_REPAIR_BUDGET_CHECK",
                    "executed": True,
                    "selected_for_execution": True,
                })
                break

            for action_record in agent_actions:
                if (
                    action_record.get("candidate_id")
                    == selected.get("candidate_id")
                    and action_record.get("state") == "POST_REPAIR_EVALUATE"
                ):
                    action_record["executed"] = True

            metadata["budget"]["regenerations_used"] += 1
            regen_no = metadata["budget"]["regenerations_used"]
            idx = len(candidates) + 1
            max_steps = candidate_count + metadata["budget"]["max_regenerations"]
            if progress_callback:
                progress_callback(
                    idx,
                    max_steps,
                    "regenerating_after_repair",
                    f"Trying a new portrait after repair QA {regen_no}/"
                    f"{metadata['budget']['max_regenerations']}…",
                )

            candidate_prompt = build_candidate_prompt(
                (
                    prompt + "\n\n" + identity_attribute_contract
                    if identity_attribute_contract else prompt
                ),
                len(regen_ref_photos),
                idx,
                max_steps,
            )
            candidate_prompt = append_recovery_constraint(
                candidate_prompt,
                recovery_action,
            )
            started_at = time.time()
            regen_title = f"{title}{profile.regenerate_title_suffix}{regen_no}"
            if use_alternate_route:
                filepath = self._gateway.create_recovery_from_references(
                    prompt=candidate_prompt,
                    reference_paths=regen_ref_photos,
                    template_path=generation_template_path,
                    title=regen_title,
                )
            elif quality_generation:
                filepath = self._gateway.create_quality_from_references(
                    prompt=candidate_prompt,
                    reference_paths=regen_ref_photos,
                    template_path=generation_template_path,
                    title=regen_title,
                )
            else:
                filepath = self._gateway.create_from_references(
                    prompt=candidate_prompt,
                    reference_paths=regen_ref_photos,
                    template_path=generation_template_path,
                    title=regen_title,
                    editing_mode=True,
                )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.regenerate_inv_id_prefix}{regen_no}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=profile.prompt_version,
                reference_ids=regen_reference_ids,
                reference_roles=regen_reference_manifest,
                candidate_index=idx,
                parent_candidate_id=selected.get("candidate_id"),
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=regen_cost,
                result_status="success",
            )
            invocation["routing_decision"] = recovery_routing
            invocation["recovery"] = {
                "trigger_stage": "identity_repair",
                "failure_class": recovery_action.get("failure_class"),
                "strategy": recovery_action.get("recovery_strategy"),
                "action": recovery_action.get("action"),
                "route_mode": recovery_action.get("route_mode", "primary"),
                "reference_order": regen_reference_ids,
            }
            if quality_generation or use_alternate_route:
                invocation["provider"] = recovery_routing.get(
                    "provider", invocation["provider"]
                )
                invocation["model"] = recovery_routing.get(
                    "model", invocation["model"]
                )
                invocation["provider_capabilities"]["provider"] = invocation[
                    "provider"
                ]
                invocation["provider_capabilities"]["model"] = invocation["model"]
                invocation["provider_capabilities"]["estimated_cost"] = regen_cost
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + regen_cost,
                4,
            )

            judgement = self._eval_service.judge_current_candidate(
                self._gateway,
                filepath,
                eval_ref_photos,
                shot_spec=shot_spec,
                identity_attributes=identity_attributes,
            )
            if self._record_recovery_outcome(
                trigger_action=recovery_action,
                judgement=judgement,
                identity_thresholds=identity_thresholds,
                shot_spec=shot_spec,
                model=invocation.get("model"),
                cost=regen_cost,
            ):
                metadata["learning"]["strategy_outcomes_recorded"] += 1
            candidate = {
                "index": idx,
                "candidate_id": f"{profile.candidate_id_prefix}{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._eval_service._aggregate_quality_score(
                    judgement
                ),
                "gate_status": self._eval_service._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "selection_profile": (
                    "hero_identity" if profile.force_closeup else "balanced"
                ),
                "repair": {
                    "action": "none",
                    "applied": False,
                    "message": "Fresh recovery candidate from original references",
                },
                "regenerated_from_candidate_id": selected.get("candidate_id"),
                "recovery": invocation["recovery"],
                "variants": [{
                    "stage": "post_repair_regenerated",
                    "filename": Path(filepath).name,
                    "judgement": judgement,
                    "aggregate_score": self._eval_service._aggregate_quality_score(
                        judgement
                    ),
                    "gate_status": self._eval_service._candidate_gate_status(
                        judgement,
                        identity_thresholds,
                    ),
                }],
            }
            variant_filepaths[Path(filepath).name] = filepath
            action = self._agent_router.decide(
                judgement,
                budget=metadata["budget"],
                shot_spec=shot_spec,
                session_feedback=session_feedback,
                edit_count=metadata["budget"]["local_edits_used"],
                identity_repairs=metadata["budget"]["identity_repairs_used"],
                identity_thresholds=identity_thresholds,
            )
            if should_prefer_local_reframe(
                judgement,
                identity_thresholds,
                shot_spec,
            ):
                action = {
                    **action,
                    "action": "LOCAL_EDIT",
                    "reason": "post_repair_recovery_reframe_small_face",
                    "repair_mode": "local_face_reframe_v1",
                }
            action = self._plan_episode_recovery(action, agent_actions)
            candidate["agent_action"] = action
            candidates.append(candidate)
            metadata["history"].append({
                "iteration": idx,
                "score": judgement.get("scores", {}).get("identity"),
                "feedback": judgement.get("notes"),
                "raw_response": judgement.get("raw_response", "")[:500],
                "accepted": False,
                "regenerated_from_candidate_id": selected.get("candidate_id"),
                "trigger_stage": "identity_repair",
            })
            agent_actions.append({
                **action,
                "candidate_id": candidate["candidate_id"],
                "candidate_index": idx,
                "state": "POST_REPAIR_RECOVERY_EVALUATE",
                "executed": False,
            })
            selected["selected"] = False
            selected = self._agent_router.select_candidate(candidates) or candidate
            selected["selected"] = True
            filepath = selected["path"]

            # A directly deliverable fresh candidate ends the recovery loop.
            # A second regeneration remains possible when policy requests it;
            # local edits are intentionally left to the strict final gate here
            # instead of silently mutating a newly realistic face.
            if selected.get("gate_status", {}).get("hard_gates_pass"):
                break

        # ── 3.5 Cross-stage variant selection ──────────────────────
        # Every stage was scored independently above. Deliver the most real
        # version that passes the hard gates instead of assuming the last
        # repair is best. When no variant passes, keep the last-stage output
        # and let the final delivery gate below intercept it honestly.
        variants = selected.get("variants") or []
        best_variant = select_best_variant(variants)
        last_variant = variants[-1] if variants else None
        if best_variant is not None and best_variant is not last_variant:
            variant_selection_reason = "earlier_stage_more_real"
            selected_filename = str(best_variant.get("filename") or "")
            filepath = variant_filepaths.get(
                selected_filename,
                str(Path(filepath).parent / selected_filename),
            )
            selected["path"] = filepath
            selected["filename"] = Path(filepath).name
            selected["judgement"] = best_variant["judgement"]
            selected["aggregate_score"] = best_variant["aggregate_score"]
            selected["gate_status"] = best_variant["gate_status"]
            selected_scores = (best_variant["judgement"] or {}).get("scores", {})
            selected_identity = selected_scores.get("identity")
        elif best_variant is not None:
            variant_selection_reason = "last_stage_most_real"
        else:
            variant_selection_reason = "no_hard_gate_passing_variant"
        metadata["variant_selection"] = {
            "selected_stage": (best_variant or last_variant or {}).get("stage"),
            "reason": variant_selection_reason,
            "total_variants": len(variants),
        }

        # ── 4. Final delivery-gate evaluation ────────────────────────
        final_gate = selected.get("gate_status", {}) or {}
        selected_has_terminal_drop = any(
            action_record.get("candidate_id") == selected.get("candidate_id")
            and action_record.get("action") == "DROP_CANDIDATE"
            and action_record.get("executed") is True
            for action_record in agent_actions
        )
        if not final_gate.get("hard_gates_pass") and not selected_has_terminal_drop:
            local_edit_applied = selected.get("local_edit", {}).get("applied")
            repair_applied = selected.get("repair", {}).get("applied")
            if local_edit_applied or repair_applied:
                agent_actions.append({
                    "action": "DROP_CANDIDATE",
                    "reason": (
                        "local_edit_failed_delivery_gate"
                        if local_edit_applied
                        else "identity_repair_failed_delivery_gate"
                    ),
                    "candidate_id": selected.get("candidate_id"),
                    "candidate_index": selected.get("index"),
                    "state": "FINAL_EVALUATE",
                    "hard_gate_failures": final_gate.get("hard_gate_failures", []),
                    "executed": True,
                    "selected_for_execution": True,
                })

        for action_record in agent_actions:
            if action_record.get("candidate_id") != selected.get("candidate_id"):
                continue
            if action_record.get("action") == "DROP_CANDIDATE":
                continue
            action_record["executed"] = bool(
                (
                    repair_needed
                    and selected.get("repair", {}).get("applied")
                )
                or (
                    action_record.get("action") == "LOCAL_EDIT"
                    and selected.get("local_edit", {}).get("applied")
                )
            )
            if not repair_needed and action_record.get("action") == "ACCEPT":
                action_record["executed"] = True

        final_score = selected_identity
        metadata["iterations"] = len(candidates)
        metadata["final_score"] = final_score
        metadata["selected_candidate"] = {
            "index": selected["index"],
            "candidate_id": selected["candidate_id"],
            "filename": Path(filepath).name,
            "aggregate_score": selected["aggregate_score"],
            "identity_score": final_score,
            "gate_status": selected.get("gate_status"),
            "deliverable": bool(
                selected.get("gate_status", {}).get("hard_gates_pass")
            ),
            "final_judgement": selected.get("judgement"),
            "variants": selected.get("variants") or [],
        }
        metadata["failure_diagnosis"] = classify_failure(
            selected.get("judgement"),
            gate_failures=selected.get("gate_status", {}).get(
                "hard_gate_failures", []
            ),
        )
        deliverable = bool(
            selected.get("gate_status", {}).get("hard_gates_pass")
        )
        metadata["shortlist"] = self._agent_router.candidate_shortlist(candidates, limit=2)
        metadata["face_swap"] = selected["repair"]
        if selected.get("local_edit"):
            metadata["local_edit"] = selected["local_edit"]
        for candidate in candidates:
            candidate.pop("path", None)
        for record in metadata["history"]:
            if record["iteration"] == selected["index"]:
                record["accepted"] = deliverable

        if progress_callback:
            total_attempts = candidate_count + metadata["budget"]["regenerations_used"]
            if deliverable:
                detail = (
                    f"{profile.final_detail_prefix} {selected['index']}/{total_attempts}"
                    + (f" · identity {final_score}/10" if final_score is not None else "")
                )
                phase = "accepted"
            else:
                failures = selected.get("gate_status", {}).get(
                    "hard_gate_failures", []
                )
                reason = ", ".join(failures[:2]) or "final quality gate"
                detail = f"Candidate {selected['index']}/{total_attempts} held: {reason}"
                phase = "failed_gate"
            progress_callback(
                selected["index"], total_attempts, phase, detail
            )

        return filepath, metadata

    # ── Controlled candidate pipeline (full set) ───────────────

    def execute_generate_with_quality_pipeline(
        self,
        session_id: str,
        prompt: str,
        photo_paths: list[str],
        title: str,
        template_path: str | None = None,
        progress_callback=None,
        shot_spec_metadata: dict | None = None,
        session_feedback: list[dict] | None = None,
    ) -> tuple[str, dict]:
        """Generate a small candidate set, QA-score it, repair once, then select.

        This is the product pipeline:
          1. Generate N constrained candidates from the same template.
          2. Ask a structured judge for identity/style/artifact scores.
          3. Select the best candidate instead of revising in free text.
          4. Apply deterministic face-swap identity repair to the selected image.

        ``session_feedback`` is the session's prior user feedback (event-tagged
        records) used to condition the policy engine.
        """
        profile = _PipelineProfile(
            prompt_version=PIPELINE_PROMPT_VERSION,
            candidate_count=PIPELINE_CANDIDATE_COUNT,
            max_regenerations=MAX_PIPELINE_REGENERATIONS,
            max_local_edits=MAX_PIPELINE_LOCAL_EDITS,
            max_identity_repairs=MAX_PIPELINE_IDENTITY_REPAIRS,
            max_total_api_cost=MAX_PIPELINE_TOTAL_API_COST,
            force_closeup=False,
            candidate_id_prefix="cand_",
            create_inv_id_prefix="create_",
            regenerate_inv_id_prefix="regenerate_",
            local_edit_inv_id_prefix="local_edit_",
            identity_repair_inv_id_prefix="identity_repair_",
            candidate_title_suffix="_cand",
            regenerate_title_suffix="_regen",
            drop_on_max_regenerations=True,
            mark_local_edit_skip_reason=True,
            budget_exhausted_message="Initial candidate budget exhausted before generation",
            no_repair_message="QA accepted candidate; no identity repair needed",
            progress_candidate_word="candidate",
            progress_regenerate_word="from original references",
            progress_repair_repair="Applying identity repair and final checks…",
            progress_repair_local_edit="Applying local artifact edit and final checks…",
            progress_repair_accept="Candidate passed QA; skipping identity repair…",
            final_detail_prefix="Selected candidate",
        )
        return self._execute_candidate_pipeline(
            profile=profile,
            session_id=session_id,
            prompt=prompt,
            photo_paths=photo_paths,
            title=title,
            template_path=template_path,
            progress_callback=progress_callback,
            shot_spec_metadata=shot_spec_metadata,
            session_feedback=session_feedback,
        )


    @staticmethod
    def _public_repair_metadata(action: str, result: FaceSwapResult) -> dict:
        """Keep repair metadata useful without exposing local absolute paths."""
        return {
            "action": action,
            "applied": result.swapped,
            "message": result.message,
            "source_face_count": result.source_face_count,
            "target_face_count": result.target_face_count,
            "output_filename": result.output_path.name,
        }

    @staticmethod
    def _decide_candidate_action(
        judgement: dict,
        edit_count: int = 0,
        identity_repairs: int = 0,
        identity_thresholds: dict | None = None,
    ) -> dict:
        """Bounded state-machine action for one evaluated candidate.

        NOTE: This static method preserves the original deterministic AgentRouter
        behavior for backward compatibility in tests. The main runtime path uses
        PolicyEngine for adaptive decisions.
        """
        router = AgentRouter(identity_threshold_profile)
        return router.decide_candidate_action(
            judgement,
            edit_count=edit_count,
            identity_repairs=identity_repairs,
            identity_thresholds=identity_thresholds,
        )

    @staticmethod
    def _should_apply_identity_repair(
        judgement: dict,
        identity_thresholds: dict | None = None,
    ) -> bool:
        """Return True only for identity-gray-zone candidates worth repairing."""
        router = AgentRouter(identity_threshold_profile)
        return router.should_apply_identity_repair(
            judgement,
            identity_thresholds=identity_thresholds,
        )

    @staticmethod
    def _select_candidate(candidates: list[dict]) -> dict | None:
        router = AgentRouter(identity_threshold_profile)
        return router.select_candidate(candidates)

    @staticmethod
    def _candidate_shortlist(candidates: list[dict], limit: int = 2) -> list[dict]:
        """Public candidate-funnel summary: top retained candidates, no paths."""
        router = AgentRouter(identity_threshold_profile)
        return router.candidate_shortlist(candidates, limit=limit)

    # ── Iterative resemblance agent ───────────────────────────

    def execute_generate_with_resemblance_loop(
        self,
        session_id: str,
        prompt: str,
        photo_paths: list[str],
        title: str,
        template_path: str | None = None,
        progress_callback=None,
    ) -> tuple[str, dict]:
        """Generate with iterative resemblance checking.

        After initial generation, asks Gemini to rate how well the output
        resembles the user's reference photo. If score < threshold, uses
        Gemini's own feedback to revise. Repeats up to MAX_RESEMBLANCE_ITERATIONS.

        Args:
            session_id: Session identifier
            prompt: Style prompt from prompts.json
            photo_paths: List of user's selfie photo paths (all uploaded photos)
            title: Filename prefix
            template_path: Style template image path
            progress_callback: callable(iteration, max_iterations, phase, detail)

        Returns:
            (filepath, metadata) where metadata contains:
                iterations: number of judge iterations run
                final_score: last resemblance score
                history: list of per-iteration records
        """
        # Step 1: Initial generation
        self._ensure_session(session_id, photo_paths[0] if photo_paths else "")
        self._turn_counts[session_id] = 1

        # Cap character references to what Nano Banana 2 handles well.
        ref_photos = photo_paths[:MAX_CHARACTER_REFERENCES]

        if progress_callback:
            progress_callback(0, MAX_RESEMBLANCE_ITERATIONS, "generating", "Initial generation…")

        wrapped_prompt = build_editing_prompt(prompt, num_user_photos=len(ref_photos))
        filepath = self._gateway.create_from_references(
            prompt=wrapped_prompt,
            reference_paths=ref_photos,
            template_path=template_path,
            title=f"{title}_initial",
            editing_mode=True,
        )

        metadata = {
            "iterations": 0,
            "final_score": None,
            "history": [],
        }

        # Step 2: Resemblance loop
        for i in range(1, MAX_RESEMBLANCE_ITERATIONS + 1):
            if progress_callback:
                progress_callback(
                    i, MAX_RESEMBLANCE_ITERATIONS, "judging",
                    f"Resemblance scoring — attempt {i}…",
                )

            # Ask Gemini to judge resemblance. The judge gets the SAME time budget
            # as image generation (settings.gemini_wait_timeout = 180s), NOT the
            # old 60s hardcode. That hardcode cut the verdict off early under
            # Gemini slowness / rate-limiting → TimeoutException → the loop
            # silently accepted an UN-JUDGED image (score=None), defeating the
            # entire resemblance differentiator.
            #
            # One bounded retry on FAST failures only (transient driver/DOM
            # glitch): if the call dies within the first 20% of the budget we try
            # once more. A SLOW timeout has already burned the full budget, so we
            # do NOT retry it — retrying would double one judge turn to
            # 2×timeout (up to ~6 min across 3 turns). Fall through to
            # accept-current instead.
            judge_timeout = settings.gemini_wait_timeout
            response_text: str | None = None
            judge_error: Exception | None = None
            for attempt in (1, 2):
                t0 = time.time()
                try:
                    response_text = self._gateway.judge(
                        current_image_path=filepath,
                        reference_paths=ref_photos,
                        judge_prompt=RESEMBLANCE_JUDGE_PROMPT,
                        timeout=judge_timeout,
                    )
                    judge_error = None
                    break
                except Exception as e:
                    elapsed = time.time() - t0
                    judge_error = e
                    fast = elapsed < judge_timeout * 0.2
                    print(f"  ⚠  Judge failed (iter {i}, attempt {attempt}/2, "
                          f"{elapsed:.1f}s in): {e}")
                    if attempt == 1 and fast:
                        time.sleep(2)  # brief backoff before the single retry
                        continue
                    break  # slow timeout, or second attempt failed

            if response_text is None:
                # All attempts failed (or a slow timeout) — accept the current
                # image rather than crash the job. This is the graceful fallback;
                # the resemblance pass is skipped, not the whole generation.
                metadata["iterations"] = i
                metadata["final_score"] = None
                metadata["history"].append({
                    "iteration": i,
                    "score": None,
                    "feedback": None,
                    "error": str(judge_error) if judge_error
                    else "judge produced no response",
                    "accepted": True,
                })
                break

            print(f"  🔍 Judge response (iter {i}): {response_text[:200]}...")

            # Parse the response
            score, feedback = self._parse_judge_response(response_text)

            iteration_record = {
                "iteration": i,
                "score": score,
                "feedback": feedback,
                "raw_response": response_text[:500],
                "accepted": False,
            }
            metadata["history"].append(iteration_record)
            metadata["iterations"] = i
            metadata["final_score"] = score

            print(f"  📊 Iteration {i}: score={score}, feedback={'yes' if feedback else 'no'}")

            if score is not None and score >= RESEMBLANCE_THRESHOLD:
                # Accept! The image is good enough.
                iteration_record["accepted"] = True
                if progress_callback:
                    progress_callback(
                        i, MAX_RESEMBLANCE_ITERATIONS, "accepted",
                        f"Resemblance {score}/10 — accepted!",
                    )
                break

            if i >= MAX_RESEMBLANCE_ITERATIONS:
                # Max iterations reached, accept whatever we have
                iteration_record["accepted"] = True
                if progress_callback:
                    progress_callback(
                        i, MAX_RESEMBLANCE_ITERATIONS, "max_reached",
                        f"Max iterations reached — best score {score or '?'}/10",
                    )
                break

            # Score < threshold: revise using feedback
            if feedback:
                revision_instruction = RESEMBLANCE_REVISION_PROMPT_TEMPLATE.format(
                    feedback=feedback
                )
            else:
                revision_instruction = GENERIC_REVISION_PROMPT

            if progress_callback:
                progress_callback(
                    i, MAX_RESEMBLANCE_ITERATIONS, "revising",
                    f"Revising — attempt {i} (score {score or '?'}/10)…",
                )

            turn = self._turn_counts.get(session_id, 1) + 1
            self._turn_counts[session_id] = turn

            filepath = self._gateway.local_edit(
                current_image_path=filepath,
                reference_paths=ref_photos,
                edit_prompt=revision_instruction,
                title=f"{title}_iter{i}",
            )

        # Post-process with dedicated face-swap model to harden identity
        # preservation. This runs after the resemblance loop so the loop still
        # optimises style/composition; the swap fixes the final face.
        if progress_callback:
            progress_callback(
                metadata["iterations"],
                MAX_RESEMBLANCE_ITERATIONS,
                "face_swapping",
                "Applying face-swap post-processing…",
            )
        swap_result = self._apply_face_swap(filepath, photo_paths, title)
        if swap_result.swapped:
            filepath = str(swap_result.output_path)

        metadata["face_swap"] = {
            "applied": swap_result.swapped,
            "message": swap_result.message,
            "source_face_count": swap_result.source_face_count,
            "target_face_count": swap_result.target_face_count,
            "output_path": str(swap_result.output_path),
        }

        return filepath, metadata

    @staticmethod
    def _parse_judge_response(text: str) -> tuple[int | None, str | None]:
        """Parse Gemini's judge response to extract score and feedback.

        Expected format includes "评分：X/10" somewhere in the text.

        Returns:
            (score, feedback) where score is None if parsing fails.
        """
        if not text:
            return None, None

        # Extract score: try patterns in priority order.
        #
        # DECIMAL SCORES (root-cause fix, verified against the live API):
        # gemini-3.1-flash-image frequently returns DECIMAL scores
        # ("评分：9.5/10", "评分：9.2/10") despite the prompt asking for an
        # integer. The old integer-only `(\d+)` groups broke on these:
        #   - pattern `评分[：:]\s*(\d+)\s*/\s*10` failed on "9.5/10" (the ".5"
        #     sits between "9" and "/10", so the \d+ can't reach "/10"), then
        #   - the fallback `(\d+)\s*/\s*10` matched the "5" in "9.5" → score=5.
        # A genuine 9.5/10 image was therefore mis-read as 5/10 → it fell below
        # the accept threshold → a pointless revision fired → the phantom "score
        # < 8" branch also turned the bare score string into fake "feedback".
        # Net effect: two needless ~$0.07 regeneration turns and the final image
        # was the 3rd revision, not the excellent 1st. Capture the decimal and
        # round it instead.
        score = None
        score_patterns = [
            r"评分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            # Colon OPTIONAL here: Gemini is instructed to reply "评分：X/10" but
            # often emits the equally-natural "评分9分" (no colon, no slash).
            # Requiring the colon made that return None → the loop mis-took it as
            # a failed judge and accepted a sub-threshold image.
            r"评分[:：]?\s*(\d+(?:\.\d+)?)\s*[分点]",
            r"相似度[：:]\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*分\s*[,，]\s*满分\s*10",  # "8分，满分10"
        ]
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                # round-half-up via +0.5 then truncate: 9.5→10, 9.2→9, 8.5→9.
                # Plain round() is banker's rounding (9.5→10 but 8.5→8), which
                # would penalise a borderline 8.5; +0.5 is the intuitive "round
                # to nearest, ties up" for a resemblance accept/revise gate.
                score = int(float(match.group(1)) + 0.5)
                score = max(1, min(10, score))  # clamp to 1-10
                break

        # Extract feedback: find text describing specific adjustments
        feedback = None
        feedback_sections = []
        lines = text.split("\n")
        capturing = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Start capturing after trigger keywords
            if any(kw in line for kw in [
                "需要调整", "差异", "不像", "不一致", "不同",
                "调整以下", "以下面部特征", "具体来说",
            ]):
                capturing = True
            if capturing and line:
                feedback_sections.append(line)

        if feedback_sections:
            feedback = "\n".join(feedback_sections)
        elif score is not None and score < 8:
            # No structured feedback found — use full text as fallback
            feedback = text

        return score, feedback

    # ── Revision (manual, user-initiated) ─────────────────────

    def execute_revise(
        self,
        session_id: str,
        instruction: str,
        title: str,
        source_image_path: str,
    ) -> str:
        """Run a bounded local-edit revision on a delivered parent image.

        ``source_image_path`` is the on-disk path of the parent image being
        retouched (resolved and path-traversal-checked by the caller). The edit
        is a LOCAL_EDIT — it retouches the existing image rather than
        regenerating from scratch, so identity is preserved.
        """
        self._ensure_session(session_id)

        turn = self._turn_counts.get(session_id, 1) + 1
        self._turn_counts[session_id] = turn

        filepath = self._gateway.local_edit(
            current_image_path=source_image_path,
            reference_paths=[],
            edit_prompt=instruction,
            title=title,
        )
        return filepath

    def _judge_current_candidate(
        self,
        image_path: str,
        reference_photo_paths: list[str] | None = None,
    ) -> dict:
        """Judge a revised image through the worker's evaluator + gateway.

        Thin delegate so callers (job_queue revise path) can ask the worker to
        score an image without knowing about EvaluationService/ImageGateway
        wiring. Mirrors the judge step used inside the candidate pipelines.
        """
        return self._eval_service.judge_current_candidate(
            self._gateway,
            image_path,
            reference_photo_paths or [],
        )

    def identity_thresholds_for_shot(self, shot_spec: dict | None = None) -> dict:
        """Return the current geometry-aware, feedback-calibrated thresholds."""
        return self._eval_service._get_identity_thresholds(shot_spec)

    def evaluate_portrait_set_visual(
        self,
        images: list[dict],
        contact_sheet_path: str,
    ) -> dict:
        """Run the non-blocking six-frame visual review after the final shot."""
        try:
            return judge_visual_portrait_set(
                self._gateway,
                images,
                contact_sheet_path=contact_sheet_path,
            )
        finally:
            Path(contact_sheet_path).unlink(missing_ok=True)

    def end_session(self, session_id: str):
        """End the conversation for a session."""
        if self.active_session_id == session_id:
            try:
                self._gateway.end_session()
            except Exception:
                pass
            self.active_session_id = None

    def release_job_resources(self, session_id: str) -> None:
        """Return native image-model memory after each serialized shot."""
        self.end_session(session_id)
        self._eval_service.release_identity_app()

        import gc
        import sys

        gc.collect()
        if sys.platform.startswith("linux"):
            try:
                import ctypes

                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

    def _ensure_session(self, session_id: str, photo_path: str = ""):
        """Make sure the generation pipeline is for the given session."""
        if self.active_session_id == session_id:
            return  # already in the right session

        # End old session if any
        if self.active_session_id is not None:
            try:
                self._gateway.end_session()
            except Exception:
                pass

        self.active_session_id = session_id
