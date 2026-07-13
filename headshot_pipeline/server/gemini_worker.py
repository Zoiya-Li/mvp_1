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
from .evaluation import EvaluationService, AgentRouter, PolicyEngine
from .generation import ImageGateway
from .image_gateway import build_provider_invocation_metadata, estimate_cost
from .learning import LearningLayer
from .models import IdentityPack, ShotSpec
from .repair import FaceSwapRepair, public_repair_metadata

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
MAX_PIPELINE_REGENERATIONS = 2
MAX_PIPELINE_LOCAL_EDITS = 2
MAX_PIPELINE_IDENTITY_REPAIRS = 1
MAX_PIPELINE_TOTAL_API_COST = 1.0
PIPELINE_PROMPT_VERSION = "controlled_candidate_v2"
IDENTITY_TEMPLATE_VERSION = "identity_pack_v2"
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
    user_indices = ", ".join(str(i) for i in range(2, 2 + num_user_photos))
    return f"""\
Image 1 is the style reference ONLY — use it for composition, lighting, background, and clothing style.
Images {user_indices} are all of the SAME PERSON (the user). This person is the ONLY subject whose face you must preserve.

Instruction:
{base_prompt}

Critical requirements:
- Generate a portrait of the person in images {user_indices}, NOT the person in image 1.
- Preserve the user's exact facial features, face shape, eyes, nose, mouth, eyebrows, skin tone, and overall identity.
- Apply only the style/composition/background/clothing from image 1.
- Natural skin texture, no beauty filter, no plastic skin.
- Photorealistic, professional headshot quality.
- If the user is wearing different clothing in the reference photos, still match the clothing style from image 1."""


def build_editing_prompt(base_prompt: str, num_user_photos: int) -> str:
    """Edit-framing wrapper: user selfie first, style template last.

    Experiments (experiments/compare_identity_preservation.py) showed that
    telling the model to "edit the user photo while preserving identity"
    produces significantly better resemblance scores than template-first
    generation on Google Gemini image models.
    """
    user_indices = ", ".join(str(i) for i in range(1, 1 + num_user_photos))
    style_index = 1 + num_user_photos
    return f"""\
Images {user_indices} are the user's photos. This is the person whose face, expression, and identity you must preserve EXACTLY.
Image {style_index} is the style reference — use it for composition, lighting, background, and clothing style.

Instruction:
Apply the style from image {style_index} to the person in images {user_indices}. Change clothing and background to match image {style_index}, but keep the face and identity identical.

Style details:
{base_prompt}

Critical requirements:
- Edit the person in images {user_indices} to match the style/composition/background/clothing of image {style_index}.
- Keep the person's facial features, face shape, eyes, nose, mouth, eyebrows, skin tone, expression, and overall identity exactly the same as in images {user_indices}.
- Change only clothing, background, lighting, and overall aesthetic to match image {style_index}.
- Do NOT generate a different person.
- Natural skin texture, no beauty filter, no plastic skin.
- Photorealistic, professional portrait quality."""


def build_candidate_prompt(base_prompt: str, num_user_photos: int,
                           candidate_index: int, total_candidates: int) -> str:
    """Variant of the edit prompt for candidate-set generation.

    We keep the template constraints fixed and only vary the production intent a
    little, so diversity comes from sampling without drifting into a new style.
    """
    wrapped = build_editing_prompt(base_prompt, num_user_photos)
    variant_notes = [
        "Candidate strategy: conservative identity-first result. Keep the user's face as close as possible to the reference photos.",
        "Candidate strategy: polished studio result. Keep identity exact while improving lighting, clothing, and background realism.",
        "Candidate strategy: natural commercial result. Prioritize believable expression and skin texture while preserving identity.",
    ]
    note = variant_notes[(candidate_index - 1) % len(variant_notes)]
    return f"""{wrapped}

Production candidate {candidate_index}/{total_candidates}.
{note}

Do not over-beautify. Do not change age, face shape, hairstyle, glasses, facial hair, moles, or other identity markers."""


def build_local_edit_prompt(judgement: dict) -> str:
    """Prompt for fixing only local artifacts on an otherwise viable candidate."""
    failures = ", ".join(judgement.get("hard_failures") or [])
    notes = str(judgement.get("notes") or "")
    return f"""\
Perform a LOCAL_EDIT only. Fix the visible local artifact(s) in the current generated portrait without changing the person's identity, face shape, age, expression, pose, camera angle, clothing style, background, lighting, framing, or overall composition.

Detected local issues: {failures or "minor local artifact"}
QA notes: {notes[:300]}

Allowed fixes:
- hand or finger artifact cleanup
- collar / hair / accessory glitch cleanup
- small background stray-object cleanup
- minor expression or gaze correction
- subtle local color cleanup

Do not regenerate a new person. Do not change the scene. Do not make a full-body/pose/background/style change. Preserve realistic skin texture."""


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
        self._learning_layer = learning_layer or LearningLayer()
        self._eval_service = EvaluationService(learning_layer=self._learning_layer)
        self._agent_router = PolicyEngine(
            agent_router=AgentRouter(identity_threshold_profile),
            learning_layer=self._learning_layer,
        )
        self._face_swap_repair = FaceSwapRepair()
        self._gateway = ImageGateway()
        self.provider_readiness: dict | None = None

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

    HERO_PREVIEW_CANDIDATE_COUNT = 4
    HERO_PREVIEW_MAX_REGENERATIONS = 1
    HERO_PREVIEW_MAX_LOCAL_EDITS = 1
    HERO_PREVIEW_MAX_IDENTITY_REPAIRS = 1
    HERO_PREVIEW_MAX_TOTAL_API_COST = 0.6
    HERO_PREVIEW_PROMPT_VERSION = "hero_preview_v1"

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

        ref_photos = photo_paths[:MAX_CHARACTER_REFERENCES]
        eval_ref_photos = photo_paths[:MAX_IDENTITY_PACK_REFERENCES]
        candidate_count = profile.candidate_count
        candidates: list[dict] = []
        provider_invocations: list[dict] = []
        agent_actions: list[dict] = []
        identity_pack = build_identity_pack_metadata(eval_ref_photos)
        shot_spec = shot_spec_metadata or build_shot_spec_metadata(
            prompt, title, template_path
        )
        if profile.force_closeup:
            shot_spec["shot_id"] = "closeup"
        identity_thresholds = self._eval_service._get_identity_thresholds(shot_spec)
        generation_task_type = model_task_type_for_shot(
            shot_spec,
            force_closeup=profile.force_closeup,
        )
        generation_routing = self._gateway.route_by_task(
            generation_task_type,
            shot_spec=shot_spec,
            budget_remaining=profile.max_total_api_cost,
        )
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
                "max_regenerations": profile.max_regenerations,
                "max_local_edits": profile.max_local_edits,
                "max_identity_repairs": profile.max_identity_repairs,
                "max_total_api_cost": profile.max_total_api_cost,
                "initial_candidates_generated": 0,
                "regenerations_used": 0,
                "local_edits_used": 0,
                "identity_repairs_used": 0,
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
                "generation_task_type": generation_task_type,
                "generation_routing": generation_routing,
            },
            "candidates": candidates,
            "agent_actions": agent_actions,
            "provider_invocations": provider_invocations,
            "selected_candidate": None,
            "shortlist": [],
        }
        reference_ids = [
            f"ref_{idx + 1}"
            for idx in range(len(ref_photos))
        ]
        reference_manifest = identity_pack_reference_manifest(
            identity_pack,
            set(reference_ids),
        )

        # ── 1. Initial candidate sampling ───────────────────────────
        for idx in range(1, candidate_count + 1):
            invocation_cost = estimate_cost(
                "CREATE_FROM_REFERENCES", len(ref_photos)
            )
            if (
                metadata["budget"]["estimated_cost_used"] + invocation_cost
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

            candidate_prompt = build_candidate_prompt(
                prompt, len(ref_photos), idx, candidate_count
            )
            started_at = time.time()
            filepath = self._gateway.create_from_references(
                prompt=candidate_prompt,
                reference_paths=ref_photos,
                template_path=template_path,
                title=f"{title}{profile.candidate_title_suffix}{idx}",
                editing_mode=True,
            )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.create_inv_id_prefix}{idx}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=profile.prompt_version,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=idx,
                parent_candidate_id=None,
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=invocation_cost,
                result_status="success",
            )
            invocation["routing_decision"] = generation_routing
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + invocation_cost, 4
            )

            if progress_callback:
                progress_callback(
                    idx, candidate_count, "candidate_judging",
                    f"Quality scoring {cand_word} {idx}/{candidate_count}…",
                )

            judgement = self._eval_service.judge_current_candidate(self._gateway, filepath, eval_ref_photos)
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
                "repair": None,
            }
            action = self._agent_router.decide(
                judgement,
                budget=metadata["budget"],
                shot_spec=shot_spec,
                session_feedback=session_feedback,
                edit_count=0,
                identity_repairs=0,
                identity_thresholds=identity_thresholds,
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
            selected.get("agent_action", {}).get("action")
            == "REGENERATE_FROM_ORIGINAL"
            and metadata["budget"]["regenerations_used"]
            < metadata["budget"]["max_regenerations"]
        ):
            regen_cost = estimate_cost("CREATE_FROM_REFERENCES", len(ref_photos))
            if (
                metadata["budget"]["estimated_cost_used"] + regen_cost
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

            candidate_prompt = build_candidate_prompt(
                prompt, len(ref_photos), idx, max_steps
            )
            started_at = time.time()
            filepath = self._gateway.create_from_references(
                prompt=candidate_prompt,
                reference_paths=ref_photos,
                template_path=template_path,
                title=f"{title}{profile.regenerate_title_suffix}{regen_no}",
                editing_mode=True,
            )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.regenerate_inv_id_prefix}{regen_no}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=profile.prompt_version,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=idx,
                parent_candidate_id=selected.get("candidate_id"),
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=regen_cost,
                result_status="success",
            )
            invocation["routing_decision"] = generation_routing
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + regen_cost, 4
            )

            judgement = self._eval_service.judge_current_candidate(self._gateway, filepath, eval_ref_photos)
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
                "repair": None,
                "regenerated_from_candidate_id": selected.get("candidate_id"),
            }
            action = self._agent_router.decide(
                judgement,
                budget=metadata["budget"],
                shot_spec=shot_spec,
                session_feedback=session_feedback,
                edit_count=0,
                identity_repairs=0,
                identity_thresholds=identity_thresholds,
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
            and selected.get("agent_action", {}).get("action")
            == "REGENERATE_FROM_ORIGINAL"
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
        repair_needed = (
            selected_action.get("action") == "IDENTITY_REPAIR"
            or self._agent_router.should_apply_identity_repair(
                selected.get("judgement", {}),
                identity_thresholds,
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
            local_edit_cost = estimate_cost("LOCAL_EDIT", len(ref_photos))
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
            local_edit_routing = self._gateway.route_by_task(
                "local_edit",
                shot_spec=shot_spec,
                budget_remaining=(
                    metadata["budget"]["max_total_api_cost"]
                    - metadata["budget"]["estimated_cost_used"]
                ),
            )
            started_at = time.time()
            edit_prompt = build_local_edit_prompt(selected.get("judgement", {}))
            edited_path = self._gateway.local_edit(
                current_image_path=filepath,
                reference_paths=ref_photos,
                edit_prompt=edit_prompt,
                title=f"{title}{profile.candidate_title_suffix}{selected['index']}_local_edit",
            )
            filepath = edited_path
            metadata["budget"]["local_edits_used"] += 1
            local_invocation = build_provider_invocation_metadata(
                invocation_id=f"{profile.local_edit_inv_id_prefix}{selected['index']}",
                operation="LOCAL_EDIT",
                prompt_version=profile.prompt_version,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=selected["index"],
                parent_candidate_id=selected["candidate_id"],
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=local_edit_cost,
                result_status="success",
            )
            local_invocation["routing_decision"] = local_edit_routing
            provider_invocations.append(local_invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + local_edit_cost, 4
            )
            edited_judgement = self._eval_service.judge_current_candidate(self._gateway,
                filepath,
                eval_ref_photos,
            )
            selected["local_edit"] = {
                "action": "local_edit",
                "applied": True,
                "output_filename": Path(filepath).name,
                "post_edit_judgement": edited_judgement,
            }
            selected["aggregate_score"] = self._eval_service._aggregate_quality_score(
                edited_judgement
            )
            selected["gate_status"] = self._eval_service._candidate_gate_status(
                edited_judgement,
                identity_thresholds,
            )
            selected_scores = edited_judgement.get("scores", {})
            selected_identity = selected_scores.get("identity")
            repair_needed = self._should_apply_identity_repair(
                edited_judgement,
                identity_thresholds,
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
                )
                selected["repair"]["post_repair_judgement"] = repaired_judgement
                selected["aggregate_score"] = self._eval_service._aggregate_quality_score(
                    repaired_judgement
                )
                selected["gate_status"] = self._eval_service._candidate_gate_status(
                    repaired_judgement,
                    identity_thresholds,
                )
                selected_scores = repaired_judgement.get("scores", {})
                selected_identity = selected_scores.get("identity")
            else:
                selected["repair"] = self._public_repair_metadata(
                    "face_swap", swap_result
                )
        else:
            selected["repair"] = {
                "action": "none",
                "applied": False,
                "message": profile.no_repair_message,
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
        }
        metadata["shortlist"] = self._agent_router.candidate_shortlist(candidates, limit=2)
        metadata["face_swap"] = selected["repair"]
        if selected.get("local_edit"):
            metadata["local_edit"] = selected["local_edit"]
        for candidate in candidates:
            candidate.pop("path", None)
        for record in metadata["history"]:
            if record["iteration"] == selected["index"]:
                record["accepted"] = True

        if progress_callback:
            detail = (
                f"{profile.final_detail_prefix} {selected['index']}/{candidate_count}"
                + (f" · identity {final_score}/10" if final_score is not None else "")
            )
            progress_callback(
                selected["index"], candidate_count, "accepted", detail
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

    def end_session(self, session_id: str):
        """End the conversation for a session."""
        if self.active_session_id == session_id:
            try:
                self._gateway.end_session()
            except Exception:
                pass
            self.active_session_id = None

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
