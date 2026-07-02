"""Gemini worker — adapts the Gemini client for the job queue.

Tracks which session currently "owns" the Gemini conversation and handles
switching between sessions (end old → start new).

Generation backend: the OpenRouter REST API (server/openrouter_client.py),
driving ``gemini-3.1-flash-image-preview`` ("Nano Banana 2") directly. This
replaced the headless-Chrome driver (persistent_client.py) — the Chrome path
was the single biggest source of "needs constant debugging" (login expiry,
DOM-selector drift, VNC re-login, profile locks). The resemblance loop below is
unchanged: it calls start_conversation / converse_text / converse /
end_conversation, all of which the API client provides as a drop-in.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from .config import settings
from .face_swap import FaceSwapper, FaceSwapResult
from .image_gateway import build_provider_invocation_metadata, estimate_cost
from .models import IdentityPack, ShotSpec

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

# Nano Banana 2 (google/gemini-3.1-flash-image-preview) supports up to 4
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

QUALITY_JUDGE_PROMPT = """\
You are a strict production QA system for AI professional headshots.

Compare image 1 (generated candidate) against image 2 (the user's real
reference photo). Return ONLY valid JSON with this schema:

{
  "scores": {
    "identity": 0-10,
    "face_quality": 0-10,
    "style_match": 0-10,
    "artifact": 0-10,
    "commercial_readiness": 0-10
  },
  "hard_failures": ["unsafe_content" | "identity_too_low" | "face_distorted" | "no_face" | "wrong_style" | "bad_artifacts"],
  "recommended_action": "accept" | "face_swap" | "retry" | "discard",
  "notes": "one short sentence explaining the main issue"
}

Scoring rules:
- identity: strict same-person judgement. A familiar person should recognize
  them immediately for 8+.
- face_quality: face clarity, natural skin, believable eyes/mouth/nose.
- style_match: whether it matches the requested/template headshot style.
- artifact: 10 means no visible AI artifacts; lower means distortions.
- commercial_readiness: whether this is good enough to show to a paying user.
- Add unsafe_content to hard_failures for nudity, sexual content, minors, hate,
  violence, self-harm, or other content that should not be delivered.

Be strict. Do not wrap the JSON in markdown."""


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


class GeminiWorker:
    """Wraps the OpenRouter Gemini client with session-aware conversation management."""

    def __init__(self):
        self.active_session_id: str | None = None
        self._turn_counts: dict[str, int] = {}
        # Lazy-loaded in the worker thread to avoid blocking the event loop.
        self._face_swapper: FaceSwapper | None = None
        self._face_swap_load_failed: bool = False
        self._identity_app = None
        self._identity_app_load_failed: bool = False

        if settings.gemini_backend == "chrome":
            # Chrome backend: connect to an already-running Chrome via CDP.
            # The Chrome instance must have been started with
            #   python persistent_client.py launch
            # and the user must be logged in to gemini.google.com.
            from persistent_client import PersistentGeminiClient

            self.client = PersistentGeminiClient(
                port=settings.chrome_cdp_port,
                output_dir=str(settings.output_dir),
                wait_timeout=settings.chrome_wait_timeout,
            )
            return

        # OpenRouter backend (default production path).
        from .openrouter_client import OpenRouterGeminiClient, OpenRouterError

        if not settings.openrouter_api_key:
            # Hard-fail construction with a clear, actionable message. The API
            # path has NO logged-in browser session to fall back on, unlike the
            # old Chrome path — so an empty key must not silently produce a
            # worker that queues-then-fails every job. The job_queue start()
            # catches this and logs a single warning.
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set and gemini_backend is openrouter. "
                "Add OPENROUTER_API_KEY=<your key> to .env, or set "
                "GEMINI_BACKEND=chrome to use a logged-in Chrome instead."
            )

        self.client = OpenRouterGeminiClient(
            api_key=settings.openrouter_api_key,
            output_dir=str(settings.output_dir),
            model=settings.gemini_model,
            base_url=settings.openrouter_base_url,
            timeout=settings.gemini_wait_timeout,
        )

    def connect(self):
        """Validate the API key is set. Called during startup (no socket to open)."""
        self.client.connect()

    def disconnect(self):
        """No-op. The API has no persistent connection to close."""
        self.client.disconnect()

    # ── Face-swap post-processing ─────────────────────────────

    def _get_face_swapper(self) -> FaceSwapper | None:
        """Lazy-load the InsightFace face-swap model in the worker thread."""
        if not settings.face_swap_enabled:
            return None
        if self._face_swapper is not None:
            return self._face_swapper
        if self._face_swap_load_failed:
            return None

        model_path = settings.face_swap_model_path
        if not model_path.exists():
            print(f"⚠ Face-swap model not found at {model_path}; skipping.")
            self._face_swap_load_failed = True
            return None

        try:
            print(f"Loading face-swap model from {model_path}...")
            self._face_swapper = FaceSwapper(model_path)
            print("✓ Face-swap model loaded")
            return self._face_swapper
        except Exception as exc:
            print(f"⚠ Failed to load face-swap model: {exc}")
            self._face_swap_load_failed = True
            return None

    def _get_identity_app(self):
        """Lazy-load InsightFace recognition for local identity scoring."""
        if self._identity_app is not None:
            return self._identity_app
        if self._identity_app_load_failed:
            return None
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._identity_app = app
            return self._identity_app
        except Exception as exc:
            print(f"⚠ Failed to load identity scorer: {exc}")
            self._identity_app_load_failed = True
            return None

    def _apply_face_swap(
        self,
        generated_path: str,
        photo_paths: list[str],
        title: str,
    ) -> FaceSwapResult:
        """Swap the user's face onto the generated portrait.

        Falls back to the original image if the model is unavailable or no faces
        are detected. The swapped image is written next to the generated image.
        """
        swapper = self._get_face_swapper()
        if swapper is None:
            return FaceSwapResult(
                output_path=Path(generated_path),
                swapped=False,
                message="Face swap disabled or model unavailable",
            )

        generated = Path(generated_path)
        output_path = generated.with_name(f"{generated.stem}_swapped{generated.suffix}")
        try:
            result = swapper.swap(
                user_photos=photo_paths,
                style_image=generated_path,
                output_path=output_path,
            )
            if result.swapped:
                print(f"✓ Face-swapped result saved to {result.output_path}")
            else:
                print(f"⚠ Face swap skipped: {result.message}")
            return result
        except Exception as exc:
            print(f"⚠ Face swap failed: {exc}")
            return FaceSwapResult(
                output_path=generated,
                swapped=False,
                message=f"Face swap failed: {exc}",
            )

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
        # Switch conversation if needed
        self._ensure_session(session_id, photo_path)

        self._turn_counts[session_id] = 1
        wrapped_prompt = build_editing_prompt(prompt, num_user_photos=1)
        filepath = self.client.start_conversation(
            prompt=wrapped_prompt,
            photo_paths=[photo_path],
            title=title,
            template_path=template_path,
            editing_mode=True,
        )
        return filepath

    # ── Hero Preview pipeline (simplified, fast, identity-first) ──

    HERO_PREVIEW_CANDIDATE_COUNT = 1
    HERO_PREVIEW_MAX_REGENERATIONS = 1
    HERO_PREVIEW_MAX_LOCAL_EDITS = 1
    HERO_PREVIEW_MAX_IDENTITY_REPAIRS = 1
    HERO_PREVIEW_MAX_TOTAL_API_COST = 0.35
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
    ) -> tuple[str, dict]:
        """Generate a single high-quality hero preview with simplified pipeline.

        Hero preview is the Aha Moment: one close-up portrait optimized for
        identity preservation and immediate user satisfaction. It uses a
        stripped-down pipeline (1 candidate, 1 regen max, 1 repair max) to
        minimize cost while maximizing first-impression quality.
        """
        self._ensure_session(session_id, photo_paths[0] if photo_paths else "")
        self._turn_counts[session_id] = 1

        ref_photos = photo_paths[:MAX_CHARACTER_REFERENCES]
        eval_ref_photos = photo_paths[:MAX_IDENTITY_PACK_REFERENCES]
        candidate_count = self.HERO_PREVIEW_CANDIDATE_COUNT
        candidates: list[dict] = []
        provider_invocations: list[dict] = []
        agent_actions: list[dict] = []
        identity_pack = build_identity_pack_metadata(eval_ref_photos)
        shot_spec = shot_spec_metadata or build_shot_spec_metadata(
            prompt, title, template_path
        )
        # Force closeup profile for hero preview
        shot_spec["shot_id"] = "closeup"
        identity_thresholds = identity_threshold_profile(shot_spec)

        metadata = {
            "pipeline": "hero_preview_v1",
            "iterations": 0,
            "final_score": None,
            "history": [],
            "identity_pack": identity_pack,
            "shot_spec": shot_spec,
            "allowed_actions": PIPELINE_ALLOWED_ACTIONS,
            "budget": {
                "initial_candidates": candidate_count,
                "max_regenerations": self.HERO_PREVIEW_MAX_REGENERATIONS,
                "max_local_edits": self.HERO_PREVIEW_MAX_LOCAL_EDITS,
                "max_identity_repairs": self.HERO_PREVIEW_MAX_IDENTITY_REPAIRS,
                "max_total_api_cost": self.HERO_PREVIEW_MAX_TOTAL_API_COST,
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
                "prompt_version": self.HERO_PREVIEW_PROMPT_VERSION,
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
                    f"Generating hero preview {idx}/{candidate_count}…",
                )

            candidate_prompt = build_candidate_prompt(
                prompt, len(ref_photos), idx, candidate_count
            )
            started_at = time.time()
            filepath = self.client.start_conversation(
                prompt=candidate_prompt,
                photo_paths=ref_photos,
                title=f"{title}_hero_cand{idx}",
                template_path=template_path,
                editing_mode=True,
            )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"hero_create_{idx}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=self.HERO_PREVIEW_PROMPT_VERSION,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=idx,
                parent_candidate_id=None,
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=invocation_cost,
                result_status="success",
            )
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + invocation_cost, 4
            )

            if progress_callback:
                progress_callback(
                    idx, candidate_count, "candidate_judging",
                    f"Quality scoring hero preview {idx}/{candidate_count}…",
                )

            judgement = self._judge_current_candidate(filepath, eval_ref_photos)
            candidate = {
                "index": idx,
                "candidate_id": f"hero_cand_{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._aggregate_quality_score(judgement),
                "gate_status": self._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "repair": None,
            }
            action = self._decide_candidate_action(
                judgement,
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
            raise RuntimeError("Hero preview candidate budget exhausted before generation")

        selected = self._select_candidate(candidates)
        if selected is None:
            selected = candidates[-1]

        # One regeneration attempt if needed
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
                    f"Regenerating hero preview {regen_no}/"
                    f"{metadata['budget']['max_regenerations']}…",
                )

            candidate_prompt = build_candidate_prompt(
                prompt, len(ref_photos), idx, max_steps
            )
            started_at = time.time()
            filepath = self.client.start_conversation(
                prompt=candidate_prompt,
                photo_paths=ref_photos,
                title=f"{title}_hero_regen{regen_no}",
                template_path=template_path,
                editing_mode=True,
            )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"hero_regenerate_{regen_no}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=self.HERO_PREVIEW_PROMPT_VERSION,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=idx,
                parent_candidate_id=selected.get("candidate_id"),
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=regen_cost,
                result_status="success",
            )
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + regen_cost, 4
            )

            judgement = self._judge_current_candidate(filepath, eval_ref_photos)
            candidate = {
                "index": idx,
                "candidate_id": f"hero_cand_{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._aggregate_quality_score(judgement),
                "gate_status": self._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "repair": None,
                "regenerated_from_candidate_id": selected.get("candidate_id"),
            }
            action = self._decide_candidate_action(
                judgement,
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

            selected = self._select_candidate(candidates) or candidate

        # Apply local edit or identity repair if needed
        selected["selected"] = True
        filepath = selected["path"]
        selected_scores = selected.get("judgement", {}).get("scores", {})
        selected_identity = selected_scores.get("identity")
        selected_action = selected.get("agent_action", {})
        local_edit_needed = selected_action.get("action") == "LOCAL_EDIT"
        repair_needed = (
            selected_action.get("action") == "IDENTITY_REPAIR"
            or self._should_apply_identity_repair(
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
                    "Applying identity repair to hero preview…"
                    if repair_needed
                    else "Applying local artifact edit…"
                    if local_edit_needed
                    else "Hero preview passed QA…"
                ),
            )

        if local_edit_needed:
            local_edit_cost = estimate_cost("LOCAL_EDIT", len(ref_photos))
            if (
                metadata["budget"]["estimated_cost_used"] + local_edit_cost
                <= metadata["budget"]["max_total_api_cost"]
            ):
                started_at = time.time()
                if hasattr(self.client, "_last_image_path"):
                    self.client._last_image_path = filepath
                edit_prompt = build_local_edit_prompt(selected.get("judgement", {}))
                edited_path = self.client.converse(
                    edit_prompt,
                    title=f"{title}_hero_cand{selected['index']}_local_edit",
                    turn_number=2,
                )
                filepath = edited_path
                metadata["budget"]["local_edits_used"] += 1
                local_invocation = build_provider_invocation_metadata(
                    invocation_id=f"hero_local_edit_{selected['index']}",
                    operation="LOCAL_EDIT",
                    prompt_version=self.HERO_PREVIEW_PROMPT_VERSION,
                    reference_ids=reference_ids,
                    reference_roles=reference_manifest,
                    candidate_index=selected["index"],
                    parent_candidate_id=selected["candidate_id"],
                    shot_id=shot_spec.get("shot_id"),
                    latency_ms=int((time.time() - started_at) * 1000),
                    cost=local_edit_cost,
                    result_status="success",
                )
                provider_invocations.append(local_invocation)
                metadata["budget"]["estimated_cost_used"] = round(
                    metadata["budget"]["estimated_cost_used"] + local_edit_cost, 4
                )
                edited_judgement = self._judge_current_candidate(
                    filepath,
                    eval_ref_photos,
                )
                selected["local_edit"] = {
                    "action": "local_edit",
                    "applied": True,
                    "output_filename": Path(filepath).name,
                    "post_edit_judgement": edited_judgement,
                }
                selected["aggregate_score"] = self._aggregate_quality_score(
                    edited_judgement
                )
                selected["gate_status"] = self._candidate_gate_status(
                    edited_judgement,
                    identity_thresholds,
                )
                selected_scores = edited_judgement.get("scores", {})
                selected_identity = selected_scores.get("identity")
                repair_needed = self._should_apply_identity_repair(
                    edited_judgement,
                    identity_thresholds,
                )

        if repair_needed:
            swap_result = self._apply_face_swap(filepath, photo_paths, title)
            if swap_result.swapped:
                metadata["budget"]["identity_repairs_used"] += 1
                filepath = str(swap_result.output_path)
                selected["repair"] = self._public_repair_metadata(
                    "face_swap", swap_result
                )
                repair_invocation = build_provider_invocation_metadata(
                    invocation_id=f"hero_identity_repair_{selected['index']}",
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
                if hasattr(self.client, "_last_image_path"):
                    self.client._last_image_path = filepath
                repaired_judgement = self._judge_current_candidate(
                    filepath,
                    eval_ref_photos,
                )
                selected["repair"]["post_repair_judgement"] = repaired_judgement
                selected["aggregate_score"] = self._aggregate_quality_score(
                    repaired_judgement
                )
                selected["gate_status"] = self._candidate_gate_status(
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
                "message": "Hero preview QA accepted; no identity repair needed",
            }

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
        metadata["shortlist"] = self._candidate_shortlist(candidates, limit=2)
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
                f"Hero preview selected candidate {selected['index']}/{candidate_count}"
                + (f" · identity {final_score}/10" if final_score is not None else "")
            )
            progress_callback(
                selected["index"], candidate_count, "accepted", detail
            )

        return filepath, metadata

    # ── Controlled candidate pipeline ─────────────────────────

    def execute_generate_with_quality_pipeline(
        self,
        session_id: str,
        prompt: str,
        photo_paths: list[str],
        title: str,
        template_path: str | None = None,
        progress_callback=None,
        shot_spec_metadata: dict | None = None,
    ) -> tuple[str, dict]:
        """Generate a small candidate set, QA-score it, repair once, then select.

        This is the product pipeline:
          1. Generate N constrained candidates from the same template.
          2. Ask a structured judge for identity/style/artifact scores.
          3. Select the best candidate instead of revising in free text.
          4. Apply deterministic face-swap identity repair to the selected image.

        The old resemblance loop remains below as a legacy fallback and for
        tests, but production should prefer this method because it turns the
        image model's randomness into a controlled sampling-and-selection step.
        """
        self._ensure_session(session_id, photo_paths[0] if photo_paths else "")
        self._turn_counts[session_id] = 1

        ref_photos = photo_paths[:MAX_CHARACTER_REFERENCES]
        eval_ref_photos = photo_paths[:MAX_IDENTITY_PACK_REFERENCES]
        candidate_count = PIPELINE_CANDIDATE_COUNT
        candidates: list[dict] = []
        provider_invocations: list[dict] = []
        agent_actions: list[dict] = []
        identity_pack = build_identity_pack_metadata(eval_ref_photos)
        shot_spec = shot_spec_metadata or build_shot_spec_metadata(
            prompt, title, template_path
        )
        identity_thresholds = identity_threshold_profile(shot_spec)

        metadata = {
            "pipeline": "controlled_candidate_v2",
            # Compatibility with the old frontend/storage wording.
            "iterations": 0,
            "final_score": None,
            "history": [],
            "identity_pack": identity_pack,
            "shot_spec": shot_spec,
            "allowed_actions": PIPELINE_ALLOWED_ACTIONS,
            "budget": {
                "initial_candidates": candidate_count,
                "max_regenerations": MAX_PIPELINE_REGENERATIONS,
                "max_local_edits": MAX_PIPELINE_LOCAL_EDITS,
                "max_identity_repairs": MAX_PIPELINE_IDENTITY_REPAIRS,
                "max_total_api_cost": MAX_PIPELINE_TOTAL_API_COST,
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
                "prompt_version": PIPELINE_PROMPT_VERSION,
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
                    f"Generating candidate {idx}/{candidate_count}…",
                )

            candidate_prompt = build_candidate_prompt(
                prompt, len(ref_photos), idx, candidate_count
            )
            started_at = time.time()
            filepath = self.client.start_conversation(
                prompt=candidate_prompt,
                photo_paths=ref_photos,
                title=f"{title}_cand{idx}",
                template_path=template_path,
                editing_mode=True,
            )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"create_{idx}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=PIPELINE_PROMPT_VERSION,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=idx,
                parent_candidate_id=None,
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=invocation_cost,
                result_status="success",
            )
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + invocation_cost, 4
            )

            if progress_callback:
                progress_callback(
                    idx, candidate_count, "candidate_judging",
                    f"Quality scoring candidate {idx}/{candidate_count}…",
                )

            judgement = self._judge_current_candidate(filepath, eval_ref_photos)
            candidate = {
                "index": idx,
                "candidate_id": f"cand_{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._aggregate_quality_score(judgement),
                "gate_status": self._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "repair": None,
            }
            action = self._decide_candidate_action(
                judgement,
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
            raise RuntimeError("Initial candidate budget exhausted before generation")

        selected = self._select_candidate(candidates)
        if selected is None:
            # Extremely defensive fallback: start_conversation succeeded at least
            # once if candidates is non-empty, so use the last candidate rather
            # than failing the paid flow after a judge-format problem.
            selected = candidates[-1]

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
                    f"Regenerating from original references {regen_no}/"
                    f"{metadata['budget']['max_regenerations']}…",
                )

            candidate_prompt = build_candidate_prompt(
                prompt, len(ref_photos), idx, max_steps
            )
            started_at = time.time()
            filepath = self.client.start_conversation(
                prompt=candidate_prompt,
                photo_paths=ref_photos,
                title=f"{title}_regen{regen_no}",
                template_path=template_path,
                editing_mode=True,
            )
            invocation = build_provider_invocation_metadata(
                invocation_id=f"regenerate_{regen_no}",
                operation="CREATE_FROM_REFERENCES",
                prompt_version=PIPELINE_PROMPT_VERSION,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=idx,
                parent_candidate_id=selected.get("candidate_id"),
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=regen_cost,
                result_status="success",
            )
            provider_invocations.append(invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + regen_cost, 4
            )

            judgement = self._judge_current_candidate(filepath, eval_ref_photos)
            candidate = {
                "index": idx,
                "candidate_id": f"cand_{idx}",
                "path": filepath,
                "filename": Path(filepath).name,
                "judgement": judgement,
                "aggregate_score": self._aggregate_quality_score(judgement),
                "gate_status": self._candidate_gate_status(
                    judgement,
                    identity_thresholds,
                ),
                "provider_invocation_id": invocation["invocation_id"],
                "selected": False,
                "repair": None,
                "regenerated_from_candidate_id": selected.get("candidate_id"),
            }
            action = self._decide_candidate_action(
                judgement,
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

            selected = self._select_candidate(candidates) or candidate

        if (
            selected.get("agent_action", {}).get("action")
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

        selected["selected"] = True
        filepath = selected["path"]
        selected_scores = selected.get("judgement", {}).get("scores", {})
        selected_identity = selected_scores.get("identity")
        selected_action = selected.get("agent_action", {})
        local_edit_needed = selected_action.get("action") == "LOCAL_EDIT"
        repair_needed = (
            selected_action.get("action") == "IDENTITY_REPAIR"
            or self._should_apply_identity_repair(
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
                    "Applying identity repair and final checks…"
                    if repair_needed
                    else "Applying local artifact edit and final checks…"
                    if local_edit_needed
                    else "Candidate passed QA; skipping identity repair…"
                ),
            )

        if local_edit_needed:
            local_edit_cost = estimate_cost("LOCAL_EDIT", len(ref_photos))
            if (
                metadata["budget"]["estimated_cost_used"] + local_edit_cost
                > metadata["budget"]["max_total_api_cost"]
            ):
                local_edit_needed = False
                for action_record in agent_actions:
                    if action_record.get("candidate_id") == selected.get("candidate_id"):
                        action_record["executed"] = False
                        action_record["skip_reason"] = "max_total_api_cost_reached"

        if local_edit_needed:
            started_at = time.time()
            if hasattr(self.client, "_last_image_path"):
                self.client._last_image_path = filepath
            edit_prompt = build_local_edit_prompt(selected.get("judgement", {}))
            edited_path = self.client.converse(
                edit_prompt,
                title=f"{title}_cand{selected['index']}_local_edit",
                turn_number=2,
            )
            filepath = edited_path
            metadata["budget"]["local_edits_used"] += 1
            local_invocation = build_provider_invocation_metadata(
                invocation_id=f"local_edit_{selected['index']}",
                operation="LOCAL_EDIT",
                prompt_version=PIPELINE_PROMPT_VERSION,
                reference_ids=reference_ids,
                reference_roles=reference_manifest,
                candidate_index=selected["index"],
                parent_candidate_id=selected["candidate_id"],
                shot_id=shot_spec.get("shot_id"),
                latency_ms=int((time.time() - started_at) * 1000),
                cost=local_edit_cost,
                result_status="success",
            )
            provider_invocations.append(local_invocation)
            metadata["budget"]["estimated_cost_used"] = round(
                metadata["budget"]["estimated_cost_used"] + local_edit_cost, 4
            )
            edited_judgement = self._judge_current_candidate(
                filepath,
                eval_ref_photos,
            )
            selected["local_edit"] = {
                "action": "local_edit",
                "applied": True,
                "output_filename": Path(filepath).name,
                "post_edit_judgement": edited_judgement,
            }
            selected["aggregate_score"] = self._aggregate_quality_score(
                edited_judgement
            )
            selected["gate_status"] = self._candidate_gate_status(
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
                    invocation_id=f"identity_repair_{selected['index']}",
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
                # Re-score the repaired image when possible. The client judge
                # reads _last_image_path, so point it at the repaired artifact.
                if hasattr(self.client, "_last_image_path"):
                    self.client._last_image_path = filepath
                repaired_judgement = self._judge_current_candidate(
                    filepath,
                    eval_ref_photos,
                )
                selected["repair"]["post_repair_judgement"] = repaired_judgement
                selected["aggregate_score"] = self._aggregate_quality_score(
                    repaired_judgement
                )
                selected["gate_status"] = self._candidate_gate_status(
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
                "message": "QA accepted candidate; no identity repair needed",
            }

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
        metadata["shortlist"] = self._candidate_shortlist(candidates, limit=2)
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
                f"Selected candidate {selected['index']}/{candidate_count}"
                + (f" · identity {final_score}/10" if final_score is not None else "")
            )
            progress_callback(
                selected["index"], candidate_count, "accepted", detail
            )

        return filepath, metadata

    def _judge_current_candidate(
        self,
        image_path: str | None = None,
        reference_photo_paths: list[str] | None = None,
    ) -> dict:
        """Ask the model for structured QA and return a normalized dict."""
        try:
            response_text = self.client.converse_text(
                QUALITY_JUDGE_PROMPT,
                timeout=settings.gemini_wait_timeout,
            )
            judgement = self._parse_quality_judge_response(response_text)
            judgement["raw_response"] = response_text
        except Exception as exc:
            judgement = {
                "scores": {
                    "identity": None,
                    "face_quality": None,
                    "style_match": None,
                    "artifact": None,
                    "commercial_readiness": None,
                },
                "hard_failures": ["judge_failed"],
                "recommended_action": "retry",
                "notes": f"Judge failed: {exc}",
                "raw_response": "",
                "error": str(exc),
            }
        if image_path and Path(image_path).exists():
            judgement = self._merge_local_quality(
                judgement, self._local_image_quality_check(image_path)
            )
            if reference_photo_paths:
                judgement = self._merge_identity_quality(
                    judgement,
                    self._local_identity_similarity_check(
                        image_path, reference_photo_paths
                    ),
                )
        judgement["quality_evaluation"] = self._quality_evaluation_summary(judgement)
        return judgement

    @staticmethod
    def _parse_quality_judge_response(text: str) -> dict:
        """Parse the structured QA JSON, with a score-regex fallback."""
        def empty(notes: str = "") -> dict:
            return {
                "scores": {
                    "identity": None,
                    "face_quality": None,
                    "style_match": None,
                    "artifact": None,
                    "commercial_readiness": None,
                },
                "hard_failures": [],
                "recommended_action": "retry",
                "notes": notes,
            }

        if not text:
            return empty("Empty judge response")

        data = None
        stripped = text.strip()
        try:
            data = json.loads(stripped)
        except Exception:
            match = re.search(r"\{.*\}", stripped, flags=re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None

        if not isinstance(data, dict):
            score, feedback = GeminiWorker._parse_judge_response(text)
            out = empty(feedback or "Could not parse structured QA JSON")
            out["scores"]["identity"] = score
            out["scores"]["commercial_readiness"] = score
            out["recommended_action"] = (
                "accept" if score is not None and score >= QUALITY_ACCEPT_THRESHOLD
                else "retry"
            )
            return out

        scores_in = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        scores = {}
        for key in (
            "identity",
            "face_quality",
            "style_match",
            "artifact",
            "commercial_readiness",
        ):
            value = scores_in.get(key)
            try:
                value = int(float(value) + 0.5)
                value = max(1, min(10, value))
            except Exception:
                value = None
            scores[key] = value

        failures = data.get("hard_failures", [])
        if not isinstance(failures, list):
            failures = []
        failures = [str(x) for x in failures if str(x)]

        action = str(data.get("recommended_action", "retry"))
        if action not in {"accept", "face_swap", "retry", "discard"}:
            action = "retry"

        return {
            "scores": scores,
            "hard_failures": failures,
            "recommended_action": action,
            "notes": str(data.get("notes", ""))[:500],
        }

    @staticmethod
    def _local_image_quality_check(image_path: str) -> dict:
        """Deterministic local quality checks for generated portrait images.

        This deliberately avoids identity claims. It only checks things that are
        cheap and objective enough to gate production output: readable image,
        resolution, blur, and whether a plausible single face is detected.
        """
        result = {
            "scores": {
                "face_quality": None,
                "artifact": None,
                "commercial_readiness": None,
            },
            "hard_failures": [],
            "measurements": {},
            "notes": "",
        }
        try:
            import cv2
        except Exception as exc:
            result["notes"] = f"local_quality_unavailable: {exc}"
            return result

        img = cv2.imread(str(image_path))
        if img is None:
            result["scores"].update({
                "face_quality": 1,
                "artifact": 1,
                "commercial_readiness": 1,
            })
            result["hard_failures"].append("unreadable_image")
            result["notes"] = "Could not read generated image"
            return result

        height, width = img.shape[:2]
        min_dim = min(width, height)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if min_dim >= 768:
            resolution_score = 10
        elif min_dim >= 512:
            resolution_score = 8
        elif min_dim >= 384:
            resolution_score = 6
        else:
            resolution_score = 3
            result["hard_failures"].append("bad_resolution")

        if blur_var >= 120:
            sharpness_score = 10
        elif blur_var >= 80:
            sharpness_score = 8
        elif blur_var >= 45:
            sharpness_score = 6
        elif blur_var >= 25:
            sharpness_score = 4
            result["hard_failures"].append("too_blurry")
        else:
            sharpness_score = 2
            result["hard_failures"].append("too_blurry")

        faces = []
        try:
            cascade_path = (
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(str(cascade_path))
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(80, 80),
            )
        except Exception:
            faces = []

        face_score = 10
        if len(faces) == 0:
            face_score = 3
            result["hard_failures"].append("no_face")
        elif len(faces) > 1:
            face_score = 6
            result["hard_failures"].append("multiple_faces")
        else:
            x, y, fw, fh = [int(v) for v in faces[0]]
            face_area_ratio = (fw * fh) / float(width * height)
            cx = x + fw / 2.0
            cy = y + fh / 2.0
            center_dx = abs(cx - width / 2.0) / width
            center_dy = abs(cy - height / 2.0) / height
            if face_area_ratio < 0.05 or face_area_ratio > 0.55:
                face_score = min(face_score, 7)
                result["hard_failures"].append("face_scale_unusual")
            if center_dx > 0.22 or center_dy > 0.22:
                face_score = min(face_score, 7)
                result["hard_failures"].append("face_off_center")
            result["measurements"].update({
                "face_area_ratio": round(face_area_ratio, 4),
                "face_center_dx": round(center_dx, 4),
                "face_center_dy": round(center_dy, 4),
            })

        face_quality = min(resolution_score, sharpness_score, face_score)
        artifact = min(resolution_score, sharpness_score)
        commercial = min(face_quality, artifact)

        result["scores"].update({
            "face_quality": face_quality,
            "artifact": artifact,
            "commercial_readiness": commercial,
        })
        result["measurements"].update({
            "width": width,
            "height": height,
            "min_dim": min_dim,
            "blur_variance": round(blur_var, 2),
            "face_count": int(len(faces)),
        })
        if result["hard_failures"]:
            result["notes"] = ", ".join(result["hard_failures"])
        return result

    @staticmethod
    def _merge_local_quality(judgement: dict, local_quality: dict) -> dict:
        """Merge deterministic local quality gates into VLM QA output."""
        merged = dict(judgement)
        scores = dict(merged.get("scores") or {})
        local_scores = local_quality.get("scores") or {}
        for key in ("face_quality", "artifact", "commercial_readiness"):
            local_value = local_scores.get(key)
            if local_value is None:
                continue
            current = scores.get(key)
            scores[key] = local_value if current is None else min(current, local_value)
        merged["scores"] = scores

        failures = list(merged.get("hard_failures") or [])
        for failure in local_quality.get("hard_failures") or []:
            if failure not in failures:
                failures.append(failure)
        merged["hard_failures"] = failures
        merged["local_quality"] = local_quality

        severe = {
            "unreadable_image",
            "no_face",
            "multiple_faces",
            "bad_resolution",
            "too_blurry",
        }
        if severe.intersection(failures):
            merged["recommended_action"] = "discard"
            note = merged.get("notes", "")
            local_note = local_quality.get("notes", "")
            merged["notes"] = (
                f"{note} Local gate: {local_note}".strip()
                if note else f"Local gate: {local_note}"
            )[:500]
        merged["quality_evaluation"] = GeminiWorker._quality_evaluation_summary(merged)
        return merged

    def _local_identity_similarity_check(
        self,
        generated_path: str,
        reference_photo_paths: list[str],
    ) -> dict:
        """Score same-person similarity using local face embeddings."""
        result = {
            "score": None,
            "cosine_similarity": None,
            "reference_consistency": None,
            "hard_failures": [],
            "measurements": {},
            "notes": "",
        }
        app = self._get_identity_app()
        if app is None:
            result["notes"] = "identity_scorer_unavailable"
            return result

        try:
            import cv2
            import numpy as np
        except Exception as exc:
            result["notes"] = f"identity_dependencies_unavailable: {exc}"
            return result

        def best_embedding(path: str):
            img = cv2.imread(str(path))
            if img is None:
                return None, 0
            faces = app.get(img)
            if not faces:
                return None, 0
            face = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            return face.normed_embedding, len(faces)

        ref_embeddings = []
        ref_face_counts = []
        for path in reference_photo_paths[:MAX_IDENTITY_PACK_REFERENCES]:
            emb, count = best_embedding(path)
            ref_face_counts.append(count)
            if emb is not None:
                ref_embeddings.append(emb)

        gen_embedding, gen_face_count = best_embedding(generated_path)
        result["measurements"] = {
            "reference_count": len(reference_photo_paths[:MAX_IDENTITY_PACK_REFERENCES]),
            "reference_faces_detected": len(ref_embeddings),
            "reference_face_counts": ref_face_counts,
            "generated_face_count": gen_face_count,
        }

        if gen_embedding is None:
            result["hard_failures"].append("identity_no_generated_face")
            result["notes"] = "No face embedding detected in generated image"
            return result
        if not ref_embeddings:
            result["hard_failures"].append("identity_no_reference_face")
            result["notes"] = "No face embedding detected in reference photos"
            return result

        ref_profile = np.mean(np.stack(ref_embeddings), axis=0)
        ref_profile = ref_profile / max(float(np.linalg.norm(ref_profile)), 1e-8)
        cosine = float(np.dot(gen_embedding, ref_profile))

        ref_consistency = None
        if len(ref_embeddings) >= 2:
            sims = []
            for i in range(len(ref_embeddings)):
                for j in range(i + 1, len(ref_embeddings)):
                    sims.append(float(np.dot(ref_embeddings[i], ref_embeddings[j])))
            ref_consistency = float(sum(sims) / len(sims)) if sims else None

        score = self._identity_cosine_to_score(cosine)
        if score < IDENTITY_REPAIR_THRESHOLD:
            result["hard_failures"].append("identity_too_low")

        result.update({
            "score": score,
            "cosine_similarity": round(cosine, 4),
            "reference_consistency": (
                round(ref_consistency, 4) if ref_consistency is not None else None
            ),
        })
        result["measurements"]["identity_accept_cosine"] = IDENTITY_COSINE_ACCEPT_THRESHOLD
        if result["hard_failures"]:
            result["notes"] = ", ".join(result["hard_failures"])
        return result

    @staticmethod
    def _identity_cosine_to_score(cosine: float) -> int:
        """Map InsightFace cosine similarity to the existing 1-10 score scale."""
        if cosine >= 0.58:
            return 10
        if cosine >= 0.52:
            return 9
        if cosine >= IDENTITY_COSINE_ACCEPT_THRESHOLD:
            return 8
        if cosine >= 0.40:
            return 7
        if cosine >= 0.34:
            return 6
        if cosine >= 0.28:
            return 5
        return 3

    @staticmethod
    def _merge_identity_quality(judgement: dict, identity_quality: dict) -> dict:
        """Merge local face-embedding identity score into VLM QA output."""
        merged = dict(judgement)
        scores = dict(merged.get("scores") or {})
        local_score = identity_quality.get("score")
        if local_score is not None:
            current = scores.get("identity")
            scores["identity"] = local_score if current is None else min(current, local_score)
        merged["scores"] = scores

        failures = list(merged.get("hard_failures") or [])
        for failure in identity_quality.get("hard_failures") or []:
            if failure not in failures:
                failures.append(failure)
        merged["hard_failures"] = failures
        merged["identity_quality"] = identity_quality

        if "identity_too_low" in failures:
            merged["recommended_action"] = "face_swap"
            note = merged.get("notes", "")
            local_note = identity_quality.get("notes", "")
            merged["notes"] = (
                f"{note} Local identity: {local_note}".strip()
                if note else f"Local identity: {local_note}"
            )[:500]
        elif any(f in failures for f in ("identity_no_generated_face", "identity_no_reference_face")):
            merged["recommended_action"] = "retry"
        merged["quality_evaluation"] = GeminiWorker._quality_evaluation_summary(merged)
        return merged

    @staticmethod
    def _score_status(score, threshold: float = QUALITY_ACCEPT_THRESHOLD) -> str:
        if score is None:
            return "unchecked"
        try:
            return "pass" if float(score) >= threshold else "fail"
        except Exception:
            return "unchecked"

    @staticmethod
    def _quality_evaluation_summary(judgement: dict) -> dict:
        """Return the product QA schema from VLM + local detector results."""
        scores = judgement.get("scores") or {}
        failures = list(judgement.get("hard_failures") or [])
        local = judgement.get("local_quality") or {}
        identity = judgement.get("identity_quality") or {}

        identity_score = scores.get("identity")
        face_quality_score = scores.get("face_quality")
        artifact_score = scores.get("artifact")
        style_score = scores.get("style_match")
        readiness_score = scores.get("commercial_readiness")
        prompt_score = (
            round(style_score / 10.0, 4)
            if isinstance(style_score, (int, float)) else None
        )
        aesthetic_source_scores = [
            value for value in (style_score, readiness_score)
            if isinstance(value, (int, float))
        ]
        aesthetic_score = (
            round(sum(aesthetic_source_scores) / (10.0 * len(aesthetic_source_scores)), 4)
            if aesthetic_source_scores else None
        )

        return {
            "identity": {
                "score": identity_score,
                "status": GeminiWorker._score_status(identity_score),
                "cosine_similarity": identity.get("cosine_similarity"),
                "reference_consistency": identity.get("reference_consistency"),
                "issues": [
                    item for item in failures
                    if str(item).startswith("identity_") or item == "identity_too_low"
                ],
                "measurements": identity.get("measurements") or {},
            },
            "face_quality": {
                "score": face_quality_score,
                "status": GeminiWorker._score_status(face_quality_score),
                "issues": [
                    item for item in (local.get("hard_failures") or failures)
                    if item in {
                        "unreadable_image",
                        "no_face",
                        "multiple_faces",
                        "too_blurry",
                        "bad_resolution",
                        "face_scale_unusual",
                        "face_off_center",
                    }
                ],
                "measurements": local.get("measurements") or {},
            },
            "artifacts": {
                "score": artifact_score,
                "status": GeminiWorker._score_status(artifact_score),
                "issues": [
                    item for item in failures
                    if item in {
                        "bad_artifacts",
                        "face_distorted",
                        "too_blurry",
                        "bad_resolution",
                    }
                ],
            },
            "composition": {
                "score": style_score,
                "status": GeminiWorker._score_status(style_score),
                "issues": [
                    item for item in failures
                    if item in {"wrong_style", "global_composition_failed"}
                ],
            },
            "prompt_adherence": {
                "score": prompt_score,
                "status": GeminiWorker._score_status(style_score),
            },
            "aesthetic": {
                "score": aesthetic_score,
                "status": GeminiWorker._score_status(readiness_score),
            },
            "safety": {
                "status": "fail" if "unsafe_content" in failures else "pass",
                "issues": [
                    item for item in failures
                    if item == "unsafe_content"
                ],
            },
        }

    @staticmethod
    def _candidate_gate_status(
        judgement: dict,
        identity_thresholds: dict | None = None,
    ) -> dict:
        """Evaluate product hard gates before any aesthetic ranking."""
        scores = judgement.get("scores", {}) or {}
        failures = set(judgement.get("hard_failures") or [])
        identity = scores.get("identity")
        face_quality = scores.get("face_quality")
        artifact = scores.get("artifact")
        commercial = scores.get("commercial_readiness")
        thresholds = identity_thresholds or identity_threshold_profile()
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )

        safety_pass = "unsafe_content" not in failures
        face_detected = not any(
            failure in failures
            for failure in ("no_face", "identity_no_generated_face")
        )
        identity_pass = (
            identity is not None and identity >= identity_pass_threshold
        )
        severe_quality_fail = any(
            failure in failures
            for failure in (
                "face_distorted",
                "bad_artifacts",
                "unreadable_image",
                "bad_resolution",
                "too_blurry",
                "multiple_faces",
            )
        )
        quality_pass = (
            face_quality is None or face_quality >= QUALITY_ACCEPT_THRESHOLD
        ) and (
            artifact is None or artifact >= QUALITY_ACCEPT_THRESHOLD
        ) and (
            commercial is None or commercial >= QUALITY_ACCEPT_THRESHOLD
        )
        hard_gates_pass = (
            safety_pass
            and face_detected
            and identity_pass
            and quality_pass
            and not severe_quality_fail
        )
        hard_gate_failures = []
        if not safety_pass:
            hard_gate_failures.append("unsafe_content")
        if not face_detected:
            hard_gate_failures.append("no_usable_face_detected")
        if not identity_pass:
            hard_gate_failures.append("identity_fail")
        if not quality_pass:
            hard_gate_failures.append("quality_below_threshold")
        if severe_quality_fail:
            hard_gate_failures.append("severe_quality_failure")
        return {
            "safety_pass": safety_pass,
            "face_detected": face_detected,
            "identity_pass": identity_pass,
            "identity_pass_threshold": identity_pass_threshold,
            "identity_repair_threshold": float(
                thresholds.get(
                    "identity_repair_threshold",
                    IDENTITY_REPAIR_THRESHOLD,
                )
            ),
            "identity_threshold_profile": thresholds.get("profile", "closeup"),
            "quality_pass": quality_pass,
            "severe_quality_fail": severe_quality_fail,
            "hard_gates_pass": hard_gates_pass,
            "hard_gate_failures": hard_gate_failures,
        }

    @staticmethod
    def _decide_candidate_action(
        judgement: dict,
        edit_count: int = 0,
        identity_repairs: int = 0,
        identity_thresholds: dict | None = None,
    ) -> dict:
        """Bounded state-machine action for one evaluated candidate."""
        scores = judgement.get("scores", {}) or {}
        failures = set(judgement.get("hard_failures") or [])
        action_hint = judgement.get("recommended_action")
        identity = scores.get("identity")
        style_match = scores.get("style_match")
        artifact = scores.get("artifact")
        thresholds = identity_thresholds or identity_threshold_profile()
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )
        identity_repair_threshold = float(
            thresholds.get("identity_repair_threshold", IDENTITY_REPAIR_THRESHOLD)
        )
        gate = GeminiWorker._candidate_gate_status(judgement, thresholds)

        if action_hint == "discard":
            return {
                "action": "DROP_CANDIDATE",
                "reason": "judge_or_local_gate_marked_discard",
            }
        if not gate["safety_pass"]:
            return {"action": "DROP_CANDIDATE", "reason": "unsafe_content"}
        if not gate["face_detected"]:
            return {
                "action": "REGENERATE_FROM_ORIGINAL",
                "reason": "no_usable_face_detected",
            }
        if identity is None:
            if identity_repairs < MAX_PIPELINE_IDENTITY_REPAIRS:
                return {
                    "action": "IDENTITY_REPAIR",
                    "reason": "identity_unverified",
                }
            return {"action": "DROP_CANDIDATE", "reason": "identity_unverified"}
        if identity < identity_repair_threshold:
            return {
                "action": "REGENERATE_FROM_ORIGINAL",
                "reason": "identity_below_repair_threshold",
            }
        if identity < identity_pass_threshold:
            good_composition = style_match is None or style_match >= QUALITY_ACCEPT_THRESHOLD
            if identity_repairs < MAX_PIPELINE_IDENTITY_REPAIRS and good_composition:
                return {
                    "action": "IDENTITY_REPAIR",
                    "reason": "identity_gray_zone_with_usable_composition",
                }
            return {
                "action": "REGENERATE_FROM_ORIGINAL",
                "reason": "identity_gray_zone_not_worth_repair",
            }
        if gate["severe_quality_fail"]:
            return {
                "action": "REGENERATE_FROM_ORIGINAL",
                "reason": "global_quality_failure",
            }
        if (
            artifact is not None
            and artifact < QUALITY_ACCEPT_THRESHOLD
            and edit_count < MAX_PIPELINE_LOCAL_EDITS
        ):
            return {"action": "LOCAL_EDIT", "reason": "local_artifact"}
        if gate["hard_gates_pass"]:
            return {"action": "ACCEPT", "reason": "all_hard_gates_pass"}
        return {"action": "DROP_CANDIDATE", "reason": "quality_below_delivery_gate"}

    @staticmethod
    def _aggregate_quality_score(judgement: dict) -> float:
        scores = judgement.get("scores", {})
        weights = {
            "identity": 0.45,
            "face_quality": 0.20,
            "style_match": 0.20,
            "artifact": 0.10,
            "commercial_readiness": 0.05,
        }
        total = 0.0
        used = 0.0
        for key, weight in weights.items():
            value = scores.get(key)
            if value is None:
                continue
            total += float(value) * weight
            used += weight
        if used == 0:
            return 0.0
        penalty = 0.0
        if judgement.get("recommended_action") == "discard":
            penalty -= 1.0
        elif judgement.get("hard_failures"):
            penalty -= 1.5
        return min(10.0, max(0.0, round(total / used + penalty, 2)))

    @staticmethod
    def _should_apply_identity_repair(
        judgement: dict,
        identity_thresholds: dict | None = None,
    ) -> bool:
        """Return True only for identity-gray-zone candidates worth repairing."""
        scores = judgement.get("scores", {})
        identity = scores.get("identity")
        failures = set(judgement.get("hard_failures") or [])
        action = judgement.get("recommended_action")
        thresholds = identity_thresholds or identity_threshold_profile()
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )
        identity_repair_threshold = float(
            thresholds.get("identity_repair_threshold", IDENTITY_REPAIR_THRESHOLD)
        )

        if identity is None:
            # If the judge failed to score identity, prefer a repair attempt over
            # silently accepting an unverified face.
            return True
        if identity < identity_repair_threshold:
            # Below the repair threshold, the state machine should regenerate
            # from the original Identity Pack or drop the branch. Repairing a
            # clearly wrong face tends to lock in drift.
            return False
        if identity >= identity_pass_threshold:
            return False
        return (
            action == "face_swap"
            or "identity_too_low" in failures
            or identity < identity_pass_threshold
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
    def _select_candidate(candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        deliverable = [
            c for c in candidates
            if c.get("gate_status", {}).get("hard_gates_pass")
        ]
        if deliverable:
            return max(deliverable, key=lambda c: c.get("aggregate_score", 0.0))

        locally_editable = [
            c for c in candidates
            if c.get("agent_action", {}).get("action") == "LOCAL_EDIT"
        ]
        if locally_editable:
            return max(locally_editable, key=lambda c: c.get("aggregate_score", 0.0))

        repairable = [
            c for c in candidates
            if c.get("agent_action", {}).get("action") == "IDENTITY_REPAIR"
        ]
        if repairable:
            return max(repairable, key=lambda c: c.get("aggregate_score", 0.0))

        regeneratable = [
            c for c in candidates
            if c.get("agent_action", {}).get("action") == "REGENERATE_FROM_ORIGINAL"
        ]
        if regeneratable:
            return max(regeneratable, key=lambda c: c.get("aggregate_score", 0.0))

        return max(candidates, key=lambda c: c.get("aggregate_score", 0.0))

    @staticmethod
    def _candidate_shortlist(candidates: list[dict], limit: int = 2) -> list[dict]:
        """Public candidate-funnel summary: top retained candidates, no paths."""
        ranked = sorted(
            candidates,
            key=lambda c: (
                bool(c.get("gate_status", {}).get("hard_gates_pass")),
                c.get("aggregate_score", 0.0),
            ),
            reverse=True,
        )
        shortlist = []
        for rank, candidate in enumerate(ranked[:limit], start=1):
            gate = candidate.get("gate_status") or {}
            action = candidate.get("agent_action") or {}
            shortlist.append({
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "candidate_index": candidate.get("index"),
                "filename": candidate.get("filename"),
                "aggregate_score": candidate.get("aggregate_score"),
                "hard_gates_pass": gate.get("hard_gates_pass"),
                "hard_gate_failures": gate.get("hard_gate_failures", []),
                "recommended_action": action.get("action"),
                "action_reason": action.get("reason"),
                "selected": bool(candidate.get("selected")),
            })
        return shortlist

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
        filepath = self.client.start_conversation(
            prompt=wrapped_prompt,
            photo_paths=ref_photos,
            title=f"{title}_initial",
            template_path=template_path,
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
                    response_text = self.client.converse_text(
                        RESEMBLANCE_JUDGE_PROMPT,
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

            filepath = self.client.converse(
                prompt=revision_instruction,
                title=f"{title}_iter{i}",
                turn_number=turn,
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
        # gemini-3.1-flash-image-preview frequently returns DECIMAL scores
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
    ) -> str:
        """Run a revision job in the same conversation. Returns path to saved image."""
        if self.active_session_id != session_id:
            raise RuntimeError(
                f"Session {session_id} is not the active conversation "
                f"(active: {self.active_session_id}). "
                "Cannot revise without an active generation."
            )

        turn = self._turn_counts.get(session_id, 1) + 1
        self._turn_counts[session_id] = turn

        filepath = self.client.converse(
            prompt=instruction,
            title=title,
            turn_number=turn,
        )
        return filepath

    def end_session(self, session_id: str):
        """End the conversation for a session."""
        if self.active_session_id == session_id:
            try:
                self.client.end_conversation()
            except Exception:
                pass
            self.active_session_id = None

    def _ensure_session(self, session_id: str, photo_path: str = ""):
        """Make sure the Gemini conversation is for the given session."""
        if self.active_session_id == session_id:
            return  # already in the right session

        # End old session if any
        if self.active_session_id is not None:
            try:
                self.client.end_conversation()
            except Exception:
                pass
            # Open fresh chat
            self.client.ensure_gemini_page()
            self.client._new_chat()

        self.active_session_id = session_id
