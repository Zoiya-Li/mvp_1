"""Pydantic models for the PortraitAI API."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .storage import utcnow


# ── Enums ──────────────────────────────────────────────

class StyleKey(str, Enum):
    id_photo = "id_photo"
    business = "business"
    academic = "academic"
    social = "social"
    jk_portrait = "jk_portrait"
    chinese_style = "chinese_style"
    fashion = "fashion"
    cinematic = "cinematic"
    creative = "creative"


class SessionStatus(str, Enum):
    created = "created"
    uploading = "uploading"
    ready = "ready"
    generating = "generating"
    hero_preview_ready = "hero_preview_ready"
    reviewing = "reviewing"
    done = "done"
    failed = "failed"


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class JobType(str, Enum):
    generate = "generate"
    revise = "revise"
    hero_preview = "hero_preview"
    full_set = "full_set"


class FeedbackEvent(str, Enum):
    downloaded = "downloaded"
    selected = "selected"
    looks_like_me = "looks_like_me"
    not_like_me = "not_like_me"
    bad_artifacts = "bad_artifacts"
    not_saved = "not_saved"


# ── Request models ─────────────────────────────────────

class CreateSessionRequest(BaseModel):
    style: StyleKey
    gender: Literal["male", "female"]


class ReviseRequest(BaseModel):
    instruction: str = Field(..., min_length=2, max_length=500)


class UserFeedbackRequest(BaseModel):
    event: FeedbackEvent
    reason: str | None = Field(default=None, max_length=500)
    score: int | None = Field(default=None, ge=0, le=2)


class UserFeedbackResponse(BaseModel):
    feedback_id: str
    session_id: str
    image_id: str
    event: FeedbackEvent
    reason: str | None = None
    score: int | None = None
    created_at: datetime


# ── Agent pipeline contract models ───────────────────────

PipelineOperation = Literal[
    "CREATE_FROM_REFERENCES",
    "LOCAL_EDIT",
    "IDENTITY_REPAIR",
    "UPSCALE",
    "FINAL_RENDER",
]

PipelineAgentActionName = Literal[
    "ACCEPT",
    "LOCAL_EDIT",
    "IDENTITY_REPAIR",
    "REGENERATE_FROM_ORIGINAL",
    "REGENERATE_WITH_POSE_REFERENCE",
    "DROP_CANDIDATE",
    "REQUEST_BETTER_REFERENCE",
]


class ReferenceAsset(BaseModel):
    reference_id: str
    slot: str | None = None
    role: str
    angle_hint: str | None = None
    expression_hint: str | None = None
    usage: str
    priority: int
    filename: str
    input_quality_status: str = "checked_before_generation"


class TemporaryFaceTemplate(BaseModel):
    storage: str = "in_memory_task_scope"
    stores_embedding_in_metadata: bool = False
    built_from_reference_ids: list[str] = Field(default_factory=list)


class IdentityPack(BaseModel):
    reference_images: list[ReferenceAsset]
    primary_reference_ids: list[str] = Field(default_factory=list)
    reference_role_order: list[str] = Field(default_factory=list)
    appearance_constraints: dict[str, Any] = Field(default_factory=dict)
    temporary_face_template: TemporaryFaceTemplate
    minimum_reference_count: int = 4
    max_identity_pack_references: int = 6
    max_generation_references: int = 4
    template_scope: str = "task_local_only"
    cross_user_search: bool = False
    persistent_face_library: bool = False
    expires_at: str = "job_finished_plus_retention_ttl"
    version: str


class StyleTemplate(BaseModel):
    style_id: str
    style_label: str | None = None
    template_id: str | None = None
    template_label: str | None = None
    prompt_version: str | None = None


class ShotPromptBlocks(BaseModel):
    identity_block: str
    scene_block: str
    style_block: str | None = None
    preservation_block: str


class ShotSpec(BaseModel):
    style_id: str | None = None
    style_label: str | None = None
    template_id: str | None = None
    template_label: str | None = None
    shot_id: str
    shot_label: str | None = None
    sequence: int | None = None
    framing: str
    pose: str
    lighting: str
    lens: str
    prompt_blocks: ShotPromptBlocks


class QualityScores(BaseModel):
    identity: float | None = None
    face_quality: float | None = None
    style_match: float | None = None
    artifact: float | None = None
    commercial_readiness: float | None = None


class IdentityQuality(BaseModel):
    score: float | None = None
    cosine_similarity: float | None = None
    reference_consistency: float | None = None
    hard_failures: list[str] = Field(default_factory=list)
    measurements: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class LocalQuality(BaseModel):
    scores: dict[str, Any] = Field(default_factory=dict)
    hard_failures: list[str] = Field(default_factory=list)
    measurements: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class EvaluationResult(BaseModel):
    scores: QualityScores = Field(default_factory=QualityScores)
    hard_failures: list[str] = Field(default_factory=list)
    recommended_action: PipelineAgentActionName | str | None = None
    notes: str | None = None
    local_quality: LocalQuality | None = None
    identity_quality: IdentityQuality | None = None


class CandidateGateStatus(BaseModel):
    safety_pass: bool | None = None
    face_detected: bool | None = None
    identity_pass: bool | None = None
    quality_pass: bool | None = None
    severe_quality_fail: bool | None = None
    hard_gates_pass: bool | None = None
    hard_gate_failures: list[str] = Field(default_factory=list)
    identity_threshold_profile: str | None = None
    identity_pass_threshold: float | None = None
    identity_repair_threshold: float | None = None


class AgentAction(BaseModel):
    action: PipelineAgentActionName
    reason: str | None = None
    candidate_id: str | None = None
    candidate_index: int | None = None
    state: str | None = None
    executed: bool | None = None
    selected_for_execution: bool | None = None


class ProviderInvocation(BaseModel):
    invocation_id: str
    provider: str
    model: str | None = None
    operation: PipelineOperation
    prompt_version: str | None = None
    reference_ids: list[str] = Field(default_factory=list)
    reference_roles: list[dict[str, Any]] = Field(default_factory=list)
    candidate_index: int | None = None
    parent_candidate_id: str | None = None
    shot_id: str | None = None
    final_asset_id: str | None = None
    latency_ms: int | None = None
    estimated_cost: float | None = None
    cost: float | None = None
    provider_capabilities: dict[str, Any] = Field(default_factory=dict)
    result_status: str = "success"


class Candidate(BaseModel):
    index: int
    candidate_id: str
    filename: str | None = None
    judgement: EvaluationResult | None = None
    aggregate_score: float | None = None
    gate_status: CandidateGateStatus | None = None
    agent_action: AgentAction | None = None
    provider_invocation_id: str | None = None
    selected: bool = False
    regenerated_from_candidate_id: str | None = None
    repair: dict[str, Any] | None = None


class FinalAsset(BaseModel):
    image_id: str
    candidate_id: str | None = None
    operation: PipelineOperation = "FINAL_RENDER"
    filename: str | None = None
    deliverable: bool = False
    visible_ai_label: bool | None = None
    metadata_ai_label: bool | None = None
    provider_invocation_id: str | None = None


class UserFeedbackRecord(BaseModel):
    feedback_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    image_id: str
    event: FeedbackEvent
    reason: str | None = None
    score: int | None = Field(default=None, ge=0, le=2)
    created_at: datetime | None = None


class PhotoJob(BaseModel):
    job_id: str
    user_id: str | None = None
    session_id: str
    identity_pack: IdentityPack
    shot_specs: list[ShotSpec] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    final_assets: list[FinalAsset] = Field(default_factory=list)
    user_feedback: list[UserFeedbackRecord] = Field(default_factory=list)
    provider_invocations: list[ProviderInvocation] = Field(default_factory=list)
    status: JobStatus = JobStatus.queued


# ── Post-processing models ─────────────────────────────

class IDPhotoSpec(str, Enum):
    one_inch = "1寸"
    two_inch = "2寸"


class BgColor(str, Enum):
    red = "red"
    blue = "blue"
    white = "white"
    gradient_gray = "gradient_gray"


class PostProcessCropRequest(BaseModel):
    image_id: str
    spec: IDPhotoSpec


class PostProcessBgRequest(BaseModel):
    image_id: str
    color: BgColor


class PostProcessCombinedRequest(BaseModel):
    image_id: str
    spec: IDPhotoSpec
    color: BgColor


class PostProcessResponse(BaseModel):
    original_image_id: str
    processed_image_id: str
    url: str
    operation: str  # "crop" | "bg" | "crop_bg"


# ── Payment / Pricing models ────────────────────────────

class PricingTier(str, Enum):
    free = "free"
    standard = "standard"
    premium = "premium"


class PaymentStatus(str, Enum):
    pending = "pending"
    paid = "paid"
    expired = "expired"
    refunded = "refunded"


# Tier permission table — aligned with landing page Pricing component.
# Overseas pricing locked at $5 Standard / $10 Pro (decided under #67; $19 was
# rejected as too close to the ChatGPT Plus price point). Amounts are USD cents.
TIER_LIMITS: dict[PricingTier, dict] = {
    PricingTier.free: {
        "label": "Free",
        "price_cents": 0,
        "max_styles": 1,
        "max_revisions": 1,
        "allow_id_photo": False,
        "allow_bg_replace": False,
        "allow_hd_download": False,
    },
    PricingTier.standard: {
        "label": "Standard",
        "price_cents": 500,  # $5
        "max_styles": 2,
        "max_revisions": 2,
        "allow_id_photo": True,
        "allow_bg_replace": True,
        "allow_hd_download": False,
    },
    PricingTier.premium: {
        "label": "Pro",
        "price_cents": 1000,  # $10
        "max_styles": 2,
        "max_revisions": 3,
        "allow_id_photo": True,
        "allow_bg_replace": True,
        "allow_hd_download": True,
    },
}


class CreatePaymentRequest(BaseModel):
    tier: PricingTier


class PaymentResponse(BaseModel):
    payment_id: str
    session_id: str
    tier: PricingTier
    status: PaymentStatus
    # Paddle hosted checkout URL the browser redirects to. None in mock mode
    # (dev auto-confirm) or for records rehydrated from the DB (the URL is
    # transient — only the live order needs it).
    checkout_url: str | None = None
    amount_cents: int
    created_at: datetime


class PaymentStatusResponse(BaseModel):
    payment_id: str
    status: PaymentStatus
    tier: PricingTier


# ── Response models ────────────────────────────────────

class GeneratedImage(BaseModel):
    image_id: str
    url: str
    prompt_id: str
    turn: int  # 1 = initial, 2+ = revision
    revised_image_id: str | None = None  # which image this revises
    created_at: datetime
    parent_image_id: str | None = None  # for post-processed variants
    operation: str | None = None  # "crop" | "bg" | "crop_bg" | None
    resemblance: dict | None = None  # {iterations, final_score, history} from agent loop


class SessionConsents(BaseModel):
    face_processing_consent: bool = False
    adult_subject_confirmed: bool = False
    no_training_by_default: bool = True
    cross_user_search_prohibited: bool = True
    long_term_face_library_prohibited: bool = True
    consented_at: datetime | None = None
    policy_version: str = "face_processing_consent_v1"


class SessionResponse(BaseModel):
    session_id: str
    owner_token: str  # returned ONCE at creation; client stores + sends back
    style: StyleKey
    gender: str
    status: SessionStatus
    uploaded_photos: list[str]
    photo_quality: dict[str, dict] = Field(default_factory=dict)
    reference_quality: dict | None = None
    session_consents: SessionConsents = Field(default_factory=SessionConsents)
    hero_preview_image_id: str | None = None
    unlocked: bool = False
    feedback_summary: dict = Field(default_factory=dict)
    pipeline_metrics: dict = Field(default_factory=dict)
    generated_images: list[GeneratedImage]
    revisions_used: int
    max_revisions: int
    created_at: datetime
    tier: PricingTier = PricingTier.free


class JobResponse(BaseModel):
    job_id: str
    session_id: str
    job_type: JobType
    status: JobStatus
    prompt_id: str | None = None
    shot_spec: dict | None = None
    progress: float = 0.0
    result_image: GeneratedImage | None = None
    error: str | None = None
    position_in_queue: int = 0


class StyleInfo(BaseModel):
    key: StyleKey
    label: str
    label_en: str = ""
    use_cases: list[str]
    templates: list[TemplateInfo]


class TemplateInfo(BaseModel):
    id: str
    gender: str
    label: str
    template_image: str | None = None


class StyleListResponse(BaseModel):
    styles: list[StyleInfo]


class QueueStatusResponse(BaseModel):
    queue_length: int
    active_session: str | None
    estimated_wait_seconds: float


# ── Internal state (not exposed directly) ──────────────

class SessionState:
    """Server-side session state stored in memory."""

    def __init__(self, session_id: str, style: StyleKey, gender: str,
                 owner_token: str):
        self.session_id = session_id
        self.style = style
        self.gender = gender
        self.owner_token = owner_token  # secret credential — never listed
        self.status = SessionStatus.created
        self.uploaded_photos: list[Path] = []
        self.photo_quality: dict[str, dict] = {}
        self.reference_quality: dict | None = None
        self.user_feedback: list[dict] = []
        self.generated_images: list[GeneratedImage] = []
        self.revisions_used: int = 0
        self.max_revisions: int = 1  # default free tier
        self.created_at = utcnow()
        self.upload_dir: Path | None = None
        self.output_dir: Path | None = None
        self.processed_images: dict[str, str] = {}  # processed_id -> original_id
        self.tier: PricingTier = PricingTier.free
        self.payment_id: str | None = None
        self.payment_status: PaymentStatus | None = None
        self.pipeline_metrics: dict = {}
        self.session_consents = SessionConsents()
        self.hero_preview_image_id: str | None = None
        self.hero_preview_generated: bool = False
        self.unlocked: bool = False

    def record_session_consents(
        self,
        *,
        face_processing_consent: bool,
        adult_subject_confirmed: bool,
        consented_at: datetime | None = None,
    ) -> None:
        self.session_consents = SessionConsents(
            face_processing_consent=face_processing_consent,
            adult_subject_confirmed=adult_subject_confirmed,
            consented_at=consented_at or utcnow(),
        )

    def to_response(self, include_token: bool = False) -> SessionResponse:
        return SessionResponse(
            session_id=self.session_id,
            # Only include the secret token in the creation response.
            owner_token=self.owner_token if include_token else "",
            style=self.style,
            gender=self.gender,
            status=self.status,
            uploaded_photos=[p.name for p in self.uploaded_photos],
            photo_quality=self.photo_quality,
            reference_quality=getattr(self, "reference_quality", None),
            session_consents=self.session_consents,
            feedback_summary=self._feedback_summary(),
            pipeline_metrics=self._pipeline_metrics(),
            generated_images=self.generated_images,
            revisions_used=self.revisions_used,
            max_revisions=self.max_revisions,
            created_at=self.created_at,
            tier=self.tier,
            hero_preview_image_id=self.hero_preview_image_id,
            unlocked=self.unlocked,
        )

    def _feedback_summary(self) -> dict:
        images = [
            img for img in self.generated_images
            if img.parent_image_id is None and not img.operation
        ]
        total = len(images)
        image_ids = {img.image_id for img in images}
        feedback = [
            item for item in self.user_feedback
            if item.get("image_id") in image_ids
        ]
        downloaded = {f["image_id"] for f in feedback if f.get("event") == "downloaded"}
        selected = {f["image_id"] for f in feedback if f.get("event") == "selected"}
        liked = {f["image_id"] for f in feedback if f.get("event") == "looks_like_me"}
        not_like_me = [
            f for f in feedback if f.get("event") == "not_like_me"
        ]
        identity_feedback = [
            f for f in feedback
            if f.get("event") in {"looks_like_me", "not_like_me"}
        ]
        ai_deliverable = 0
        deliverable_image_ids: set[str] = set()
        for img in images:
            if self._image_passed_delivery_gate(img):
                ai_deliverable += 1
                deliverable_image_ids.add(img.image_id)
        qualified_saved = downloaded & deliverable_image_ids
        qualified_selected = selected & deliverable_image_ids
        return {
            "total_generated": total,
            "ai_deliverable_count": ai_deliverable,
            "ai_deliverable_rate": round(ai_deliverable / total, 4) if total else 0,
            "downloaded_count": len(downloaded),
            "selected_count": len(selected),
            "qualified_saved_count": len(qualified_saved),
            "qualified_downloaded_count": len(qualified_saved),
            "qualified_selected_count": len(qualified_selected),
            "liked_identity_count": len(liked),
            "not_like_me_count": len(not_like_me),
            "user_saved_rate": round(len(downloaded) / total, 4) if total else 0,
            "user_selected_rate": round(len(selected) / total, 4) if total else 0,
            "qualified_saved_rate": (
                round(len(qualified_saved) / total, 4) if total else 0
            ),
            "qualified_downloaded_rate": (
                round(len(qualified_saved) / total, 4) if total else 0
            ),
            "qualified_selected_rate": (
                round(len(qualified_selected) / total, 4) if total else 0
            ),
            "not_like_me_rate": (
                round(len(not_like_me) / len(identity_feedback), 4)
                if identity_feedback else 0
            ),
        }

    @staticmethod
    def _image_shot_id(img: GeneratedImage) -> str:
        meta = img.resemblance or {}
        shot_spec = meta.get("shot_spec")
        if isinstance(shot_spec, dict):
            shot_id = shot_spec.get("shot_id")
            if shot_id:
                return str(shot_id)
        prompt_id = img.prompt_id or "unknown"
        known_shots = (
            "street_medium",
            "half_body",
            "full_body",
            "environmental",
            "closeup",
            "standard",
        )
        for shot in known_shots:
            if prompt_id == shot or prompt_id.endswith(f"_{shot}"):
                return shot
        if "_" in prompt_id:
            tail = prompt_id.rsplit("_", 1)[-1]
            if tail:
                return tail
        return prompt_id

    @staticmethod
    def _image_passed_delivery_gate(img: GeneratedImage) -> bool:
        meta = img.resemblance if isinstance(img.resemblance, dict) else {}
        selected_candidate = (
            meta.get("selected_candidate") if isinstance(meta, dict) else None
        )
        if not isinstance(selected_candidate, dict):
            return False
        gate_status = selected_candidate.get("gate_status")
        return bool(
            selected_candidate.get("deliverable")
            and isinstance(gate_status, dict)
            and gate_status.get("hard_gates_pass")
        )

    @staticmethod
    def _empty_shot_metric() -> dict[str, Any]:
        return {
            "attempts": 0,
            "completed": 0,
            "failed": 0,
            "deliverable_count": 0,
            "failure_reasons": {},
            "provider_invocations": 0,
            "estimated_cost": 0.0,
            "candidates_generated": 0,
            "identity_first_pass_candidates": 0,
            "identity_first_passes": 0,
            "identity_repairs": 0,
            "local_edits": 0,
            "regenerations": 0,
            "downloaded_count": 0,
            "selected_count": 0,
            "liked_identity_count": 0,
            "not_like_me_count": 0,
            "identity_feedback_count": 0,
        }

    @staticmethod
    def _percentile(values: list[float] | list[int], percentile: float) -> float:
        if not values:
            return 0
        ordered = sorted(float(value) for value in values)
        index = max(
            0,
            min(
                len(ordered) - 1,
                int((len(ordered) * percentile + 99) // 100) - 1,
            ),
        )
        return round(ordered[index], 2)

    def _pipeline_metrics(self) -> dict:
        images = [
            img for img in self.generated_images
            if img.parent_image_id is None and not img.operation
        ]
        session_metrics = self.pipeline_metrics if isinstance(self.pipeline_metrics, dict) else {}
        photo_quality_records = list((self.photo_quality or {}).values())
        input_photo_count = len(photo_quality_records)
        input_photo_passed = sum(
            1 for rec in photo_quality_records if rec.get("pass")
        )
        input_photo_failed = input_photo_count - input_photo_passed
        reference_quality = self.reference_quality or {}
        reference_issues = reference_quality.get("issues") or []
        total_images = len(images)
        total_invocations = 0
        create_invocations = 0
        identity_repairs = 0
        local_edits = 0
        regenerations = 0
        identity_repair_successes = 0
        local_edit_successes = 0
        regeneration_successes = 0
        estimated_cost = 0.0
        deliverable_count = 0
        latency_values: list[int] = []
        delivery_latency_seconds: list[float] = []
        candidates_generated = 0
        identity_first_pass_candidates = 0
        identity_first_passes = 0
        operation_counts: dict[str, int] = {}
        provider_invocation_raw = {
            "by_operation": {},
            "by_provider_model": {},
            "by_prompt_version": {},
        }
        downloaded_image_ids = {
            str(item.get("image_id"))
            for item in self.user_feedback
            if item.get("image_id") and item.get("event") == "downloaded"
        }
        selected_image_ids = {
            str(item.get("image_id"))
            for item in self.user_feedback
            if item.get("image_id") and item.get("event") == "selected"
        }
        qualified_saved_count = 0
        qualified_selected_count = 0
        payment_status = self.payment_status.value if self.payment_status else None
        paid_payment_count = 1 if self.payment_status in {
            PaymentStatus.paid,
            PaymentStatus.refunded,
        } else 0
        refunded_payment_count = (
            1 if self.payment_status == PaymentStatus.refunded else 0
        )

        def _empty_invocation_metric() -> dict:
            return {
                "invocations": 0,
                "successes": 0,
                "failures": 0,
                "estimated_cost": 0.0,
                "latency_values": [],
            }

        def _add_invocation_metric(group: str, key: str, inv: dict) -> None:
            bucket = provider_invocation_raw[group].setdefault(
                key, _empty_invocation_metric()
            )
            bucket["invocations"] += 1
            status = str(inv.get("result_status") or "success").lower()
            if status == "success":
                bucket["successes"] += 1
            else:
                bucket["failures"] += 1
            try:
                cost_value = inv.get("cost")
                if cost_value is None:
                    cost_value = inv.get("estimated_cost")
                bucket["estimated_cost"] += float(cost_value or 0.0)
            except Exception:
                pass
            latency = inv.get("latency_ms")
            if isinstance(latency, int):
                bucket["latency_values"].append(latency)

        def _record_invocation(inv: dict) -> None:
            operation = str(inv.get("operation") or "UNKNOWN")
            provider = str(inv.get("provider") or "unknown")
            model = str(inv.get("model") or "unknown")
            prompt_version = str(inv.get("prompt_version") or "unknown")
            _add_invocation_metric("by_operation", operation, inv)
            _add_invocation_metric("by_provider_model", f"{provider}:{model}", inv)
            _add_invocation_metric("by_prompt_version", prompt_version, inv)

        for img in images:
            meta = img.resemblance or {}
            selected_candidate = meta.get("selected_candidate") or {}
            deliverable = self._image_passed_delivery_gate(img)
            if deliverable:
                deliverable_count += 1
                if img.image_id in downloaded_image_ids:
                    qualified_saved_count += 1
                if img.image_id in selected_image_ids:
                    qualified_selected_count += 1
                latency = (img.created_at - self.created_at).total_seconds()
                if latency >= 0:
                    delivery_latency_seconds.append(latency)
            candidates = meta.get("candidates") or []
            candidates_generated += len(candidates)
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("regenerated_from_candidate_id"):
                    continue
                identity_first_pass_candidates += 1
                gate = candidate.get("gate_status") or {}
                if isinstance(gate, dict) and gate.get("identity_pass"):
                    identity_first_passes += 1

            budget = meta.get("budget") or {}
            image_identity_repairs = int(budget.get("identity_repairs_used") or 0)
            image_local_edits = int(budget.get("local_edits_used") or 0)
            image_regenerations = int(budget.get("regenerations_used") or 0)
            identity_repairs += image_identity_repairs
            local_edits += image_local_edits
            regenerations += image_regenerations

            if deliverable and image_identity_repairs:
                repair_meta = meta.get("face_swap") or {}
                if isinstance(repair_meta, dict) and repair_meta.get("applied"):
                    identity_repair_successes += image_identity_repairs
            if deliverable and image_local_edits:
                local_edit_meta = meta.get("local_edit") or {}
                if isinstance(local_edit_meta, dict) and local_edit_meta.get("applied"):
                    local_edit_successes += image_local_edits
            if deliverable and image_regenerations:
                selected_candidate_id = selected_candidate.get("candidate_id")
                selected_from_regen = any(
                    isinstance(candidate, dict)
                    and candidate.get("candidate_id") == selected_candidate_id
                    and candidate.get("regenerated_from_candidate_id")
                    for candidate in (meta.get("candidates") or [])
                )
                if selected_from_regen:
                    regeneration_successes += 1

            for inv in meta.get("provider_invocations") or []:
                if not isinstance(inv, dict):
                    continue
                total_invocations += 1
                operation = str(inv.get("operation") or "UNKNOWN")
                operation_counts[operation] = operation_counts.get(operation, 0) + 1
                _record_invocation(inv)
                if operation == "CREATE_FROM_REFERENCES":
                    create_invocations += 1
                try:
                    cost_value = inv.get("cost")
                    if cost_value is None:
                        cost_value = inv.get("estimated_cost")
                    estimated_cost += float(cost_value or 0.0)
                except Exception:
                    pass
                latency = inv.get("latency_ms")
                if isinstance(latency, int):
                    latency_values.append(latency)

        def _int_metric(key: str, default: int = 0) -> int:
            try:
                return int(session_metrics.get(key, default) or 0)
            except Exception:
                return default

        def _float_metric(key: str, default: float = 0.0) -> float:
            try:
                return float(session_metrics.get(key, default) or 0.0)
            except Exception:
                return default

        generation_attempts = max(_int_metric("generation_attempts"), total_images)
        generation_failures = max(
            _int_metric("generation_failures"),
            max(0, generation_attempts - total_images),
        )
        failed_reasons = session_metrics.get("failed_generation_reasons") or {}
        if not isinstance(failed_reasons, dict):
            failed_reasons = {}

        failed_operation_counts = session_metrics.get("failed_operation_counts") or {}
        if not isinstance(failed_operation_counts, dict):
            failed_operation_counts = {}
        for operation, count in failed_operation_counts.items():
            try:
                failed_count = int(count)
                operation_counts[operation] = operation_counts.get(operation, 0) + failed_count
                bucket = provider_invocation_raw["by_operation"].setdefault(
                    str(operation), _empty_invocation_metric()
                )
                bucket["invocations"] += failed_count
                bucket["failures"] += failed_count
            except Exception:
                continue
        total_invocations += _int_metric("failed_provider_invocations")
        create_invocations += _int_metric("failed_create_from_reference_invocations")
        identity_repairs += _int_metric("failed_identity_repairs")
        local_edits += _int_metric("failed_local_edits")
        regenerations += _int_metric("failed_regenerations")
        identity_first_pass_candidates += _int_metric(
            "failed_initial_identity_candidates"
        )
        identity_first_passes += _int_metric("failed_initial_identity_passes")
        estimated_cost += _float_metric("failed_estimated_cost")
        candidates_generated += _int_metric("failed_candidates_generated")
        failed_latency_values = session_metrics.get("failed_latency_values") or []
        if isinstance(failed_latency_values, list):
            latency_values.extend(
                item for item in failed_latency_values if isinstance(item, int)
            )
        feedback_by_image: dict[str, list[dict]] = {}
        for item in self.user_feedback:
            image_id = item.get("image_id")
            if image_id:
                feedback_by_image.setdefault(str(image_id), []).append(item)

        identity_threshold_raw: dict[str, dict] = {}
        for img in images:
            meta = img.resemblance or {}
            strategy = meta.get("strategy") or {}
            threshold_profile = strategy.get("identity_threshold_profile") or {}
            selected_candidate = meta.get("selected_candidate") or {}
            gate_status = selected_candidate.get("gate_status") or {}
            profile = (
                threshold_profile.get("profile")
                or gate_status.get("identity_threshold_profile")
                or "unknown"
            )
            item = identity_threshold_raw.setdefault(str(profile), {
                "delivered_count": 0,
                "identity_score_sum": 0.0,
                "identity_score_count": 0,
                "pass_threshold": (
                    threshold_profile.get("identity_pass_threshold")
                    or gate_status.get("identity_pass_threshold")
                ),
                "repair_threshold": (
                    threshold_profile.get("identity_repair_threshold")
                    or gate_status.get("identity_repair_threshold")
                ),
                "user_identity_score_sum": 0.0,
                "user_identity_score_count": 0,
                "liked_identity_count": 0,
                "not_like_me_count": 0,
                "identity_feedback_count": 0,
            })
            item["delivered_count"] += 1
            identity_score = selected_candidate.get("identity_score")
            if isinstance(identity_score, (int, float)):
                item["identity_score_sum"] += float(identity_score)
                item["identity_score_count"] += 1
            if item["pass_threshold"] is None:
                item["pass_threshold"] = gate_status.get("identity_pass_threshold")
            if item["repair_threshold"] is None:
                item["repair_threshold"] = gate_status.get("identity_repair_threshold")

            for feedback_item in feedback_by_image.get(img.image_id, []):
                event = feedback_item.get("event")
                if event == "looks_like_me":
                    item["liked_identity_count"] += 1
                    item["identity_feedback_count"] += 1
                elif event == "not_like_me":
                    item["not_like_me_count"] += 1
                    item["identity_feedback_count"] += 1
                else:
                    continue
                score = feedback_item.get("score")
                if isinstance(score, (int, float)):
                    item["user_identity_score_sum"] += float(score)
                    item["user_identity_score_count"] += 1

        raw_shot_metrics = session_metrics.get("shot_metrics") or {}
        shot_metrics_raw: dict[str, dict] = {}
        if isinstance(raw_shot_metrics, dict):
            for shot_id, raw_item in raw_shot_metrics.items():
                if not isinstance(raw_item, dict):
                    continue
                item = self._empty_shot_metric()
                try:
                    item["attempts"] = int(raw_item.get("attempts") or 0)
                    item["completed"] = int(raw_item.get("completed") or 0)
                    item["failed"] = int(raw_item.get("failed") or 0)
                    item["deliverable_count"] = int(raw_item.get("deliverable_count") or 0)
                    item["provider_invocations"] = int(raw_item.get("provider_invocations") or 0)
                    item["candidates_generated"] = int(raw_item.get("candidates_generated") or 0)
                    item["identity_first_pass_candidates"] = int(
                        raw_item.get("identity_first_pass_candidates") or 0
                    )
                    item["identity_first_passes"] = int(
                        raw_item.get("identity_first_passes") or 0
                    )
                    item["identity_repairs"] = int(raw_item.get("identity_repairs") or 0)
                    item["local_edits"] = int(raw_item.get("local_edits") or 0)
                    item["regenerations"] = int(raw_item.get("regenerations") or 0)
                    item["estimated_cost"] = float(raw_item.get("estimated_cost") or 0.0)
                except Exception:
                    continue
                reasons = raw_item.get("failure_reasons") or {}
                item["failure_reasons"] = reasons if isinstance(reasons, dict) else {}
                shot_metrics_raw[str(shot_id)] = item

        for img in images:
            shot_id = self._image_shot_id(img)
            item = shot_metrics_raw.setdefault(shot_id, self._empty_shot_metric())
            if item["attempts"] == 0 and item["completed"] == 0:
                item["attempts"] += 1
                item["completed"] += 1
                meta = img.resemblance or {}
                if self._image_passed_delivery_gate(img):
                    item["deliverable_count"] += 1
                for candidate in meta.get("candidates") or []:
                    if not isinstance(candidate, dict):
                        continue
                    if candidate.get("regenerated_from_candidate_id"):
                        continue
                    item["identity_first_pass_candidates"] += 1
                    gate = candidate.get("gate_status") or {}
                    if isinstance(gate, dict) and gate.get("identity_pass"):
                        item["identity_first_passes"] += 1

            feedback_items = feedback_by_image.get(img.image_id, [])
            item["downloaded_count"] += len({
                f.get("image_id") for f in feedback_items
                if f.get("event") == "downloaded"
            })
            item["selected_count"] += len({
                f.get("image_id") for f in feedback_items
                if f.get("event") == "selected"
            })
            item["liked_identity_count"] += len({
                f.get("image_id") for f in feedback_items
                if f.get("event") == "looks_like_me"
            })
            item["not_like_me_count"] += len({
                f.get("image_id") for f in feedback_items
                if f.get("event") == "not_like_me"
            })
            item["identity_feedback_count"] += len({
                f.get("image_id") for f in feedback_items
                if f.get("event") in {"looks_like_me", "not_like_me"}
            })

        shot_metrics: dict[str, dict] = {}
        for shot_id, item in shot_metrics_raw.items():
            attempts = int(item.get("attempts") or 0)
            completed = int(item.get("completed") or 0)
            failed = int(item.get("failed") or 0)
            shot_deliverable = int(item.get("deliverable_count") or 0)
            shot_provider_calls = int(item.get("provider_invocations") or 0)
            shot_candidates = int(item.get("candidates_generated") or 0)
            shot_identity_candidates = int(
                item.get("identity_first_pass_candidates") or 0
            )
            shot_identity_passes = int(item.get("identity_first_passes") or 0)
            shot_cost = float(item.get("estimated_cost") or 0.0)
            downloaded_count = int(item.get("downloaded_count") or 0)
            selected_count = int(item.get("selected_count") or 0)
            liked_identity_count = int(item.get("liked_identity_count") or 0)
            not_like_me_count = int(item.get("not_like_me_count") or 0)
            identity_feedback_count = int(item.get("identity_feedback_count") or 0)
            reasons = item.get("failure_reasons") or {}
            shot_metrics[str(shot_id)] = {
                "attempts": attempts,
                "completed": completed,
                "failed": failed,
                "deliverable_count": shot_deliverable,
                "failure_reasons": reasons if isinstance(reasons, dict) else {},
                "first_pass_rate": (
                    round(completed / attempts, 4) if attempts else 0
                ),
                "failure_rate": round(failed / attempts, 4) if attempts else 0,
                "deliverable_rate": (
                    round(shot_deliverable / attempts, 4) if attempts else 0
                ),
                "provider_invocations": shot_provider_calls,
                "estimated_cost": round(shot_cost, 4),
                "estimated_cost_per_deliverable": (
                    round(shot_cost / shot_deliverable, 4)
                    if shot_deliverable else 0
                ),
                "avg_api_calls_per_attempt": (
                    round(shot_provider_calls / attempts, 4) if attempts else 0
                ),
                "candidates_generated": shot_candidates,
                "avg_candidates_per_attempt": (
                    round(shot_candidates / attempts, 4) if attempts else 0
                ),
                "identity_first_pass_candidates": shot_identity_candidates,
                "identity_first_passes": shot_identity_passes,
                "identity_first_pass_rate": (
                    round(shot_identity_passes / shot_identity_candidates, 4)
                    if shot_identity_candidates else 0
                ),
                "identity_repairs": int(item.get("identity_repairs") or 0),
                "local_edits": int(item.get("local_edits") or 0),
                "regenerations": int(item.get("regenerations") or 0),
                "downloaded_count": downloaded_count,
                "selected_count": selected_count,
                "liked_identity_count": liked_identity_count,
                "not_like_me_count": not_like_me_count,
                "user_saved_rate": (
                    round(downloaded_count / shot_deliverable, 4)
                    if shot_deliverable else 0
                ),
                "user_selected_rate": (
                    round(selected_count / shot_deliverable, 4)
                    if shot_deliverable else 0
                ),
                "not_like_me_rate": (
                    round(not_like_me_count / identity_feedback_count, 4)
                    if identity_feedback_count else 0
                ),
            }

        identity_threshold_metrics: dict[str, dict] = {}
        for profile, item in identity_threshold_raw.items():
            identity_score_count = int(item.get("identity_score_count") or 0)
            user_identity_score_count = int(item.get("user_identity_score_count") or 0)
            identity_feedback_count = int(item.get("identity_feedback_count") or 0)
            delivered_count = int(item.get("delivered_count") or 0)
            liked_identity_count = int(item.get("liked_identity_count") or 0)
            not_like_me_count = int(item.get("not_like_me_count") or 0)
            identity_threshold_metrics[profile] = {
                "delivered_count": delivered_count,
                "identity_pass_threshold": item.get("pass_threshold"),
                "identity_repair_threshold": item.get("repair_threshold"),
                "avg_ai_identity_score": (
                    round(item["identity_score_sum"] / identity_score_count, 4)
                    if identity_score_count else None
                ),
                "identity_feedback_count": identity_feedback_count,
                "liked_identity_count": liked_identity_count,
                "not_like_me_count": not_like_me_count,
                "not_like_me_rate": (
                    round(not_like_me_count / identity_feedback_count, 4)
                    if identity_feedback_count else 0
                ),
                "avg_user_identity_score": (
                    round(item["user_identity_score_sum"] / user_identity_score_count, 4)
                    if user_identity_score_count else None
                ),
            }

        agent_action_metrics = {
            "IDENTITY_REPAIR": {
                "attempts": identity_repairs,
                "successes": identity_repair_successes,
                "success_rate": (
                    round(identity_repair_successes / identity_repairs, 4)
                    if identity_repairs else 0
                ),
            },
            "LOCAL_EDIT": {
                "attempts": local_edits,
                "successes": local_edit_successes,
                "success_rate": (
                    round(local_edit_successes / local_edits, 4)
                    if local_edits else 0
                ),
            },
            "REGENERATE_FROM_ORIGINAL": {
                "attempts": regenerations,
                "successes": regeneration_successes,
                "success_rate": (
                    round(regeneration_successes / regenerations, 4)
                    if regenerations else 0
                ),
            },
        }

        def _finalize_invocation_metrics(items: dict[str, dict]) -> dict:
            out: dict[str, dict] = {}
            for key, item in items.items():
                invocations = int(item.get("invocations") or 0)
                successes = int(item.get("successes") or 0)
                failures = int(item.get("failures") or 0)
                item_latency_values = item.get("latency_values") or []
                estimated_group_cost = float(item.get("estimated_cost") or 0.0)
                out[str(key)] = {
                    "invocations": invocations,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": (
                        round(successes / invocations, 4) if invocations else 0
                    ),
                    "estimated_cost": round(estimated_group_cost, 4),
                    "avg_cost_per_invocation": (
                        round(estimated_group_cost / invocations, 4)
                        if invocations else 0
                    ),
                    "avg_latency_ms": (
                        round(sum(item_latency_values) / len(item_latency_values), 2)
                        if item_latency_values else 0
                    ),
                    "p50_latency_ms": self._percentile(item_latency_values, 50),
                    "p95_latency_ms": self._percentile(item_latency_values, 95),
                }
            return out

        provider_invocation_metrics = {
            "by_operation": _finalize_invocation_metrics(
                provider_invocation_raw["by_operation"]
            ),
            "by_provider_model": _finalize_invocation_metrics(
                provider_invocation_raw["by_provider_model"]
            ),
            "by_prompt_version": _finalize_invocation_metrics(
                provider_invocation_raw["by_prompt_version"]
            ),
        }

        return {
            "input_photo_count": input_photo_count,
            "input_photo_passed": input_photo_passed,
            "input_photo_failed": input_photo_failed,
            "input_photo_pass_rate": (
                round(input_photo_passed / input_photo_count, 4)
                if input_photo_count else 0
            ),
            "reference_quality_pass": bool(reference_quality.get("pass", False)),
            "reference_quality_issue_count": (
                len(reference_issues) if isinstance(reference_issues, list) else 0
            ),
            "total_images": total_images,
            "deliverable_count": deliverable_count,
            "generation_attempts": generation_attempts,
            "generation_failures": generation_failures,
            "failed_generation_reasons": failed_reasons,
            "shot_metrics": shot_metrics,
            "identity_threshold_metrics": identity_threshold_metrics,
            "delivered_image_deliverable_rate": (
                round(deliverable_count / total_images, 4)
                if total_images else 0
            ),
            "deliverable_rate": (
                round(deliverable_count / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "generation_failure_rate": (
                round(generation_failures / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "qualified_saved_count": qualified_saved_count,
            "qualified_downloaded_count": qualified_saved_count,
            "qualified_selected_count": qualified_selected_count,
            "north_star_qualified_save_rate": (
                round(qualified_saved_count / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "qualified_saved_rate": (
                round(qualified_saved_count / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "qualified_downloaded_rate": (
                round(qualified_saved_count / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "qualified_selected_rate": (
                round(qualified_selected_count / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "payment_status": payment_status,
            "paid_payment_count": paid_payment_count,
            "refunded_payment_count": refunded_payment_count,
            "refund_rate": (
                round(refunded_payment_count / paid_payment_count, 4)
                if paid_payment_count else 0
            ),
            "total_provider_invocations": total_invocations,
            "create_from_reference_invocations": create_invocations,
            "operation_counts": operation_counts,
            "provider_invocation_metrics": provider_invocation_metrics,
            "estimated_total_cost": round(estimated_cost, 4),
            "estimated_cost_per_image": (
                round(estimated_cost / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "estimated_cost_per_deliverable": (
                round(estimated_cost / deliverable_count, 4)
                if deliverable_count else 0
            ),
            "avg_api_calls_per_image": (
                round(total_invocations / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "candidates_generated": candidates_generated,
            "avg_candidates_per_image": (
                round(candidates_generated / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "identity_first_pass_candidates": identity_first_pass_candidates,
            "identity_first_passes": identity_first_passes,
            "identity_first_pass_rate": (
                round(identity_first_passes / identity_first_pass_candidates, 4)
                if identity_first_pass_candidates else 0
            ),
            "identity_repairs": identity_repairs,
            "local_edits": local_edits,
            "regenerations": regenerations,
            "agent_action_metrics": agent_action_metrics,
            "identity_repair_successes": identity_repair_successes,
            "local_edit_successes": local_edit_successes,
            "regeneration_successes": regeneration_successes,
            "identity_repair_success_rate": (
                agent_action_metrics["IDENTITY_REPAIR"]["success_rate"]
            ),
            "local_edit_success_rate": (
                agent_action_metrics["LOCAL_EDIT"]["success_rate"]
            ),
            "regeneration_success_rate": (
                agent_action_metrics["REGENERATE_FROM_ORIGINAL"]["success_rate"]
            ),
            "identity_repair_rate": (
                round(identity_repairs / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "regeneration_rate": (
                round(regenerations / generation_attempts, 4)
                if generation_attempts else 0
            ),
            "avg_provider_latency_ms": (
                round(sum(latency_values) / len(latency_values), 2)
                if latency_values else 0
            ),
            "p50_provider_latency_ms": self._percentile(latency_values, 50),
            "p95_provider_latency_ms": self._percentile(latency_values, 95),
            "p50_delivery_latency_seconds": self._percentile(
                delivery_latency_seconds, 50
            ),
            "p95_delivery_latency_seconds": self._percentile(
                delivery_latency_seconds, 95
            ),
        }


class Job:
    """Internal job representation."""

    def __init__(
        self,
        session_id: str,
        job_type: JobType,
        prompt: str,
        prompt_id: str | None = None,
        instruction: str | None = None,
        revised_image_id: str | None = None,
        template_path: str | None = None,
        shot_spec: dict | None = None,
    ):
        self.job_id = f"j_{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        self.job_type = job_type
        self.prompt = prompt
        self.prompt_id = prompt_id
        self.instruction = instruction
        self.revised_image_id = revised_image_id
        self.template_path = template_path
        self.shot_spec = shot_spec
        self.status = JobStatus.queued
        self.progress = 0.0
        self.result_image: GeneratedImage | None = None
        self.error: str | None = None
        self.turn = 1
        self.created_at = utcnow()

    def to_response(self, position: int = 0) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            session_id=self.session_id,
            job_type=self.job_type,
            status=self.status,
            prompt_id=self.prompt_id,
            shot_spec=self.shot_spec,
            progress=self.progress,
            result_image=self.result_image,
            error=self.error,
            position_in_queue=position,
        )
