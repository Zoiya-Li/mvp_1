export type StyleKey =
  | "id_photo"
  | "business"
  | "academic"
  | "social"
  | "jk_portrait"
  | "chinese_style"
  | "fashion"
  | "cinematic"
  | "creative";
export type Gender = "male" | "female";
export type SessionStatus = "created" | "uploading" | "ready" | "generating" | "hero_preview_ready" | "reviewing" | "done" | "failed";
export type JobStatus = "queued" | "processing" | "completed" | "failed";
export type JobType = "generate" | "revise" | "hero_preview" | "full_set";
export type FeedbackEvent =
  | "downloaded"
  | "selected"
  | "looks_like_me"
  | "not_like_me"
  | "bad_artifacts"
  | "not_saved";

export interface GeneratedImage {
  image_id: string;
  url: string;
  prompt_id: string;
  turn: number;
  revised_image_id: string | null;
  created_at: string;
  parent_image_id?: string | null;
  operation?: string | null;
  resemblance?: GenerationMetadata | null;
}

export interface QualityScores {
  identity?: number | null;
  face_quality?: number | null;
  style_match?: number | null;
  artifact?: number | null;
  commercial_readiness?: number | null;
}

export interface IdentityQuality {
  score?: number | null;
  cosine_similarity?: number | null;
  reference_consistency?: number | null;
  hard_failures?: string[];
  measurements?: Record<string, unknown>;
  notes?: string;
}

export interface LocalQuality {
  scores?: Pick<QualityScores, "face_quality" | "artifact" | "commercial_readiness">;
  hard_failures?: string[];
  measurements?: Record<string, unknown>;
  notes?: string;
}

export interface CandidateJudgement {
  scores?: QualityScores;
  hard_failures?: string[];
  recommended_action?: string;
  notes?: string;
  local_quality?: LocalQuality;
  identity_quality?: IdentityQuality;
}

export interface CandidateGateStatus {
  safety_pass?: boolean;
  face_detected?: boolean;
  identity_pass?: boolean;
  quality_pass?: boolean;
  severe_quality_fail?: boolean;
  hard_gates_pass?: boolean;
  hard_gate_failures?: string[];
}

export interface AgentAction {
  action?: string;
  reason?: string;
  candidate_id?: string;
  candidate_index?: number;
  state?: string;
  executed?: boolean;
  selected_for_execution?: boolean;
}

export interface ProviderInvocation {
  invocation_id?: string;
  provider?: string;
  model?: string | null;
  operation?: string;
  prompt_version?: string | null;
  reference_ids?: string[];
  candidate_index?: number;
  parent_candidate_id?: string | null;
  latency_ms?: number | null;
  estimated_cost?: number | null;
  result_status?: string;
}

export interface ShotSpec {
  style_id?: string;
  style_label?: string;
  template_id?: string;
  template_label?: string;
  shot_id?: string;
  shot_label?: string;
  sequence?: number;
  framing?: string;
  pose?: string;
  lighting?: string;
  lens?: string;
  prompt_blocks?: Record<string, unknown>;
}

export interface GenerationCandidate {
  index: number;
  candidate_id?: string;
  filename?: string;
  judgement?: CandidateJudgement;
  aggregate_score?: number;
  gate_status?: CandidateGateStatus;
  agent_action?: AgentAction;
  provider_invocation_id?: string;
  selected?: boolean;
  repair?: Record<string, unknown> | null;
}

export interface CandidateShortlistItem {
  rank?: number;
  candidate_id?: string;
  candidate_index?: number;
  filename?: string;
  aggregate_score?: number;
  hard_gates_pass?: boolean;
  hard_gate_failures?: string[];
  recommended_action?: string;
  action_reason?: string;
  selected?: boolean;
}

export interface GenerationMetadata {
  pipeline?: string;
  iterations?: number;
  final_score?: number | null;
  history?: Array<Record<string, unknown>>;
  identity_pack?: Record<string, unknown>;
  shot_spec?: ShotSpec;
  allowed_actions?: string[];
  budget?: Record<string, unknown>;
  strategy?: Record<string, unknown>;
  candidates?: GenerationCandidate[];
  shortlist?: CandidateShortlistItem[];
  agent_actions?: AgentAction[];
  provider_invocations?: ProviderInvocation[];
  selected_candidate?: {
    index?: number;
    candidate_id?: string;
    filename?: string;
    aggregate_score?: number;
    identity_score?: number | null;
    gate_status?: CandidateGateStatus;
    deliverable?: boolean;
  } | null;
  face_swap?: {
    action?: string;
    applied?: boolean;
    message?: string;
    output_filename?: string;
    source_face_count?: number;
    target_face_count?: number;
  } | null;
}

export interface SessionConsents {
  face_processing_consent: boolean;
  adult_subject_confirmed: boolean;
  no_training_by_default: boolean;
  cross_user_search_prohibited: boolean;
  long_term_face_library_prohibited: boolean;
  consented_at: string | null;
  policy_version: string;
}

export interface SessionResponse {
  session_id: string;
  /**
   * Secret owner token. Returned ONLY at session creation; the server echoes an
   * empty string on subsequent GETs. The client must persist it and send it back
   * as the X-Session-Token header (and ?token= for <img> URLs).
   */
  owner_token: string;
  style: StyleKey;
  gender: Gender;
  status: SessionStatus;
  uploaded_photos: string[];
  photo_quality: Record<string, Record<string, unknown>>;
  reference_quality: Record<string, unknown> | null;
  session_consents: SessionConsents;
  feedback_summary: Record<string, unknown>;
  pipeline_metrics: Record<string, unknown>;
  generated_images: GeneratedImage[];
  revisions_used: number;
  max_revisions: number;
  created_at: string;
  tier: PricingTier;
  hero_preview_image_id: string | null;
  unlocked: boolean;
}

// ── Post-processing ──────────────────────────────────

// ID photo size specs. NOTE: the literal string values ("1寸"/"2寸") are the
// machine contract values sent to the Python backend API, so they must stay
// as-is; only this comment is translated.
export type IDPhotoSpec = "1寸" | "2寸";
export type BgColor = "red" | "blue" | "white" | "gradient_gray";

export interface PostProcessCropRequest {
  image_id: string;
  spec: IDPhotoSpec;
}

export interface PostProcessBgRequest {
  image_id: string;
  color: BgColor;
}

export interface PostProcessCombinedRequest {
  image_id: string;
  spec: IDPhotoSpec;
  color: BgColor;
}

export interface PostProcessResponse {
  original_image_id: string;
  processed_image_id: string;
  url: string;
  operation: string;
}

export interface JobResponse {
  job_id: string;
  session_id: string;
  job_type: JobType;
  status: JobStatus;
  prompt_id: string | null;
  shot_spec: ShotSpec | null;
  progress: number;
  result_image: GeneratedImage | null;
  error: string | null;
  position_in_queue: number;
}

export interface TemplateInfo {
  id: string;
  gender: string;
  label: string;
  template_image: string | null;
}

export interface StyleInfo {
  key: StyleKey;
  label: string;
  label_en: string;
  use_cases: string[];
  templates: TemplateInfo[];
}

export interface StyleListResponse {
  styles: StyleInfo[];
}

export interface WSMessage {
  type: string;
  [key: string]: unknown;
}

/**
 * Live progress for the controlled production pipeline, broadcast as a WS
 * message with type "generation_progress" while a generate job runs. The
 * backend generates a small candidate set, scores each candidate, applies a
 * deterministic repair step when useful, then selects the best image.
 *
 * `phase` follows gemini_worker.py:
 *   candidate_generating → one candidate image being produced
 *   candidate_judging    → structured QA scoring for that candidate
 *   repairing            → deterministic post-processing / face-swap repair
 *   accepted             → final candidate selected
 *
 * Legacy phases are kept for compatibility with old in-flight jobs.
 *
 * `detail` is the human-facing status string emitted by gemini_worker.py. The
 * frontend renders it verbatim, so localization belongs at the source.
 */
export type GenerationPhase =
  | "candidate_generating"
  | "candidate_judging"
  | "repairing"
  | "generating"
  | "judging"
  | "revising"
  | "accepted"
  | "max_reached";

export interface GenerationProgress {
  iteration: number;
  max_iterations: number;
  phase: GenerationPhase;
  detail: string;
  shot_spec?: ShotSpec | null;
}

// ── Payment / Pricing ──────────────────────────────────

export type PricingTier = "free" | "standard" | "premium";
export type PaymentStatus = "pending" | "paid" | "expired" | "refunded";

export interface TierInfo {
  tier: PricingTier;
  label: string;
  price_cents: number;
  max_styles: number;
  max_revisions: number;
  allow_id_photo: boolean;
  allow_bg_replace: boolean;
  allow_hd_download: boolean;
}

export interface PricingResponse {
  tiers: TierInfo[];
}

export interface PaymentResponse {
  payment_id: string;
  session_id: string;
  tier: PricingTier;
  status: PaymentStatus;
  /** Paddle hosted checkout URL the browser is redirected to. null in mock
   *  mode (dev auto-confirm) or when the record is re-read after creation. */
  checkout_url: string | null;
  amount_cents: number;
  created_at: string;
}

export interface PaymentStatusResponse {
  payment_id: string;
  status: PaymentStatus;
  tier: PricingTier;
}

export interface UserFeedbackRequest {
  event: FeedbackEvent;
  reason?: string | null;
  score?: number | null;
}

export interface UserFeedbackResponse {
  feedback_id: string;
  session_id: string;
  image_id: string;
  event: FeedbackEvent;
  reason?: string | null;
  score?: number | null;
  created_at: string;
}
