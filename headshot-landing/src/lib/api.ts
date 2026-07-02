import type {
  SessionResponse,
  JobResponse,
  StyleListResponse,
  StyleKey,
  Gender,
  PostProcessCropRequest,
  PostProcessBgRequest,
  PostProcessCombinedRequest,
  PostProcessResponse,
  PricingTier,
  PaymentResponse,
  PaymentStatusResponse,
  PricingResponse,
  UserFeedbackRequest,
  UserFeedbackResponse,
} from "./types";

const API = "/api";

// ── Owner-token store ───────────────────────────────────
// The owner_token is returned ONCE at session creation and must be sent back on
// every session-scoped request (X-Session-Token header; ?token= for <img src>).
// We keep it in memory + localStorage (keyed by sessionId) so a reload can still
// authenticate. localStorage is guarded so this never throws during SSR.

const TOKEN_PREFIX = "sx_token_";
const _memoryTokens = new Map<string, string>();

function tokenKey(sessionId: string): string {
  return `${TOKEN_PREFIX}${sessionId}`;
}

export function setSessionToken(sessionId: string, token: string): void {
  _memoryTokens.set(sessionId, token);
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(tokenKey(sessionId), token);
    } catch {
      /* storage unavailable — in-memory copy still works for this tab */
    }
  }
}

export function getSessionToken(sessionId: string): string | null {
  const mem = _memoryTokens.get(sessionId);
  if (mem) return mem;
  if (typeof window !== "undefined") {
    try {
      const t = window.localStorage.getItem(tokenKey(sessionId));
      if (t) {
        _memoryTokens.set(sessionId, t);
        return t;
      }
    } catch {
      /* ignore */
    }
  }
  return null;
}

export function clearSessionToken(sessionId: string): void {
  _memoryTokens.delete(sessionId);
  if (typeof window !== "undefined") {
    try {
      window.localStorage.removeItem(tokenKey(sessionId));
    } catch {
      /* ignore */
    }
  }
}

// ── Fetch wrapper: auth header + timeout + error body ──

const DEFAULT_TIMEOUT_MS = 30_000;

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface ApiFetchOpts extends RequestInit {
  /** When set, the matching owner token is attached as X-Session-Token. */
  sessionId?: string;
  /** Per-request timeout (default 30s). */
  timeoutMs?: number;
}

async function apiFetch(path: string, opts: ApiFetchOpts = {}): Promise<Response> {
  const { sessionId, timeoutMs = DEFAULT_TIMEOUT_MS, headers, ...rest } = opts;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  const finalHeaders: Record<string, string> = {};
  if (headers) {
    // Copy provided headers (Content-Type for JSON bodies, etc.)
    for (const [k, v] of Object.entries(headers)) {
      if (typeof v === "string") finalHeaders[k] = v;
    }
  }
  if (sessionId) {
    const token = getSessionToken(sessionId);
    if (token) finalHeaders["X-Session-Token"] = token;
  }

  try {
    return await fetch(`${API}${path}`, {
      ...rest,
      headers: finalHeaders,
      signal: ctrl.signal,
    });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new ApiError(0, `Request timed out (${Math.round(timeoutMs / 1000)}s): ${path}`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function asJson<T>(p: Promise<Response>, label: string): Promise<T> {
  const res = await p;
  if (!res.ok) {
    let detail = "";
    try {
      detail = await res.text();
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, `${label}：${res.status}${detail ? ` ${detail}` : ""}`.trim());
  }
  return (await res.json()) as T;
}

// ── Public endpoints (no auth) ──────────────────────────

export async function getStyles(): Promise<StyleListResponse> {
  return asJson(apiFetch("/styles"), "GET /styles");
}

export async function getPricingTiers(): Promise<PricingResponse> {
  return asJson(apiFetch("/sessions/pricing/tiers"), "GET /pricing/tiers");
}

/** Public site config (ICP filing number for CN deployments, etc.). For the
 *  overseas build icp_beian is empty and the footer auto-hides the block.
 *  No auth; safe to call pre-session. */
export async function getPublicConfig(): Promise<{ icp_beian: string }> {
  return asJson(apiFetch("/config/public"), "GET /config/public");
}

// ── Session lifecycle ───────────────────────────────────

export async function createSession(
  style: StyleKey,
  gender: Gender
): Promise<SessionResponse> {
  const sess = await asJson<SessionResponse>(
    apiFetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ style, gender }),
    }),
    "POST /sessions"
  );
  // Persist the one-time owner token so subsequent calls + reloads authenticate.
  if (sess.owner_token) setSessionToken(sess.session_id, sess.owner_token);
  return sess;
}

export async function uploadPhotos(
  sessionId: string,
  files: File[],
  faceProcessingConsent: boolean,
  adultSubjectConfirmed: boolean
): Promise<SessionResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("face_processing_consent", String(faceProcessingConsent));
  form.append("adult_subject_confirmed", String(adultSubjectConfirmed));
  // NOTE: do NOT set Content-Type — the browser sets the multipart boundary.
  return asJson(
    apiFetch(`/sessions/${sessionId}/photos`, {
      method: "POST",
      sessionId,
      body: form,
      // uploads of several 10MB images may take a while
      timeoutMs: 120_000,
    }),
    "POST /photos"
  );
}

export async function getSession(
  sessionId: string
): Promise<SessionResponse> {
  return asJson(
    apiFetch(`/sessions/${sessionId}`, { method: "GET", sessionId }),
    "GET /session"
  );
}

export async function deleteSession(sessionId: string): Promise<void> {
  await asJson(
    apiFetch(`/sessions/${sessionId}`, { method: "DELETE", sessionId }),
    "DELETE /session"
  );
  clearSessionToken(sessionId);
}

// ── Image URLs (loaded as <img src> — token via query) ──

export function imageUrl(sessionId: string, imageId: string): string {
  return withTokenQuery(`${API}/sessions/${sessionId}/images/${imageId}`, sessionId);
}

export function uploadedPhotoUrl(
  sessionId: string,
  filename: string
): string {
  return withTokenQuery(
    `${API}/sessions/${sessionId}/photos/${encodeURIComponent(filename)}`,
    sessionId
  );
}

/**
 * Download URL — appends ``?download=1`` so the backend sets
 * ``Content-Disposition: attachment`` and the browser saves the file. This works
 * even when the API is on a different origin than the page (where the HTML
 * ``download`` attribute alone is ignored). Token is still required for auth.
 */
export function downloadImageUrl(sessionId: string, imageId: string): string {
  return withTokenQuery(
    `${API}/sessions/${sessionId}/images/${imageId}`,
    sessionId,
    { download: "1" }
  );
}

function withTokenQuery(
  url: string,
  sessionId: string,
  extra?: Record<string, string>
): string {
  const params = new URLSearchParams();
  const token = getSessionToken(sessionId);
  if (token) params.set("token", token);
  if (extra) for (const [k, v] of Object.entries(extra)) params.set(k, v);
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

// ── Generation / jobs ───────────────────────────────────

export async function startHeroPreview(
  sessionId: string,
  style?: StyleKey
): Promise<JobResponse[]> {
  const url = style
    ? `/sessions/${sessionId}/hero-preview?style=${encodeURIComponent(style)}`
    : `/sessions/${sessionId}/hero-preview`;
  return asJson(
    apiFetch(url, { method: "POST", sessionId }),
    "POST /hero-preview"
  );
}

export async function unlockFullSet(
  sessionId: string
): Promise<JobResponse[]> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/unlock`, { method: "POST", sessionId }),
    "POST /unlock"
  );
}

export async function startGeneration(
  sessionId: string
): Promise<JobResponse[]> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/generate`, { method: "POST", sessionId }),
    "POST /generate"
  );
}

export async function startMultiStyleGeneration(
  sessionId: string,
  styles: StyleKey[]
): Promise<JobResponse[]> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/generate-multi`, {
      method: "POST",
      sessionId,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ styles }),
    }),
    "POST /generate-multi"
  );
}

export async function submitRevision(
  sessionId: string,
  imageId: string,
  instruction: string
): Promise<JobResponse> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/revise/${imageId}`, {
      method: "POST",
      sessionId,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction }),
    }),
    "POST /revise"
  );
}

export async function getJobs(sessionId: string): Promise<JobResponse[]> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/jobs`, { method: "GET", sessionId }),
    "GET /jobs"
  );
}

export async function submitImageFeedback(
  sessionId: string,
  imageId: string,
  req: UserFeedbackRequest
): Promise<UserFeedbackResponse> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/images/${imageId}/feedback`, {
      method: "POST",
      sessionId,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
    "POST /feedback"
  );
}

// ── Post-processing ─────────────────────────────────────

function postJson<T>(
  path: string,
  sessionId: string,
  body: unknown,
  label: string
): Promise<T> {
  return asJson<T>(
    apiFetch(path, {
      method: "POST",
      sessionId,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    label
  );
}

export function cropIDPhoto(
  sessionId: string,
  req: PostProcessCropRequest
): Promise<PostProcessResponse> {
  return postJson(`/sessions/${sessionId}/crop`, sessionId, req, "POST /crop");
}

export function replaceBackground(
  sessionId: string,
  req: PostProcessBgRequest
): Promise<PostProcessResponse> {
  return postJson(`/sessions/${sessionId}/background`, sessionId, req, "POST /background");
}

export function cropAndReplaceBg(
  sessionId: string,
  req: PostProcessCombinedRequest
): Promise<PostProcessResponse> {
  return postJson(`/sessions/${sessionId}/crop-background`, sessionId, req, "POST /crop-background");
}

export function upscaleImage(
  sessionId: string,
  imageId: string
): Promise<PostProcessResponse> {
  return postJson(
    `/sessions/${sessionId}/upscale`,
    sessionId,
    { image_id: imageId },
    "POST /upscale"
  );
}

// ── Payment ─────────────────────────────────────────────

export async function createPayment(
  sessionId: string,
  tier: PricingTier
): Promise<PaymentResponse> {
  return asJson(
    apiFetch(`/sessions/${sessionId}/payment`, {
      method: "POST",
      sessionId,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tier }),
    }),
    "POST /payment"
  );
}

export async function getPaymentStatus(
  paymentId: string
): Promise<PaymentStatusResponse> {
  return asJson(
    apiFetch(`/sessions/payment/${paymentId}/status`, { method: "GET" }),
    "GET /payment/status"
  );
}

export async function getSessionPayment(
  sessionId: string
): Promise<PaymentResponse | null> {
  const res = await apiFetch(`/sessions/${sessionId}/payment`, {
    method: "GET",
    sessionId,
  });
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new ApiError(res.status, `GET /session payment：${res.status}`);
  }
  // Server returns null body when no paid payment exists.
  const text = await res.text();
  return text ? (JSON.parse(text) as PaymentResponse) : null;
}
