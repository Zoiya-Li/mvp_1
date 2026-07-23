export interface PortraitTheme {
  theme_id: string;
  slug: string;
  title: string;
  title_en: string;
  tagline: string;
  category: string;
  cover_image: string;
  preview_images: string[];
  featured: boolean;
  source_style_key: string;
  active_version: number;
  use_cases?: string[];
  shot_count?: number;
  reference_min?: number;
  reference_max?: number;
}

export interface GuestIdentity {
  user_id: string;
  access_token: string;
  created_at: string;
}

export interface PortraitProject {
  project_id: string;
  user_id: string;
  theme_id: string | null;
  source: "official_theme" | "private_inspiration" | "shared_recipe";
  status: string;
  gender: string;
  inspiration_spec: Record<string, unknown> | null;
  hero_asset_id?: string | null;
  photo_set_id?: string | null;
  legacy_session_id?: string | null;
  preview_retries_used?: number;
  preview_retries_remaining?: number;
  created_at?: string;
  updated_at?: string;
}

export interface PortraitSetAsset {
  asset_id: string;
  mime_type: string;
  position: number;
}

export interface PortraitPhotoSet {
  photo_set_id: string;
  project_id: string;
  title: string;
  status: "delivered";
  cover_asset_id: string | null;
  assets: PortraitSetAsset[];
  created_at: string;
  delivered_at: string;
}

export interface ReferenceUploadResult {
  project_id: string;
  legacy_session_id: string;
  reference_count: number;
  status: string;
  reference_quality: {
    pass?: boolean;
    issues?: string[];
    role_coverage?: ReferenceRoleQuality[];
    [key: string]: unknown;
  };
}

export interface ReferenceRoleQuality {
  role: string;
  filename?: string | null;
  pass: boolean;
  issues: string[];
  headline?: string;
  guidance?: string;
}

export interface PortraitOrder {
  order_id: string;
  project_id: string;
  product_code: "portrait_set" | "portrait_set_hd";
  status: "pending" | "paid" | "expired" | "refunded";
  amount_cents: number;
  currency: string;
  checkout_url: string | null;
}

export interface SharedRecipe {
  share_token: string;
  title: string;
  theme_id: string | null;
  theme_slug: string | null;
  source: PortraitProject["source"];
  recipe: { inspiration_spec?: Record<string, unknown> | null; [key: string]: unknown };
  portrait_available: boolean;
}

export const FALLBACK_THEMES: PortraitTheme[] = [
  {
    theme_id: "fallback-cinematic",
    slug: "cinematic-mood",
    title: "电影感叙事",
    title_en: "Cinematic Mood",
    tagline: "Step into a scene that feels pulled from a film.",
    category: "Cinematic",
    cover_image: "/images/film_f_cinematic.png",
    preview_images: [
      "/images/film_f_cinematic.png",
      "/images/film_m_cyber.png",
      "/images/film_f_dark.png",
    ],
    featured: true,
    source_style_key: "cinematic",
    active_version: 1,
  },
  {
    theme_id: "fallback-korean",
    slug: "japanese-korean-portrait",
    title: "日系·韩系写真",
    title_en: "Japanese & Korean Portrait",
    tagline: "Soft studio moods, quiet color, and a version of you worth keeping.",
    category: "Korean & Japanese",
    cover_image: "/images/kr_f_elegant.png",
    preview_images: [
      "/images/kr_f_elegant.png",
      "/images/kr_m_minimal.png",
      "/images/jp_f_fresh.png",
      "/images/jp_m_fresh.png",
    ],
    featured: true,
    source_style_key: "jk_portrait",
    active_version: 1,
  },
  {
    theme_id: "fallback-eastern",
    slug: "chinese-traditional",
    title: "东方美学",
    title_en: "Eastern Aesthetic",
    tagline: "Traditional silhouettes, photographed with a contemporary eye.",
    category: "Eastern Aesthetic",
    cover_image: "/images/gf_f_qipao.png",
    preview_images: ["/images/gf_f_qipao.png", "/images/gf_m_hanfu.png"],
    featured: true,
    source_style_key: "chinese_style",
    active_version: 1,
  },
  {
    theme_id: "fallback-lifestyle",
    slug: "lifestyle-portrait",
    title: "生活感写真",
    title_en: "Lifestyle Portrait",
    tagline: "Natural light, effortless clothes, and moments that feel lived in.",
    category: "Lifestyle",
    cover_image: "/images/social_f_french.png",
    preview_images: [
      "/images/social_f_french.png",
      "/images/social_m_street.png",
      "/images/lw_f_01.png",
      "/images/lw_m_01.png",
    ],
    featured: false,
    source_style_key: "social",
    active_version: 1,
  },
  {
    theme_id: "fallback-fashion",
    slug: "fashion-editorial",
    title: "时装编辑",
    title_en: "Fashion Editorial",
    tagline: "A clean editorial story with considered styling and light.",
    category: "Fashion",
    cover_image: "/images/fz_f_editorial.png",
    preview_images: ["/images/fz_f_editorial.png", "/images/fz_m_editorial.png"],
    featured: false,
    source_style_key: "fashion",
    active_version: 1,
  },
  {
    theme_id: "fallback-professional",
    slug: "urban-professional",
    title: "都市精英",
    title_en: "Urban Professional",
    tagline: "Quiet confidence, photographed like a modern leader.",
    category: "Professional",
    cover_image: "/images/bf_f_tech.png",
    preview_images: [
      "/images/bf_f_tech.png",
      "/images/bf_m_tech.png",
      "/images/bf_f_01.png",
      "/images/bf_m_01.png",
    ],
    featured: false,
    source_style_key: "business",
    active_version: 1,
  },
];

// Same-origin is the production-safe default: Caddy routes /api to FastAPI.
// Local split-port development can override this with NEXT_PUBLIC_API_URL.
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";
const USER_TOKEN_KEY = "flashshot_user_token_v2";

export function shouldBypassImageOptimization(url: string): boolean {
  return url.startsWith("/api/") || url.startsWith("blob:") || url.startsWith("data:");
}

export async function getThemes(): Promise<PortraitTheme[]> {
  const response = await fetch(`${API_BASE}/v2/themes`, { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load portrait themes");
  const payload = (await response.json()) as { themes: PortraitTheme[] };
  return payload.themes;
}

export async function getTheme(identifier: string): Promise<PortraitTheme> {
  const response = await fetch(`${API_BASE}/v2/themes/${encodeURIComponent(identifier)}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Theme not found");
  return response.json();
}

export function storedUserToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(USER_TOKEN_KEY);
}

export async function ensureGuestIdentity(): Promise<string> {
  const existing = storedUserToken();
  if (existing) return existing;
  const response = await fetch(`${API_BASE}/v2/users/guest`, { method: "POST" });
  if (!response.ok) throw new Error("Could not create a private workspace");
  const identity = (await response.json()) as GuestIdentity;
  localStorage.setItem(USER_TOKEN_KEY, identity.access_token);
  return identity.access_token;
}

export async function createPortraitProject(input: {
  theme_id?: string;
  source: PortraitProject["source"];
  gender?: "male" | "female" | "unspecified";
  shared_recipe_id?: string;
}): Promise<PortraitProject> {
  const token = await ensureGuestIdentity();
  const response = await fetch(`${API_BASE}/v2/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-User-Token": token },
    body: JSON.stringify({ gender: "unspecified", ...input }),
  });
  if (!response.ok) throw new Error("Could not start this portrait project");
  return response.json();
}

export async function uploadInspiration(projectId: string, file: File) {
  const token = await ensureGuestIdentity();
  const body = new FormData();
  body.append("file", file);
  body.append("rights_confirmed", "true");
  body.append("private_style_reference_only", "true");
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/inspiration`, {
    method: "POST",
    headers: { "X-User-Token": token },
    body,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not analyze this inspiration");
  return payload as {
    project_id: string;
    analysis_status: "analyzed" | "pending" | "rejected";
    inspiration_spec: Record<string, unknown> | null;
    message: string;
  };
}

async function userHeaders(): Promise<Record<string, string>> {
  return { "X-User-Token": await ensureGuestIdentity() };
}

export async function getPortraitProject(projectId: string): Promise<PortraitProject> {
  const response = await fetch(`${API_BASE}/v2/projects/${encodeURIComponent(projectId)}`, {
    headers: await userHeaders(),
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Could not open this portrait project");
  return response.json();
}

export async function listPortraitProjects(): Promise<PortraitProject[]> {
  const response = await fetch(`${API_BASE}/v2/projects`, {
    headers: await userHeaders(),
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Could not load your portrait library");
  const payload = (await response.json()) as { projects: PortraitProject[] };
  return payload.projects;
}

export async function uploadIdentityReferences(input: {
  projectId: string;
  files: File[];
  gender: "female" | "male";
}): Promise<ReferenceUploadResult> {
  const body = new FormData();
  input.files.forEach((file) => body.append("files", file));
  body.append("gender", input.gender);
  body.append("face_processing_consent", "true");
  body.append("adult_subject_confirmed", "true");
  const response = await fetch(`${API_BASE}/v2/projects/${input.projectId}/references`, {
    method: "POST",
    headers: await userHeaders(),
    body,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not check these photos");
  return payload;
}

export async function startPortraitPreview(projectId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/preview`, {
    method: "POST",
    headers: await userHeaders(),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not start the preview");
}

export async function retryPortraitPreview(
  projectId: string,
  reason: "identity" | "expression" | "overall",
): Promise<{ project_id: string; status: "preview_generating"; retries_remaining: number }> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/preview/retry`, {
    method: "POST",
    headers: { ...(await userHeaders()), "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not create a closer match");
  return payload;
}

export async function loadPortraitHero(projectId: string): Promise<string> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/hero`, {
    headers: await userHeaders(),
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Preview is not ready");
  return URL.createObjectURL(await response.blob());
}

export async function getPortraitPhotoSet(
  projectId: string,
  photoSetId: string,
): Promise<PortraitPhotoSet> {
  const response = await fetch(
    `${API_BASE}/v2/projects/${encodeURIComponent(projectId)}/sets/${encodeURIComponent(photoSetId)}`,
    { headers: await userHeaders(), cache: "no-store" },
  );
  if (!response.ok) throw new Error("Could not open this finished portrait set");
  return response.json();
}

export async function loadPortraitAsset(
  projectId: string,
  assetId: string,
): Promise<string> {
  const response = await fetch(
    `${API_BASE}/v2/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(assetId)}`,
    { headers: await userHeaders(), cache: "no-store" },
  );
  if (!response.ok) throw new Error("Could not open a finished portrait");
  return URL.createObjectURL(await response.blob());
}

export async function getEntitlementBalance(): Promise<number> {
  const response = await fetch(`${API_BASE}/v2/entitlements/balance`, {
    headers: await userHeaders(),
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Could not load your studio balance");
  const payload = (await response.json()) as { credit_balance: number };
  return payload.credit_balance;
}

export async function getCheckoutAvailability(): Promise<boolean> {
  const response = await fetch(`${API_BASE}/config/public`, { cache: "no-store" });
  if (!response.ok) return false;
  const payload = (await response.json()) as { checkout_available?: boolean };
  return payload.checkout_available === true;
}

export async function createPortraitOrder(
  projectId: string,
  productCode: PortraitOrder["product_code"],
): Promise<PortraitOrder> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/orders`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await userHeaders()) },
    body: JSON.stringify({ product_code: productCode }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not prepare checkout");
  return payload;
}

export async function getPortraitOrder(
  projectId: string,
  orderId: string,
): Promise<PortraitOrder> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/orders/${orderId}`, {
    headers: await userHeaders(), cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not verify payment");
  return payload;
}

export async function unlockPortraitSet(projectId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/unlock`, {
    method: "POST", headers: await userHeaders(),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not start the complete shoot");
}

export async function deletePortraitProject(projectId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}`, {
    method: "DELETE", headers: await userHeaders(),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not delete this project");
}

export async function deletePortraitWorkspace(): Promise<void> {
  const response = await fetch(`${API_BASE}/v2/users/me`, {
    method: "DELETE", headers: await userHeaders(),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not delete this workspace");
  localStorage.removeItem(USER_TOKEN_KEY);
}

export async function createSharedRecipe(
  projectId: string,
  includePortrait: boolean,
): Promise<SharedRecipe> {
  const response = await fetch(`${API_BASE}/v2/projects/${projectId}/share-recipe`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await userHeaders()) },
    body: JSON.stringify({ include_portrait: includePortrait }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail ?? "Could not create a share link");
  return payload;
}

export async function getSharedRecipe(shareToken: string): Promise<SharedRecipe> {
  const response = await fetch(`${API_BASE}/v2/shares/${encodeURIComponent(shareToken)}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("This shared portrait direction is no longer available");
  return response.json();
}

export function sharedHeroUrl(shareToken: string): string {
  return `${API_BASE}/v2/shares/${encodeURIComponent(shareToken)}/hero`;
}
