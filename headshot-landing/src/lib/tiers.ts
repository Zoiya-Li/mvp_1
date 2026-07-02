import type { PricingTier } from "./types";

/**
 * Per-tier feature flags used for CLIENT-SIDE UX gating.
 *
 * These mirror ``TIER_LIMITS`` in ``headshot_pipeline/server/models.py``. The
 * backend is always the real authority — if these ever drift, the worst case is
 * a cosmetic mismatch (a feature shown as unlocked gets a 403, or shown locked
 * when the backend would allow it). Keep them in sync when editing tiers.
 */
export interface TierPermissions {
  allow_id_photo: boolean;
  allow_bg_replace: boolean;
  allow_hd_download: boolean;
}

const TIER_PERMISSIONS: Record<PricingTier, TierPermissions> = {
  free: {
    allow_id_photo: false,
    allow_bg_replace: false,
    allow_hd_download: false,
  },
  standard: {
    allow_id_photo: true,
    allow_bg_replace: true,
    allow_hd_download: false,
  },
  premium: {
    allow_id_photo: true,
    allow_bg_replace: true,
    allow_hd_download: true,
  },
};

export function getTierPermissions(tier: PricingTier): TierPermissions {
  return TIER_PERMISSIONS[tier] ?? TIER_PERMISSIONS.free;
}
