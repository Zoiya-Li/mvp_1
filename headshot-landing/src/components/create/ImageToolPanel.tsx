"use client";

import { useState } from "react";
import { Crop, Palette, Loader2, Check, Lock } from "lucide-react";
import type {
  IDPhotoSpec,
  BgColor,
  PostProcessResponse,
  GeneratedImage,
} from "@/lib/types";
import type { TierPermissions } from "@/lib/tiers";

/* ------------------------------------------------------------------ */
/*  Config                                                             */
/* ------------------------------------------------------------------ */

// `value` is the backend ID-photo spec identifier (sent to cropIDPhoto /
// cropAndReplaceBg) and MUST stay exactly as the backend expects — it is the
// backend contract, not display text. The Chinese "1寸"/"2寸" keys encode the
// server's spec lookup; do NOT rename them here.
//
// The displayed `label` is dimension-based on purpose: for an overseas audience
// "1-inch photo" is misleading (it is a Chinese unit ≈ 33mm, not an inch), so
// we surface the real print dimensions instead.
//
// NOTE (overseas product gap, tracked under deploy #63 / compliance #66): the
// spec set itself is China-centric. A US/EU launch should add backend support
// for US passport 2×2in (51×51mm) and EU/UK passport (35×45mm) specs. That is a
// backend + product change, out of scope for the frontend translation pass.
const ID_SPECS: { value: IDPhotoSpec; label: string; dims: string }[] = [
  { value: "1寸", label: "Small ID", dims: "25×35mm" },
  { value: "2寸", label: "Large ID", dims: "35×49mm" },
];

// Human label per spec value — used wherever a result message would otherwise
// interpolate the raw Chinese backend key.
const SPEC_LABEL: Record<IDPhotoSpec, string> = {
  "1寸": "Small ID (25×35mm)",
  "2寸": "Large ID (35×49mm)",
};

const BG_COLORS: { value: BgColor; label: string; css: string }[] = [
  { value: "red", label: "Red", css: "bg-red-500" },
  { value: "blue", label: "Blue", css: "bg-blue-400" },
  { value: "white", label: "White", css: "bg-white border border-stone-300" },
  { value: "gradient_gray", label: "Gradient grey", css: "bg-gradient-to-b from-stone-300 to-stone-400" },
];

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

interface Props {
  image: GeneratedImage;
  sessionId: string;
  /** Feature flags for the current tier — gate crop/bg/quick here. */
  perms: TierPermissions;
  onProcessed: (response: PostProcessResponse) => void;
  /** Open the payment/upgrade modal. */
  onUpgrade: () => void;
}

export function ImageToolPanel({
  image,
  sessionId,
  perms,
  onProcessed,
  onUpgrade,
}: Props) {
  const [activeTool, setActiveTool] = useState<"crop" | "bg" | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<string | null>(null);

  // Skip if this is already a post-processed variant
  const isProcessed = !!image.operation;

  // Crop + bg are coupled across tiers (both gated at free, both open at
  // standard+). The combined "can use any post-processing" flag drives whether
  // we render the tools at all vs. an upgrade prompt.
  const canPostProcess = perms.allow_id_photo || perms.allow_bg_replace;

  const handleCrop = async (spec: IDPhotoSpec) => {
    setLoading(true);
    setLastResult(null);
    try {
      const { cropIDPhoto } = await import("@/lib/api");
      const res = await cropIDPhoto(sessionId, {
        image_id: image.image_id,
        spec,
      });
      setLastResult(`${SPEC_LABEL[spec]} crop done`);
      onProcessed(res);
    } catch {
      setLastResult("Crop failed — please try again");
    } finally {
      setLoading(false);
    }
  };

  const handleBg = async (color: BgColor) => {
    setLoading(true);
    setLastResult(null);
    try {
      const { replaceBackground } = await import("@/lib/api");
      const res = await replaceBackground(sessionId, {
        image_id: image.image_id,
        color,
      });
      setLastResult("Background swapped");
      onProcessed(res);
    } catch {
      setLastResult("Background swap failed — please try again");
    } finally {
      setLoading(false);
    }
  };

  const handleCropAndBg = async (spec: IDPhotoSpec, color: BgColor) => {
    setLoading(true);
    setLastResult(null);
    try {
      const { cropAndReplaceBg } = await import("@/lib/api");
      const res = await cropAndReplaceBg(sessionId, {
        image_id: image.image_id,
        spec,
        color,
      });
      setLastResult(`${SPEC_LABEL[spec]} + background swap done`);
      onProcessed(res);
    } catch {
      setLastResult("Processing failed — please try again");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="border-t border-stone-200 mt-4 pt-4">
      {/* Locked state — feature not unlocked for this tier. */}
      {!canPostProcess && (
        <div className="flex flex-col items-center gap-3 py-2">
          <div className="flex items-center gap-1.5 text-sm text-stone-500">
            <Lock size={14} className="text-stone-400" />
            ID-photo crop and background swap are premium-plan features
          </div>
          <button
            onClick={onUpgrade}
            className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-accent text-white text-xs font-medium hover:bg-accent-hover transition-colors"
          >
            <Lock size={12} />
            Upgrade to unlock
          </button>
        </div>
      )}

      {/* Tool buttons */}
      {canPostProcess && (
        <div className="flex items-center gap-3">
          <button
            onClick={() => setActiveTool(activeTool === "crop" ? null : "crop")}
            disabled={loading || isProcessed}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTool === "crop"
                ? "bg-accent text-white"
                : "bg-stone-100 text-stone-700 hover:bg-stone-200"
            } disabled:opacity-40`}
          >
            <Crop size={14} />
            ID-photo crop
          </button>
          <button
            onClick={() => setActiveTool(activeTool === "bg" ? null : "bg")}
            disabled={loading || isProcessed}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activeTool === "bg"
                ? "bg-accent text-white"
                : "bg-stone-100 text-stone-700 hover:bg-stone-200"
            } disabled:opacity-40`}
          >
            <Palette size={14} />
            Swap background
          </button>

          {loading && (
            <Loader2 size={16} className="animate-spin text-accent" />
          )}
        </div>
      )}

      {isProcessed && (
        <p className="mt-2 text-xs text-stone-400">
          {/* image.operation is a backend-generated identifier (e.g. "crop_1寸"
              or "cropbg_2寸_blue"). It may contain the Chinese spec key since
              that is the server's spec lookup value — localizing it fully
              requires backend changes (see ID_SPECS note above). */}
          This photo is already a processed result ({image.operation})
        </p>
      )}

      {/* Crop options */}
      {canPostProcess && activeTool === "crop" && !loading && (
        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs text-stone-500">Choose size:</span>
          {ID_SPECS.map((s) => (
            <button
              key={s.value}
              onClick={() => handleCrop(s.value)}
              className="px-3 py-1 rounded-md bg-stone-50 border border-stone-200 text-xs font-medium text-stone-700 hover:bg-accent/10 hover:border-accent/30 transition-colors"
            >
              {s.label}
              <span className="text-stone-400 ml-1">{s.dims}</span>
            </button>
          ))}
        </div>
      )}

      {/* Background options */}
      {canPostProcess && activeTool === "bg" && !loading && (
        <div className="mt-3 flex items-center gap-3">
          <span className="text-xs text-stone-500">Choose background:</span>
          {BG_COLORS.map((c) => (
            <button
              key={c.value}
              onClick={() => handleBg(c.value)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md hover:bg-stone-50 transition-colors group"
            >
              <span
                className={`w-5 h-5 rounded-full inline-block ${c.css}`}
              />
              <span className="text-xs text-stone-600 group-hover:text-stone-900">
                {c.label}
              </span>
            </button>
          ))}
        </div>
      )}

      {/* Quick combos */}
      {canPostProcess && activeTool && !loading && (
        <div className="mt-2 flex items-center gap-2">
          <span className="text-xs text-stone-400">Quick:</span>
          <button
            onClick={() => handleCropAndBg("1寸", "blue")}
            className="px-2.5 py-0.5 rounded-md text-xs bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100 transition-colors"
          >
            Small ID · blue
          </button>
          <button
            onClick={() => handleCropAndBg("2寸", "red")}
            className="px-2.5 py-0.5 rounded-md text-xs bg-red-50 text-red-700 border border-red-200 hover:bg-red-100 transition-colors"
          >
            Large ID · red
          </button>
          <button
            onClick={() => handleCropAndBg("1寸", "white")}
            className="px-2.5 py-0.5 rounded-md text-xs bg-stone-50 text-stone-700 border border-stone-200 hover:bg-stone-100 transition-colors"
          >
            Small ID · white
          </button>
        </div>
      )}

      {/* Result feedback */}
      {lastResult && !loading && (
        <p className="mt-2 text-xs text-accent flex items-center gap-1">
          <Check size={12} />
          {lastResult}
        </p>
      )}
    </div>
  );
}
