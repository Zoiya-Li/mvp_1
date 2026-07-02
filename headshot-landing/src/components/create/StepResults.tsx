"use client";

import { useState, useCallback, useMemo } from "react";
import {
  Download,
  Pencil,
  X,
  ZoomIn,
  RefreshCw,
  ImageIcon,
  Columns2,
  Lock,
  Loader2,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import type {
  GeneratedImage,
  SessionResponse,
  GenerationMetadata,
} from "@/lib/types";
import { RevisionChat } from "./RevisionChat";
import { ImageToolPanel } from "./ImageToolPanel";
import { BeforeAfterSlider } from "./BeforeAfterSlider";
import { PaymentModal } from "./PaymentModal";
import {
  imageUrl,
  downloadImageUrl,
  uploadedPhotoUrl,
  upscaleImage,
  submitImageFeedback,
} from "@/lib/api";
import { getTierPermissions } from "@/lib/tiers";

const SHOW_QA_DEBUG =
  process.env.NODE_ENV !== "production" ||
  process.env.NEXT_PUBLIC_QA_DEBUG === "1";

function fmtScore(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(value % 1 ? 1 : 0) : "–";
}

function fmtCosine(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(3) : "–";
}

function metricNumber(
  metrics: Record<string, unknown>,
  key: string
): number | null {
  const value = metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metricRecord(
  metrics: Record<string, unknown>,
  key: string
): Record<string, unknown> | null {
  const value = metrics[key];
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function fmtRate(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "–";
}

function fmtMoney(value: number | null | undefined) {
  return typeof value === "number" ? `$${value.toFixed(2)}` : "–";
}

function fmtMs(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value)}ms` : "–";
}

function fmtSec(value: number | null | undefined) {
  return typeof value === "number" ? `${Math.round(value)}s` : "–";
}

function selectedCandidate(meta?: GenerationMetadata | null) {
  if (!meta?.candidates || !meta.selected_candidate?.index) return null;
  return (
    meta.candidates.find((c) => c.index === meta.selected_candidate?.index) ??
    null
  );
}

function SessionDebugPanel({ session }: { session: SessionResponse }) {
  const metrics = session.pipeline_metrics ?? {};
  const feedback = session.feedback_summary ?? {};
  const actionMetrics = metricRecord(metrics, "agent_action_metrics") ?? {};
  const shotMetrics = metricRecord(metrics, "shot_metrics") ?? {};

  const primaryMetrics = [
    ["Input pass", fmtRate(metricNumber(metrics, "input_photo_pass_rate"))],
    ["ID first", fmtRate(metricNumber(metrics, "identity_first_pass_rate"))],
    ["Deliverable", fmtRate(metricNumber(metrics, "deliverable_rate"))],
    [
      "North star",
      fmtRate(
        metricNumber(metrics, "north_star_qualified_save_rate") ||
          metricNumber(feedback, "qualified_saved_rate"),
      ),
    ],
    ["Saved", fmtRate(metricNumber(feedback, "user_saved_rate"))],
    ["Q saved", fmtRate(metricNumber(feedback, "qualified_saved_rate"))],
    ["Selected", fmtRate(metricNumber(feedback, "user_selected_rate"))],
    ["Not me", fmtRate(metricNumber(feedback, "not_like_me_rate"))],
    ["Refund", fmtRate(metricNumber(metrics, "refund_rate"))],
    ["Cost/img", fmtMoney(metricNumber(metrics, "estimated_cost_per_image"))],
    [
      "Cost/pass",
      fmtMoney(metricNumber(metrics, "estimated_cost_per_deliverable")),
    ],
    ["API/img", fmtScore(metricNumber(metrics, "avg_api_calls_per_image"))],
    ["P95 API", fmtMs(metricNumber(metrics, "p95_provider_latency_ms"))],
    ["P95 delivery", fmtSec(metricNumber(metrics, "p95_delivery_latency_seconds"))],
    ["Failures", fmtScore(metricNumber(metrics, "generation_failures"))],
  ];

  const actionRows = [
    ["Identity repair", "IDENTITY_REPAIR"],
    ["Local edit", "LOCAL_EDIT"],
    ["Regen", "REGENERATE_FROM_ORIGINAL"],
  ].map(([label, key]) => {
    const row = metricRecord(actionMetrics, key) ?? {};
    return {
      label,
      attempts: metricNumber(row, "attempts"),
      successes: metricNumber(row, "successes"),
      successRate: metricNumber(row, "success_rate"),
    };
  });

  const shotRows = Object.entries(shotMetrics)
    .slice(0, 4)
    .map(([shotId, raw]) => {
      const row =
        raw && typeof raw === "object" && !Array.isArray(raw)
          ? (raw as Record<string, unknown>)
          : {};
      return {
        shotId,
        attempts: metricNumber(row, "attempts"),
        identityFirstPassRate: metricNumber(row, "identity_first_pass_rate"),
        deliverableRate: metricNumber(row, "deliverable_rate"),
        failureRate: metricNumber(row, "failure_rate"),
        notLikeMeRate: metricNumber(row, "not_like_me_rate"),
      };
    });

  return (
    <div className="mt-6 rounded-lg border border-stone-200 bg-stone-50/80 px-4 py-3 text-[11px] text-stone-600">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium text-stone-800">Session QA</span>
        <span className="text-stone-400">
          {session.status} · {session.tier}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        {primaryMetrics.map(([label, value]) => (
          <div
            key={label}
            className="rounded-md border border-stone-200 bg-white px-2 py-1.5"
          >
            <div className="text-stone-400">{label}</div>
            <div className="mt-0.5 font-medium text-stone-800">{value}</div>
          </div>
        ))}
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <div className="rounded-md border border-stone-200 bg-white px-3 py-2">
          <div className="font-medium text-stone-700">Agent actions</div>
          <div className="mt-2 space-y-1">
            {actionRows.map((row) => (
              <div
                key={row.label}
                className="grid grid-cols-[1fr_auto_auto_auto] gap-2"
              >
                <span>{row.label}</span>
                <span>{fmtScore(row.attempts)} tries</span>
                <span>{fmtScore(row.successes)} pass</span>
                <span className="font-medium text-stone-800">
                  {fmtRate(row.successRate)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-stone-200 bg-white px-3 py-2">
          <div className="font-medium text-stone-700">Shots</div>
          <div className="mt-2 space-y-1">
            {shotRows.length ? (
              shotRows.map((row) => (
                <div
                  key={row.shotId}
                  className="grid grid-cols-[1fr_auto_auto_auto_auto_auto] gap-2"
                >
                  <span className="truncate">{row.shotId}</span>
                  <span>{fmtScore(row.attempts)} tries</span>
                  <span>ID {fmtRate(row.identityFirstPassRate)}</span>
                  <span>pass {fmtRate(row.deliverableRate)}</span>
                  <span>fail {fmtRate(row.failureRate)}</span>
                  <span>not-me {fmtRate(row.notLikeMeRate)}</span>
                </div>
              ))
            ) : (
              <div className="text-stone-400">–</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function DebugScorePanel({ image }: { image: GeneratedImage }) {
  const meta = image.resemblance;
  const candidate = selectedCandidate(meta);
  const judgement = candidate?.judgement;
  const scores = judgement?.scores;
  const identity = judgement?.identity_quality;
  const local = judgement?.local_quality;
  const action =
    candidate?.agent_action?.action ??
    meta?.agent_actions?.find((item) => item.candidate_index === candidate?.index)
      ?.action;
  const deliverable = meta?.selected_candidate?.deliverable;
  const measurements = local?.measurements ?? {};
  const faceCount =
    typeof measurements.face_count === "number" ? measurements.face_count : null;
  const blur =
    typeof measurements.blur_variance === "number"
      ? measurements.blur_variance
      : null;

  if (!meta || !candidate) return null;

  return (
    <div className="mt-2 rounded-lg border border-stone-200 bg-white px-3 py-2 text-[11px] text-stone-500">
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-stone-700">
          QA {meta.pipeline ?? "pipeline"}
        </span>
        <span>
          cand {candidate.index} · agg {fmtScore(candidate.aggregate_score)}
        </span>
      </div>
      <div className="mt-1 grid grid-cols-3 gap-x-2 gap-y-1">
        <span>ID {fmtScore(scores?.identity)}</span>
        <span>Face {fmtScore(scores?.face_quality)}</span>
        <span>Style {fmtScore(scores?.style_match)}</span>
        <span>Art {fmtScore(scores?.artifact)}</span>
        <span>Ready {fmtScore(scores?.commercial_readiness)}</span>
        <span>cos {fmtCosine(identity?.cosine_similarity)}</span>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
        <span>deliver {deliverable ? "yes" : "no"}</span>
        <span>action {action ?? "–"}</span>
        <span>ref {fmtCosine(identity?.reference_consistency)}</span>
        <span>faces {faceCount ?? "–"}</span>
        <span>blur {blur ? Math.round(blur) : "–"}</span>
        <span>swap {meta.face_swap?.applied ? "yes" : "no"}</span>
      </div>
      {(judgement?.hard_failures?.length || identity?.hard_failures?.length) ? (
        <div className="mt-1 truncate text-amber-600">
          {[...(judgement?.hard_failures ?? []), ...(identity?.hard_failures ?? [])]
            .filter(Boolean)
            .join(", ")}
        </div>
      ) : null}
    </div>
  );
}

interface Props {
  session: SessionResponse;
  onRevise: (imageId: string, instruction: string) => Promise<void>;
  onStartOver?: () => void;
  onRefreshSession?: () => void;
}

export function StepResults({
  session,
  onRevise,
  onStartOver,
  onRefreshSession,
}: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [reviseFor, setReviseFor] = useState<string | null>(null);
  const [showCompare, setShowCompare] = useState(false);
  const [showUpgrade, setShowUpgrade] = useState(false);
  const [upscaling, setUpscaling] = useState(false);
  const [feedbackByImage, setFeedbackByImage] = useState<Record<string, string>>({});

  const images = session.generated_images;
  const selectedImg = images.find((i) => i.image_id === selected);
  const perms = useMemo(() => getTierPermissions(session.tier), [session.tier]);

  const handleProcessed = useCallback(() => {
    // Refresh session to get the new processed image in the list.
    // (ImageToolPanel calls onProcessed(response); we don't need the response —
    // a fewer-param callback is assignable, so we drop it to stay warning-free.)
    onRefreshSession?.();
  }, [onRefreshSession]);

  const recordFeedback = useCallback(
    async (
      imageId: string,
      event: "downloaded" | "selected" | "looks_like_me" | "not_like_me",
      score?: number
    ) => {
      setFeedbackByImage((prev) => ({ ...prev, [imageId]: event }));
      try {
        await submitImageFeedback(session.session_id, imageId, {
          event,
          score,
        });
      } catch {
        // Feedback is analytics, not a blocking user action.
      }
    },
    [session.session_id]
  );

  // Empty state — all jobs failed or no images generated
  if (images.length === 0) {
    return (
      <div className="max-w-md mx-auto text-center py-16">
        <div className="w-16 h-16 rounded-full bg-stone-100 flex items-center justify-center mx-auto mb-4">
          <ImageIcon size={28} className="text-stone-400" />
        </div>
        <h2 className="text-2xl font-semibold tracking-tight">
          No portraits generated yet
        </h2>
        <p className="mt-3 text-stone-500 text-sm leading-relaxed">
          Portrait generation didn&apos;t succeed — the service may still be starting up.
          <br />
          Please try again in a moment, or start over.
        </p>
        {onStartOver && (
          <button
            onClick={onStartOver}
            className="mt-6 h-11 px-7 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors inline-flex items-center gap-2"
          >
            <RefreshCw size={16} />
            Start over
          </button>
        )}
      </div>
    );
  }

  // Determine if comparison is possible (uses a tokenized URL for the upload).
  const hasOriginalPhoto =
    session.uploaded_photos && session.uploaded_photos.length > 0;
  const originalPhotoUrl = hasOriginalPhoto
    ? uploadedPhotoUrl(
        session.session_id,
        session.uploaded_photos[0]
      )
    : null;

  // Display URLs (tokenized) — the raw `img.url` carries no token and the image
  // endpoint 401s without one, so we rebuild the URL client-side.
  const displayUrl = (img: GeneratedImage) =>
    imageUrl(session.session_id, img.image_id);

  return (
    <div className="max-w-5xl mx-auto">
      <h2 className="text-3xl font-semibold tracking-tight text-center">
        Your AI portrait collection
      </h2>
      <p className="mt-3 text-stone-500 text-center">
        Click any image to view it full size. Not happy? Hit &quot;Revise&quot; to let the AI generate another.
      </p>

      {SHOW_QA_DEBUG && <SessionDebugPanel session={session} />}

      {/* Upgrade banner for free tier */}
      {session.tier === "free" && (
        <div className="mt-6 flex justify-center">
          <button
            onClick={() => setShowUpgrade(true)}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-accent/10 text-accent text-sm font-medium hover:bg-accent/20 transition-colors"
          >
            <Lock size={14} />
            Upgrade to unlock ID-photo crop, background swap, and HD downloads
          </button>
        </div>
      )}

      {/* Image grid */}
      <div className="mt-8 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {images.map((img) => {
          const dUrl = displayUrl(img);
          return (
            <div key={img.image_id} className="group">
              <div className="relative aspect-[3/4] rounded-2xl overflow-hidden bg-stone-100 cursor-pointer">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={dUrl}
                  alt={`Portrait ${img.prompt_id}`}
                  className="object-cover w-full h-full transition-transform duration-300 group-hover:scale-105"
                  onClick={() => {
                    setSelected(img.image_id);
                    void recordFeedback(img.image_id, "selected");
                  }}
                />

                {/* Overlay actions */}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-stone-900/60 to-transparent p-3 pt-8 opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => {
                        setSelected(img.image_id);
                        void recordFeedback(img.image_id, "selected");
                      }}
                      className="text-white text-xs flex items-center gap-1"
                    >
                      <ZoomIn size={14} /> View
                    </button>
                    <div className="flex gap-2">
                      <a
                        href={downloadImageUrl(session.session_id, img.image_id)}
                        download
                        onClick={() =>
                          void recordFeedback(img.image_id, "downloaded", 2)
                        }
                        className="text-white text-xs flex items-center gap-1"
                      >
                        <Download size={14} /> Download
                      </a>
                      <button
                        onClick={() =>
                          setReviseFor(
                            reviseFor === img.image_id ? null : img.image_id
                          )
                        }
                        className="text-white text-xs flex items-center gap-1"
                      >
                        <Pencil size={14} /> Revise
                      </button>
                    </div>
                  </div>
                </div>

                {/* Badges */}
                {img.turn > 1 && !img.operation && (
                  <span className="absolute top-2 right-2 bg-accent/90 text-white text-xs px-2 py-0.5 rounded-full">
                    v{img.turn}
                  </span>
                )}
                {img.operation && (
                  <span className="absolute top-2 right-2 bg-stone-900/70 text-white text-xs px-2 py-0.5 rounded-full">
                    {img.operation.includes("crop") && img.operation.includes("bg")
                      ? img.operation.replace("cropbg_", "")
                      : img.operation.replace(/^(crop_|bg_)/, "")}
                  </span>
                )}
              </div>

              {/* Revision chat */}
              {SHOW_QA_DEBUG && <DebugScorePanel image={img} />}

              {!img.operation && (
                <div className="mt-2 flex gap-2">
                  <button
                    onClick={() =>
                      void recordFeedback(img.image_id, "looks_like_me", 2)
                    }
                    className={`inline-flex h-8 flex-1 items-center justify-center gap-1 rounded-full border text-xs transition-colors ${
                      feedbackByImage[img.image_id] === "looks_like_me"
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-stone-200 text-stone-500 hover:bg-stone-50"
                    }`}
                  >
                    <ThumbsUp size={13} />
                    Like me
                  </button>
                  <button
                    onClick={() =>
                      void recordFeedback(img.image_id, "not_like_me", 0)
                    }
                    className={`inline-flex h-8 flex-1 items-center justify-center gap-1 rounded-full border text-xs transition-colors ${
                      feedbackByImage[img.image_id] === "not_like_me"
                        ? "border-rose-200 bg-rose-50 text-rose-700"
                        : "border-stone-200 text-stone-500 hover:bg-stone-50"
                    }`}
                  >
                    <ThumbsDown size={13} />
                    Not me
                  </button>
                </div>
              )}

              {reviseFor === img.image_id && !img.operation && (
                <RevisionChat
                  image={img}
                  revisionsUsed={session.revisions_used}
                  maxRevisions={session.max_revisions}
                  onSubmit={onRevise}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Start over */}
      {onStartOver && (
        <div className="mt-10 text-center">
          <button
            onClick={onStartOver}
            className="h-10 px-5 rounded-full border border-stone-300 text-stone-600 text-sm font-medium hover:bg-stone-50 transition-colors inline-flex items-center gap-2"
          >
            <RefreshCw size={14} />
            Start over
          </button>
        </div>
      )}

      {/* Lightbox with tools */}
      {selectedImg && (
        <div
          className="fixed inset-0 z-50 bg-stone-900/80 flex items-center justify-center p-4 md:p-8"
          onClick={() => {
            setSelected(null);
            setShowCompare(false);
          }}
        >
          <button
            onClick={() => {
              setSelected(null);
              setShowCompare(false);
            }}
            className="absolute top-6 right-6 text-white z-10"
          >
            <X size={28} />
          </button>

          <div
            className="flex flex-col items-center max-h-full overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Image or comparison slider */}
            <div className="relative max-h-[70vh] w-full flex items-center justify-center">
              {showCompare && originalPhotoUrl ? (
                <BeforeAfterSlider
                  beforeSrc={originalPhotoUrl}
                  afterSrc={displayUrl(selectedImg)}
                  className="max-h-[65vh] rounded-2xl"
                />
              ) : (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img
                  src={displayUrl(selectedImg)}
                  alt="Full size"
                  className="max-h-[65vh] max-w-full rounded-2xl shadow-2xl"
                />
              )}
            </div>

            {/* Toolbar */}
            <div className="mt-3 flex items-center gap-3 flex-wrap justify-center">
              {/* Compare toggle */}
              {hasOriginalPhoto && !selectedImg.operation && (
                <button
                  onClick={() => setShowCompare(!showCompare)}
                  className={`h-8 px-3 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-colors ${
                    showCompare
                      ? "bg-white text-stone-900"
                      : "bg-white/20 text-white hover:bg-white/30"
                  }`}
                >
                  <Columns2 size={13} />
                  {showCompare ? "Back to portrait" : "Compare to original"}
                </button>
              )}

              {/* Download */}
              <a
                href={downloadImageUrl(session.session_id, selectedImg.image_id)}
                download
                className="h-8 px-3 rounded-lg bg-white/20 text-white text-xs font-medium flex items-center gap-1.5 hover:bg-white/30 transition-colors"
              >
                <Download size={13} />
                Download
              </a>

              {/* HD Upscale — only when the tier allows it. */}
              {perms.allow_hd_download &&
                !selectedImg.operation?.startsWith("upscale") && (
                  <button
                    onClick={async () => {
                      if (upscaling) return;
                      setUpscaling(true);
                      try {
                        await upscaleImage(
                          session.session_id,
                          selectedImg.image_id
                        );
                        onRefreshSession?.();
                      } catch (e) {
                        console.error("Upscale failed:", e);
                      } finally {
                        setUpscaling(false);
                      }
                    }}
                    disabled={upscaling}
                    className="h-8 px-3 rounded-lg bg-white/20 text-white text-xs font-medium flex items-center gap-1.5 hover:bg-white/30 transition-colors disabled:opacity-50"
                  >
                    {upscaling ? (
                      <Loader2 size={13} className="animate-spin" />
                    ) : (
                      <Download size={13} />
                    )}
                    {upscaling ? "Processing HD…" : "Download HD 2×"}
                  </button>
                )}

              {/* HD locked upsell — tier doesn't allow HD download. */}
              {!perms.allow_hd_download && !showUpgrade && (
                <button
                  onClick={() => setShowUpgrade(true)}
                  className="h-8 px-3 rounded-lg bg-white/10 text-white/80 text-xs font-medium flex items-center gap-1.5 hover:bg-white/20 transition-colors"
                >
                  <Lock size={13} />
                  Upgrade for HD 2×
                </button>
              )}
            </div>

            {/* Post-processing tools (tier-gated inside the panel) */}
            {!showCompare && (
              <div className="mt-2 w-full max-w-lg bg-white rounded-xl p-4">
                <ImageToolPanel
                  image={selectedImg}
                  sessionId={session.session_id}
                  perms={perms}
                  onProcessed={handleProcessed}
                  onUpgrade={() => setShowUpgrade(true)}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Payment upgrade modal */}
      {showUpgrade && (
        <PaymentModal
          sessionId={session.session_id}
          currentTier={session.tier}
          onClose={() => setShowUpgrade(false)}
          onPaymentSuccess={() => {
            setShowUpgrade(false);
            onRefreshSession?.();
          }}
        />
      )}
    </div>
  );
}
