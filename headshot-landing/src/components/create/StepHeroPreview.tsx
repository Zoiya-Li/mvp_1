"use client";

import { useState } from "react";
import {
  Sparkles,
  RefreshCw,
  Lock,
  ThumbsUp,
  ThumbsDown,
  ArrowRight,
  ImageIcon,
} from "lucide-react";
import type { SessionResponse } from "@/lib/types";
import { imageUrl, submitImageFeedback } from "@/lib/api";
import { PaymentModal } from "./PaymentModal";
import { RevisionChat } from "./RevisionChat";

interface Props {
  session: SessionResponse;
  onUnlock: () => Promise<void>;
  onRevise: (imageId: string, instruction: string) => Promise<void>;
  onTryAnotherStyle: () => void;
  onStartOver?: () => void;
}

export function StepHeroPreview({
  session,
  onUnlock,
  onRevise,
  onTryAnotherStyle,
  onStartOver,
}: Props) {
  const [showUpgrade, setShowUpgrade] = useState(false);
  const [reviseOpen, setReviseOpen] = useState(false);
  const [unlocking, setUnlocking] = useState(false);
  const [feedbackSent, setFeedbackSent] = useState<string | null>(null);

  const heroImageId = session.hero_preview_image_id;
  const heroImage = session.generated_images.find(
    (img) => img.image_id === heroImageId
  );
  const isPaid = session.tier !== "free";

  const handleUnlock = async () => {
    if (unlocking) return;
    setUnlocking(true);
    try {
      await onUnlock();
    } catch (e) {
      console.error("Unlock failed:", e);
    } finally {
      setUnlocking(false);
    }
  };

  const handleFeedback = async (
    event: "looks_like_me" | "not_like_me"
  ) => {
    if (!heroImageId || feedbackSent) return;
    setFeedbackSent(event);
    try {
      await submitImageFeedback(session.session_id, heroImageId, {
        event,
        score: event === "looks_like_me" ? 2 : 0,
      });
    } catch {
      // feedback is best-effort
    }
  };

  return (
    <div className="max-w-2xl mx-auto text-center">
      <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 text-accent mb-6">
        <Sparkles size={32} />
      </div>

      <h2 className="text-3xl font-semibold tracking-tight">
        Your preview portrait is ready
      </h2>
      <p className="mt-3 text-stone-500">
        This is a single preview generated from your photos. Unlock the full set
        to get a curated collection of portraits in this style.
      </p>

      {/* Hero image */}
      <div className="mt-8">
        {heroImage ? (
          <div className="relative mx-auto max-w-md">
            <div className="aspect-[3/4] rounded-2xl overflow-hidden bg-stone-100 shadow-lg">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imageUrl(session.session_id, heroImage.image_id)}
                alt="Preview portrait"
                className="object-cover w-full h-full"
              />
            </div>

            {/* Feedback buttons */}
            <div className="mt-4 flex gap-3 justify-center">
              <button
                onClick={() => handleFeedback("looks_like_me")}
                disabled={feedbackSent !== null}
                className={`inline-flex h-10 items-center gap-2 rounded-full border px-5 text-sm font-medium transition-colors ${
                  feedbackSent === "looks_like_me"
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                    : "border-stone-200 text-stone-600 hover:bg-stone-50"
                }`}
              >
                <ThumbsUp size={15} />
                Looks like me
              </button>
              <button
                onClick={() => handleFeedback("not_like_me")}
                disabled={feedbackSent !== null}
                className={`inline-flex h-10 items-center gap-2 rounded-full border px-5 text-sm font-medium transition-colors ${
                  feedbackSent === "not_like_me"
                    ? "border-rose-200 bg-rose-50 text-rose-700"
                    : "border-stone-200 text-stone-600 hover:bg-stone-50"
                }`}
              >
                <ThumbsDown size={15} />
                Not like me
              </button>
            </div>
          </div>
        ) : (
          <div className="mx-auto max-w-md rounded-2xl bg-stone-100 p-12">
            <ImageIcon size={48} className="mx-auto text-stone-300" />
            <p className="mt-4 text-stone-400 text-sm">
              Preview image not found. Please try again.
            </p>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="mt-8 space-y-3">
        {/* Unlock full set */}
        {isPaid ? (
          <button
            onClick={handleUnlock}
            disabled={unlocking}
            className="inline-flex h-12 items-center gap-2 rounded-full bg-accent px-8 text-white font-medium text-sm hover:bg-accent-hover transition-colors disabled:opacity-50"
          >
            {unlocking ? (
              <RefreshCw size={16} className="animate-spin" />
            ) : (
              <Sparkles size={16} />
            )}
            {unlocking ? "Starting full set…" : "Generate full portrait set"}
            <ArrowRight size={16} />
          </button>
        ) : (
          <button
            onClick={() => setShowUpgrade(true)}
            className="inline-flex h-12 items-center gap-2 rounded-full bg-accent px-8 text-white font-medium text-sm hover:bg-accent-hover transition-colors"
          >
            <Lock size={16} />
            Unlock full portrait set
            <ArrowRight size={16} />
          </button>
        )}

        {/* Secondary actions */}
        <div className="flex flex-wrap justify-center gap-3 pt-2">
          {heroImage && (
            <button
              onClick={() => setReviseOpen(!reviseOpen)}
              className="inline-flex h-10 items-center gap-1.5 rounded-full border border-stone-200 px-5 text-sm text-stone-600 hover:bg-stone-50 transition-colors"
            >
              <RefreshCw size={14} />
              {reviseOpen ? "Close revision" : "Make it more like me"}
            </button>
          )}
          <button
            onClick={onTryAnotherStyle}
            className="inline-flex h-10 items-center gap-1.5 rounded-full border border-stone-200 px-5 text-sm text-stone-600 hover:bg-stone-50 transition-colors"
          >
            <RefreshCw size={14} />
            Try another style
          </button>
          {onStartOver && (
            <button
              onClick={onStartOver}
              className="inline-flex h-10 items-center gap-1.5 rounded-full border border-stone-200 px-5 text-sm text-stone-600 hover:bg-stone-50 transition-colors"
            >
              Start over
            </button>
          )}
        </div>
      </div>

      {/* Revision chat */}
      {reviseOpen && heroImage && (
        <div className="mt-6 mx-auto max-w-md">
          <RevisionChat
            image={heroImage}
            revisionsUsed={session.revisions_used}
            maxRevisions={session.max_revisions}
            onSubmit={async (imageId, instruction) => {
              await onRevise(imageId, instruction);
              setReviseOpen(false);
            }}
          />
        </div>
      )}

      {/* Upgrade modal */}
      {showUpgrade && (
        <PaymentModal
          sessionId={session.session_id}
          currentTier={session.tier}
          onClose={() => setShowUpgrade(false)}
          onPaymentSuccess={() => {
            setShowUpgrade(false);
            handleUnlock();
          }}
        />
      )}
    </div>
  );
}
