"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { X, Check, Loader2 } from "lucide-react";
import type { PricingTier, TierInfo } from "@/lib/types";
import {
  getPricingTiers,
  createPayment,
  getPaymentStatus,
} from "@/lib/api";

interface Props {
  sessionId: string;
  currentTier: PricingTier;
  onClose: () => void;
  onPaymentSuccess: () => void;
}

const TIER_BADGE: Record<PricingTier, { label: string; accent: string }> = {
  free: { label: "Free", accent: "bg-stone-100 text-stone-600" },
  standard: { label: "Standard", accent: "bg-accent/10 text-accent" },
  premium: { label: "Pro", accent: "bg-accent text-white" },
};

export function PaymentModal({
  sessionId,
  currentTier,
  onClose,
  onPaymentSuccess,
}: Props) {
  const [tiers, setTiers] = useState<TierInfo[]>([]);
  // Default to the tier one step above the user's current tier. Derived at
  // mount: the modal is freshly mounted each time it opens, so currentTier is
  // stable for this component's lifetime — initial-state derivation beats a
  // setState-in-effect (which the react-hooks rule forbids).
  const [selected, setSelected] = useState<PricingTier>(() => {
    const order: PricingTier[] = ["free", "standard", "premium"];
    const idx = order.indexOf(currentTier);
    return idx < 2 ? order[idx + 1] : currentTier;
  });
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(false);
  const [paymentId, setPaymentId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  // Load pricing tiers
  useEffect(() => {
    getPricingTiers()
      .then((data) => setTiers(data.tiers))
      .catch(() => setError("Couldn't load pricing"));
  }, []);

  // Poll payment status — with proper cleanup
  const pollStatus = useCallback(
    async (pid: string) => {
      setPolling(true);
      const poll = async () => {
        if (!mountedRef.current) return;
        try {
          const status = await getPaymentStatus(sessionId, pid);
          if (!mountedRef.current) return;
          if (status.status === "paid") {
            setPolling(false);
            onPaymentSuccess();
            return;
          }
          if (status.status === "expired") {
            setError("Payment expired — please try again");
            setPolling(false);
            return;
          }
          // Still pending, poll again in 3s
          pollTimerRef.current = setTimeout(poll, 3000);
        } catch {
          if (!mountedRef.current) return;
          pollTimerRef.current = setTimeout(poll, 3000);
        }
      };
      poll();
    },
    [onPaymentSuccess, sessionId]
  );

  // Cleanup polling on unmount — cancel the actual timer
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, []);

  const handlePay = async () => {
    setLoading(true);
    setError(null);
    // Open a blank tab synchronously (inside the click gesture) so the popup
    // isn't blocked — we point it at the Paddle checkout URL once the backend
    // has minted the order. If the browser still blocks the popup, fall back
    // to a same-tab redirect.
    const checkoutTab =
      typeof window !== "undefined" ? window.open("", "_blank") : null;
    try {
      const payment = await createPayment(sessionId, selected);
      setPaymentId(payment.payment_id);
      if (payment.checkout_url) {
        if (checkoutTab) {
          checkoutTab.location.href = payment.checkout_url;
        } else {
          // Popup blocked — redirect this tab to the hosted checkout instead.
          window.location.href = payment.checkout_url;
        }
      } else if (checkoutTab) {
        // No checkout URL (dev mock mode auto-confirms server-side) — close the
        // blank tab we opened; the poll below still resolves to "paid".
        checkoutTab.close();
      }
      pollStatus(payment.payment_id);
    } catch (e) {
      if (checkoutTab) checkoutTab.close();
      setError(e instanceof Error ? e.message : "Couldn't create payment");
    } finally {
      setLoading(false);
    }
  };

  // Price is stored in cents and rendered in USD (overseas pricing is locked at
  // $5 Standard / $10 Pro). Drop trailing cents for whole-dollar prices.
  const formatPrice = (cents: number) => {
    const dollars = cents / 100;
    return dollars % 1 === 0 ? `$${dollars}` : `$${dollars.toFixed(2)}`;
  };

  const selectedTierInfo = tiers.find((t) => t.tier === selected);

  return (
    <div
      className="fixed inset-0 z-50 bg-stone-900/70 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl max-w-md w-full overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-stone-100">
          <div>
            <h3 className="text-base font-semibold text-stone-900">
              Upgrade your portrait package
            </h3>
            <p className="text-xs text-stone-400 mt-0.5">
              Unlock more styles, HD downloads, and post-processing tools
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full bg-stone-100 flex items-center justify-center hover:bg-stone-200 transition-colors"
          >
            <X size={14} className="text-stone-500" />
          </button>
        </div>

        {/* Current tier badge */}
        <div className="px-5 pt-4">
          <p className="text-xs text-stone-400 mb-2">Current tier</p>
          <span
            className={`inline-block px-3 py-1 rounded-full text-xs font-medium ${TIER_BADGE[currentTier].accent}`}
          >
            {TIER_BADGE[currentTier].label}
          </span>
        </div>

        {/* Tier selection */}
        <div className="px-5 py-4 space-y-3">
          {tiers
            .filter((t) => t.price_cents > 0)
            .map((t) => {
              const isSelected = selected === t.tier;
              return (
                <button
                  key={t.tier}
                  onClick={() => setSelected(t.tier)}
                  className={`w-full text-left p-4 rounded-xl border-2 transition-colors ${
                    isSelected
                      ? "border-accent bg-accent/5"
                      : "border-stone-200 hover:border-stone-300"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm font-semibold text-stone-900">
                        {t.label}
                      </span>
                      <span className="ml-2 text-lg font-bold text-accent">
                        {formatPrice(t.price_cents)}
                      </span>
                    </div>
                    {isSelected && (
                      <div className="w-5 h-5 rounded-full bg-accent flex items-center justify-center">
                        <Check size={12} className="text-white" />
                      </div>
                    )}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {t.allow_id_photo && (
                      <FeatureBadge>ID photo crop</FeatureBadge>
                    )}
                    {t.allow_bg_replace && (
                      <FeatureBadge>Background swap</FeatureBadge>
                    )}
                    {t.allow_hd_download && (
                      <FeatureBadge>HD download</FeatureBadge>
                    )}
                    <FeatureBadge>{t.max_revisions} revisions</FeatureBadge>
                    <FeatureBadge>{t.max_styles} styles</FeatureBadge>
                  </div>
                </button>
              );
            })}
        </div>

        {/* Error */}
        {error && (
          <div className="px-5">
            <p className="text-xs text-red-600 bg-red-50 px-3 py-2 rounded-lg">
              {error === "Couldn't load pricing" ? "Couldn't load pricing" : error === "Payment expired — please try again" ? "Payment expired — please try again" : error === "Couldn't create payment" ? "Couldn't create payment" : error}
            </p>
          </div>
        )}

        {/* Action area */}
        {/* Paddle checkout flow (#65): clicking Pay opens Paddle's hosted
            checkout (createPayment returns checkout_url) in a new tab, then we
            poll getPaymentStatus until the HMAC-signed Paddle webhook marks the
            session paid. In mock mode (PAYMENT_MOCK_ENABLED=1) there is no
            checkout_url and the backend auto-confirms after ~5s, so the poll
            still resolves. The tier upgrade happens ONLY server-side via the
            verified webhook — this client never grants premium itself. */}
        <div className="px-5 pb-5">
          {!paymentId ? (
            <button
              onClick={handlePay}
              disabled={loading || !selectedTierInfo}
              className="w-full h-11 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Creating order…
                </>
              ) : (
                <>
                  Pay {selectedTierInfo ? formatPrice(selectedTierInfo.price_cents) : ""}
                </>
              )}
            </button>
          ) : (
            <div className="text-center">
              {polling ? (
                <div className="py-4">
                  <Loader2
                    size={24}
                    className="animate-spin text-accent mx-auto mb-2"
                  />
                  <p className="text-sm text-stone-600">
                    Waiting for payment confirmation…
                  </p>
                  <p className="text-xs text-stone-400 mt-1">
                    Please complete payment in the checkout tab
                  </p>
                </div>
              ) : (
                <p className="text-sm text-stone-600 py-4">Payment complete</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FeatureBadge({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block px-2 py-0.5 rounded-md bg-stone-100 text-stone-600 text-xs">
      {children}
    </span>
  );
}
