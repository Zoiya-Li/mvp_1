"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ArrowLeft, Check, LoaderCircle, LockKeyhole, ShieldCheck, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import {
  createPortraitOrder, getCheckoutAvailability, getPortraitOrder, getPortraitProject,
  PortraitOrder, PortraitProject, unlockPortraitSet,
} from "@/lib/portrait-v2";
import { PortalHeader } from "./PortalNav";

type ProductCode = PortraitOrder["product_code"];
type CheckoutPhase = "loading" | "offer" | "unavailable" | "redirecting" | "verifying" | "paid" | "error";

const PRODUCTS: Record<ProductCode, { label: string; price: string; detail: string }> = {
  portrait_set: { label: "Complete Shoot", price: "$5", detail: "6 finished portraits · standard resolution" },
  portrait_set_hd: { label: "Complete Shoot HD", price: "$10", detail: "6 finished portraits · full-resolution downloads" },
};

export function PortraitCheckout() {
  const params = useSearchParams();
  const projectId = params.get("project");
  const returnedOrder = params.get("order");
  const [project, setProject] = useState<PortraitProject | null>(null);
  const [product, setProduct] = useState<ProductCode>("portrait_set_hd");
  const [phase, setPhase] = useState<CheckoutPhase>(
    !projectId ? "error" : returnedOrder ? "verifying" : "loading",
  );
  const [error, setError] = useState<string | null>(
    projectId ? null : "This checkout is not connected to a portrait project.",
  );

  useEffect(() => {
    if (!projectId) return;
    Promise.all([getPortraitProject(projectId), getCheckoutAvailability()])
      .then(([value, checkoutAvailable]) => {
        setProject(value);
        if (!returnedOrder) setPhase(checkoutAvailable ? "offer" : "unavailable");
      })
      .catch((cause) => { setError(cause instanceof Error ? cause.message : "Could not open checkout"); setPhase("error"); });
  }, [projectId, returnedOrder]);

  useEffect(() => {
    if (!projectId || !returnedOrder || !project) return;
    let cancelled = false;
    let attempts = 0;
    const verify = async () => {
      try {
        const order = await getPortraitOrder(projectId, returnedOrder);
        if (cancelled) return;
        if (order.status === "paid") {
          await unlockPortraitSet(projectId);
          setPhase("paid");
          return;
        }
        if (order.status === "expired" || order.status === "refunded") {
          setError(`This order is ${order.status}.`); setPhase("error"); return;
        }
        attempts += 1;
        if (attempts < 30) window.setTimeout(verify, 2000);
        else { setError("Payment is still being confirmed. Your order remains safe in the library."); setPhase("error"); }
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Could not verify payment"); setPhase("error");
      }
    };
    void verify();
    return () => { cancelled = true; };
  }, [project, projectId, returnedOrder]);

  async function checkout() {
    if (!projectId) return;
    setPhase("redirecting"); setError(null);
    try {
      const order = await createPortraitOrder(projectId, product);
      if (order.checkout_url) window.location.assign(order.checkout_url);
      else {
        setError("Checkout is running in local test mode. Payment confirmation will appear in your library.");
        setPhase("verifying");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not prepare checkout");
      setPhase("offer");
    }
  }

  return (
    <div className="checkout-page">
      <PortalHeader />
      <main className="portrait-checkout">
        <Link href={projectId ? `/create?project=${projectId}` : "/library"} className="inline-back"><ArrowLeft size={18} /> First look</Link>
        {(phase === "loading" || phase === "verifying" || phase === "redirecting") && (
          <section className="checkout-state"><LoaderCircle className="spin" size={28} /><h1>{phase === "verifying" ? "Confirming your payment" : phase === "redirecting" ? "Opening secure checkout" : "Preparing your shoot"}</h1><p>Keep this page open. Entitlements are granted only after the signed payment confirmation arrives.</p></section>
        )}
        {phase === "offer" && project && (
          <section className="checkout-offer">
            <div className="checkout-copy">
              <p className="step-count">Your complete photo story</p>
              <h1>Keep the feeling. Build the set.</h1>
              <p>Your first portrait proved the direction. The complete shoot turns it into six coordinated images with the same identity and visual language.</p>
              <div className="checkout-includes"><span><Check size={16} /> 6 final-gated portraits</span><span><Check size={16} /> One guided remake</span><span><Check size={16} /> Private library delivery</span></div>
            </div>
            <div className="checkout-panel">
              {(Object.keys(PRODUCTS) as ProductCode[]).map((code) => (
                <button key={code} className={product === code ? "active" : ""} onClick={() => setProduct(code)}>
                  <span><strong>{PRODUCTS[code].label}</strong><small>{PRODUCTS[code].detail}</small></span><b>{PRODUCTS[code].price}</b>
                </button>
              ))}
              {error && <p className="form-error">{error}</p>}
              <button className="checkout-pay" onClick={checkout}><LockKeyhole size={17} /> Continue to secure checkout</button>
              <p className="checkout-trust"><ShieldCheck size={15} /> Tax, card processing, Apple Pay, and Google Pay availability are handled by Paddle.</p>
            </div>
          </section>
        )}
        {phase === "paid" && (
          <section className="checkout-state"><div className="paid-mark"><Sparkles size={24} /></div><h1>Your complete shoot is underway.</h1><p>The six-image set will appear in your private library as each portrait clears final QA.</p><Link href={`/create?project=${projectId}`} className="dark-command-button">Watch the studio</Link></section>
        )}
        {phase === "unavailable" && (
          <section className="checkout-state checkout-unavailable">
            <ShieldCheck size={28} />
            <p className="step-count">Purchases paused</p>
            <h1>Your first look stays yours.</h1>
            <p>Secure checkout is not open yet. Your preview and private project remain in the library, ready for the complete shoot when purchasing becomes available.</p>
            <Link href={projectId ? `/create?project=${projectId}` : "/library"} className="dark-command-button">Return to your project</Link>
          </section>
        )}
        {phase === "error" && (
          <section className="checkout-state"><ShieldCheck size={28} /><h1>Your project is safe.</h1><p>{error}</p><Link href={projectId ? `/create?project=${projectId}` : "/library"} className="dark-command-button">Return to your project</Link></section>
        )}
      </main>
    </div>
  );
}
