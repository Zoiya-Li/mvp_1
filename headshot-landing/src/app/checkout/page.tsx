"use client";

import { useCallback, useEffect, useState } from "react";
import Script from "next/script";
import { CheckCircle2, LoaderCircle, ShieldCheck } from "lucide-react";

import { getPublicConfig } from "@/lib/api";

type PaddleEvent = { name?: string };
type PaddleSdk = {
  Environment: { set(environment: "sandbox"): void };
  Initialize(options: {
    token: string;
    checkout: { settings: Record<string, string> };
    eventCallback(event: PaddleEvent): void;
  }): void;
};

declare global {
  interface Window {
    Paddle?: PaddleSdk;
  }
}

export default function CheckoutPage() {
  const [scriptReady, setScriptReady] = useState(false);
  const [status, setStatus] = useState<"loading" | "ready" | "paid" | "error">(
    "loading"
  );
  const [message, setMessage] = useState("Preparing secure checkout...");

  const initialize = useCallback(async () => {
    if (!scriptReady || !window.Paddle) return;
    try {
      const config = await getPublicConfig();
      if (!config.paddle_client_token) {
        throw new Error("Checkout is not configured yet.");
      }
      if (config.paddle_environment === "sandbox") {
        window.Paddle.Environment.set("sandbox");
      }
      window.Paddle.Initialize({
        token: config.paddle_client_token,
        checkout: {
          settings: {
            displayMode: "overlay",
            theme: "light",
            locale: "en",
            successUrl: `${window.location.origin}/create?payment=success`,
          },
        },
        eventCallback(event) {
          if (event.name === "checkout.loaded") {
            setStatus("ready");
            setMessage("Secure checkout opened.");
          }
          if (event.name === "checkout.completed") {
            setStatus("paid");
            setMessage("Payment received. You can return to FlashShot.");
          }
          if (event.name === "checkout.closed") {
            setStatus("ready");
            setMessage("Checkout closed. Return to the previous tab to continue.");
          }
        },
      });
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Checkout failed to load.");
    }
  }, [scriptReady]);

  useEffect(() => {
    void initialize();
  }, [initialize]);

  return (
    <main className="min-h-screen bg-white text-stone-900 flex items-center justify-center px-6 py-12">
      <section className="w-full max-w-lg text-center" aria-live="polite">
        <div className="mx-auto mb-6 flex size-12 items-center justify-center rounded-full bg-emerald-50 text-emerald-700">
          {status === "paid" ? (
            <CheckCircle2 className="size-6" aria-hidden="true" />
          ) : status === "loading" ? (
            <LoaderCircle className="size-6 animate-spin" aria-hidden="true" />
          ) : (
            <ShieldCheck className="size-6" aria-hidden="true" />
          )}
        </div>
        <h1 className="text-2xl font-semibold">FlashShot secure checkout</h1>
        <p className="mt-3 text-sm leading-6 text-stone-500">{message}</p>
        <a
          href="/create"
          className="mt-8 inline-flex h-10 items-center justify-center border border-stone-300 px-4 text-sm font-medium text-stone-700 hover:border-stone-500"
        >
          Return to your portrait
        </a>
      </section>
      <Script
        src="https://cdn.paddle.com/paddle/v2/paddle.js"
        strategy="afterInteractive"
        onLoad={() => setScriptReady(true)}
        onError={() => {
          setStatus("error");
          setMessage("Secure checkout could not be loaded.");
        }}
      />
    </main>
  );
}
