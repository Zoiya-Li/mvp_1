"use client";

import { useEffect, useState } from "react";
import { getPublicConfig } from "@/lib/api";

/**
 * Site footer.
 *
 * - Optionally fetches a mainland-China ICP filing number from the backend
 *   (Chinese-hosted sites are legally required to show it, linking to
 *   beian.miit.gov.cn). For the overseas deployment this stays empty and the
 *   line is hidden. Only rendered when a number is configured.
 * - Includes Privacy / Terms / Contact links required for a service that
 *   collects face photos and processes payments.
 */
export function Footer() {
  const [icp, setIcp] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    getPublicConfig()
      .then((cfg) => {
        if (!cancelled) setIcp(cfg.icp_beian ?? "");
      })
      .catch(() => {
        /* config endpoint unreachable (e.g. API down) — footer still renders */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <footer className="bg-stone-900 border-t border-stone-800 py-10">
      <div className="mx-auto max-w-7xl px-6 flex flex-col items-center gap-4 text-sm text-stone-500">
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
          <a href="#pricing" className="hover:text-stone-300 transition-colors">
            Pricing
          </a>
          <a href="#faq" className="hover:text-stone-300 transition-colors">
            FAQ
          </a>
          <a href="/privacy" className="hover:text-stone-300 transition-colors">
            Privacy
          </a>
          <a href="/terms" className="hover:text-stone-300 transition-colors">
            Terms
          </a>
          <a href="#cta" className="hover:text-stone-300 transition-colors">
            Contact
          </a>
        </div>

        <p>FlashShot — AI Portrait Studio · Artistic portraits that look like you</p>

        <p className="text-stone-600">
          © {new Date().getFullYear()} FlashShot
        </p>

        {/* ICP filing — only relevant for sites hosted/served in mainland China.
            Stays empty (hidden) for the overseas deployment. */}
        {icp && (
          <a
            href="https://beian.miit.gov.cn"
            target="_blank"
            rel="noopener noreferrer"
            className="text-stone-600 hover:text-stone-400 transition-colors text-xs"
          >
            {icp}
          </a>
        )}
      </div>
    </footer>
  );
}
