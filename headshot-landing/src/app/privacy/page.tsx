"use client";

import Link from "next/link";
import { ArrowLeft, Check, LockKeyhole, Trash2 } from "lucide-react";
import { useState } from "react";

import { PortalBottomNav, PortalHeader } from "@/components/portrait/PortalNav";
import { deletePortraitWorkspace } from "@/lib/portrait-v2";

export default function PrivacyPage() {
  const [confirming, setConfirming] = useState(false);
  const [confirmation, setConfirmation] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  async function deleteWorkspace() {
    if (confirmation !== "DELETE") return;
    try {
      await deletePortraitWorkspace();
      setStatus("Your workspace and portrait media have been deleted from FlashShot.");
      setConfirming(false);
    } catch (cause) {
      setStatus(cause instanceof Error ? cause.message : "Could not delete this workspace");
    }
  }

  return (
    <div className="privacy-page">
      <PortalHeader />
      <main className="privacy-main">
        <Link href="/account" className="inline-back"><ArrowLeft size={18} /> You</Link>
        <header className="privacy-heading"><p className="step-count">Privacy by design</p><h1>Your face is not our dataset.</h1><p>FlashShot uses your photos to make the portrait project you requested. Private references do not become public templates.</p></header>
        <section className="privacy-principles">
          <div><LockKeyhole size={20} /><h2>Task-local identity</h2><p>Face analysis is scoped to your project. We prohibit cross-user face search and a long-term facial recognition library.</p></div>
          <div><Check size={20} /><h2>No training by default</h2><p>Your identity photos, inspiration references, and generated portraits are not used to train models or in marketing without a separate explicit opt-in.</p></div>
        </section>
        <section className="policy-sections">
          <div><h2>What we process</h2><p>Identity photos, private inspiration images, generated portraits, project settings, quality results, essential device/session data, Apple account identifiers, and order status. Apple processes iOS in-app purchases; Paddle may process purchases made directly on the website. FlashShot does not store full card numbers.</p></div>
          <div><h2>Retention</h2><p>Identity and inspiration source images expire after 7 days. Generated portrait pixels remain available for up to 30 days. Minimum operational metadata is removed after 90 days. Payment records are pseudonymized and retained where tax, accounting, chargeback, or fraud law requires it.</p></div>
          <div><h2>Service providers</h2><p>Images may be sent to the configured generation and quality providers, currently SiliconFlow or OpenRouter, solely to perform your request. Hosting infrastructure stores encrypted transport data and private project files. Apple provides iOS identity and in-app purchase services; Paddle may provide website checkout services.</p></div>
          <div><h2>Your controls</h2><p>You can delete an individual project from Library at any time. Deleting a project removes its identity photos, private inspiration, and generated pixels. Deleting the workspace removes all projects and invalidates the guest access token.</p></div>
          <div><h2>Eligibility and contact</h2><p>FlashShot currently accepts portraits of consenting adults aged 18 or older. Privacy requests can be sent to <a href="mailto:support@flashshot.top">support@flashshot.top</a>.</p></div>
        </section>
        <section className="delete-workspace">
          <div><Trash2 size={20} /><div><h2>Delete this workspace</h2><p>Permanent. Financial records required by law are retained without your active workspace identifier.</p></div></div>
          {!confirming ? <button onClick={() => setConfirming(true)}>Delete workspace</button> : <div className="delete-confirm"><label>Type DELETE to confirm<input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label><button disabled={confirmation !== "DELETE"} onClick={deleteWorkspace}>Permanently delete</button><button onClick={() => setConfirming(false)}>Cancel</button></div>}
          {status && <p className="delete-status">{status}</p>}
        </section>
        <footer className="policy-footer"><span>Last updated July 14, 2026</span><Link href="/terms">Terms of Service</Link></footer>
      </main>
      <PortalBottomNav />
    </div>
  );
}
