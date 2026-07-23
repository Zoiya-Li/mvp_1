"use client";

import Link from "next/link";
import { ChevronRight, LockKeyhole, ReceiptText, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import { getEntitlementBalance, listPortraitProjects } from "@/lib/portrait-v2";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

export function AccountStudio() {
  const [balance, setBalance] = useState<number | null>(null);
  const [projectCount, setProjectCount] = useState<number | null>(null);

  useEffect(() => {
    Promise.all([getEntitlementBalance(), listPortraitProjects()])
      .then(([credits, projects]) => { setBalance(credits); setProjectCount(projects.length); })
      .catch(() => { setBalance(0); setProjectCount(0); });
  }, []);

  return (
    <div className="simple-portal-page">
      <PortalHeader />
      <main className="account-page">
        <p className="step-count">Your private studio</p>
        <h1>You</h1>
        <section className="account-balance"><Sparkles size={20} /><div><strong>{balance === null ? "Checking balance" : `${balance} preview credit${balance === 1 ? "" : "s"}`}</strong><span>Credits are only used when generation starts</span></div></section>
        <nav className="account-links">
          <Link href="/library"><span><ReceiptText size={19} /> Projects and orders</span><small>{projectCount ?? "–"}</small><ChevronRight size={18} /></Link>
          <Link href="/privacy"><span><LockKeyhole size={19} /> Privacy and deletion</span><ChevronRight size={18} /></Link>
        </nav>
      </main>
      <PortalBottomNav />
    </div>
  );
}
