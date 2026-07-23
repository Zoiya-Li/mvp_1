"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowRight, LockKeyhole, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

import {
  FALLBACK_THEMES, getSharedRecipe, getTheme, PortraitTheme,
  sharedHeroUrl, SharedRecipe, shouldBypassImageOptimization,
} from "@/lib/portrait-v2";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

export function SharedLook({ token }: { token: string }) {
  const [shared, setShared] = useState<SharedRecipe | null>(null);
  const [theme, setTheme] = useState<PortraitTheme | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const recipe = await getSharedRecipe(token);
        const selected = recipe.theme_id || recipe.theme_slug
          ? await getTheme(recipe.theme_id || recipe.theme_slug || "")
          : FALLBACK_THEMES[0];
        if (!cancelled) { setShared(recipe); setTheme(selected); }
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "This look is unavailable");
      }
    })();
    return () => { cancelled = true; };
  }, [token]);

  if (error) return <div className="shared-look-error"><PortalHeader /><main><h1>This portrait link has closed.</h1><p>{error}</p><Link href="/" className="dark-command-button">Discover portrait stories</Link></main><PortalBottomNav /></div>;
  if (!shared || !theme) return <div className="portal-loading">Opening this portrait look...</div>;

  return (
    <div className="shared-look-page">
      <PortalHeader light />
      <main className="shared-look-hero">
        {shared.portrait_available ? (
          // Public only after the owner explicitly opts in.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={sharedHeroUrl(token)} alt="A shared FlashShot portrait" />
        ) : <Image src={theme.cover_image} alt="" fill priority sizes="100vw" unoptimized={shouldBypassImageOptimization(theme.cover_image)} />}
        <div className="shared-look-shade" />
        <div className="shared-look-copy">
          <p><Sparkles size={15} /> A FlashShot look</p>
          <h1>{shared.title}</h1>
          <span>{shared.portrait_available ? "Someone made this portrait story and shared the visual direction with you." : "A private visual recipe was shared without publishing its creator's portrait."}</span>
          <Link href={`/create?recipe=${encodeURIComponent(token)}`} className="primary-light-button">Shoot this look with me <ArrowRight size={18} /></Link>
          <small><LockKeyhole size={13} /> Their identity photos and original inspiration are never shared.</small>
        </div>
      </main>
      <PortalBottomNav />
    </div>
  );
}
