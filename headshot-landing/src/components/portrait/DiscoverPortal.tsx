"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowRight, Camera, Check, ShieldCheck, Sparkles, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  FALLBACK_THEMES,
  getThemes,
  shouldBypassImageOptimization,
  type PortraitTheme,
} from "@/lib/portrait-v2";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

function ThemeRail({ title, themes }: { title: string; themes: PortraitTheme[] }) {
  return (
    <section className="theme-section">
      <div className="section-heading">
        <h2>{title}</h2>
        <span>{themes.length} shoots</span>
      </div>
      <div className="theme-rail">
        {themes.map((theme, index) => (
          <Link href={`/themes/${theme.slug}`} className="theme-tile" key={theme.theme_id}>
            <div className="theme-tile-image">
              <Image
                src={theme.cover_image}
                alt={`${theme.title_en} portrait example`}
                fill
                sizes="(max-width: 720px) 72vw, 320px"
                priority={index < 2}
                unoptimized={shouldBypassImageOptimization(theme.cover_image)}
              />
            </div>
            <div className="theme-tile-copy">
              <div>
                <p>{theme.category}</p>
                <h3>{theme.title_en}</h3>
              </div>
              <ArrowRight aria-hidden="true" size={18} />
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}

export function DiscoverPortal() {
  const [themes, setThemes] = useState(FALLBACK_THEMES);

  useEffect(() => {
    getThemes().then((liveThemes) => {
      if (liveThemes.length > 0) setThemes(liveThemes);
    }).catch(() => {
      // The curated fallback keeps the discovery portal useful during a brief
      // API restart; the live catalog replaces it as soon as it is available.
    });
  }, []);

  const featured = useMemo(
    () => themes.find((theme) => theme.featured) ?? themes[0],
    [themes],
  );
  const stories = themes.filter((theme) => theme.theme_id !== featured.theme_id);

  return (
    <div className="portal-page">
      <section className="featured-shoot">
        <Image
          src={featured.cover_image}
          alt={`${featured.title_en} featured portrait`}
          fill
          priority
          sizes="100vw"
          unoptimized={shouldBypassImageOptimization(featured.cover_image)}
        />
        <div className="featured-shade" />
        <PortalHeader light />
        <div className="featured-copy">
          <p className="featured-kicker">This week&apos;s story</p>
          <h1>{featured.title_en}</h1>
          <p>{featured.tagline}</p>
          <Link href={`/themes/${featured.slug}`} className="primary-light-button">
            Shoot this look
            <ArrowRight aria-hidden="true" size={17} />
          </Link>
        </div>
        <a href="#inspiration" className="next-section-cue">
          Or bring your own reference
        </a>
      </section>

      <main className="portal-main">
        <section id="inspiration" className="inspiration-band">
          <div className="inspiration-copy">
            <div className="eyebrow"><Sparkles size={15} /> Make it yours</div>
            <h2>Already found a photo you love?</h2>
            <p>
              Bring the scene, light, and mood. We keep the reference private and
              rebuild the shoot around your identity.
            </p>
            <Link href="/inspiration" className="dark-command-button">
              <Upload aria-hidden="true" size={17} />
              Upload a reference
            </Link>
          </div>
          <div className="inspiration-visual" aria-hidden="true">
            <div className="inspiration-photo inspiration-photo-back">
              <Image src="/images/social_f_french.png" alt="" fill sizes="240px" />
            </div>
            <div className="inspiration-photo inspiration-photo-front">
              <Image src="/images/film_f_cinematic.png" alt="" fill sizes="240px" />
            </div>
            <div className="inspiration-arrow"><ArrowRight size={22} /></div>
          </div>
        </section>

        <ThemeRail title="Stories worth stepping into" themes={stories} />

        <section className="studio-promise">
          <div className="studio-promise-image">
            <Image
              src="/images/kr_m_minimal.png"
              alt="Natural Korean studio portrait"
              fill
              sizes="(max-width: 800px) 100vw, 45vw"
            />
          </div>
          <div className="studio-promise-copy">
            <div className="eyebrow"><Camera size={15} /> A complete shoot</div>
            <h2>Not a filter. A portrait story built around you.</h2>
            <ul>
              <li><Check size={17} /> A close-up, half-body, and environmental set</li>
              <li><Check size={17} /> Every final checked for likeness and artifacts</li>
              <li><Check size={17} /> One hero preview before you unlock the set</li>
            </ul>
          </div>
        </section>

        <section className="privacy-strip">
          <ShieldCheck aria-hidden="true" size={23} />
          <div>
            <h2>Your face stays yours.</h2>
            <p>Private by default. Never added to a public gallery. Delete everything in one action.</p>
          </div>
          <Link href="/privacy">Our privacy promise <ArrowRight size={15} /></Link>
        </section>
      </main>

      <footer className="portal-footer">
        <span>FlashShot</span>
        <p>Your private AI portrait studio.</p>
        <div><Link href="/privacy">Privacy</Link><Link href="/terms">Terms</Link></div>
      </footer>
      <PortalBottomNav />
    </div>
  );
}
