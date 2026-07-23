"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, ArrowRight, Check, Images, LockKeyhole } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  FALLBACK_THEMES,
  getTheme,
  shouldBypassImageOptimization,
  type PortraitTheme,
} from "@/lib/portrait-v2";
import { PortalHeader } from "./PortalNav";

export function ThemeExperience({ slug }: { slug: string }) {
  const fallback = useMemo(
    () => FALLBACK_THEMES.find((item) => item.slug === slug) ?? FALLBACK_THEMES[0],
    [slug],
  );
  const [theme, setTheme] = useState<PortraitTheme>(fallback);

  useEffect(() => {
    getTheme(slug).then(setTheme).catch(() => {});
  }, [slug]);

  const previews = theme.preview_images.length ? theme.preview_images : [theme.cover_image];

  return (
    <div className="theme-page">
      <section className="theme-hero">
        <Image
          src={theme.cover_image}
          alt={`${theme.title_en} portrait`}
          fill
          priority
          sizes="100vw"
          unoptimized={shouldBypassImageOptimization(theme.cover_image)}
        />
        <div className="featured-shade" />
        <PortalHeader light />
        <Link href="/" className="theme-back" aria-label="Back to discover">
          <ArrowLeft size={19} />
        </Link>
        <div className="theme-hero-copy">
          <p>{theme.category}</p>
          <h1>{theme.title_en}</h1>
          <span>{theme.tagline}</span>
          <Link href={`/create?theme=${encodeURIComponent(theme.slug)}`} className="theme-hero-action">
            Shoot this look <ArrowRight size={17} />
          </Link>
        </div>
      </section>

      <main className="theme-content">
        <section className="theme-intro">
          <div>
            <div className="eyebrow"><Images size={15} /> Complete portrait set</div>
            <h2>A story with room to breathe.</h2>
          </div>
          <p>
            One clear hero portrait, varied framing, and a consistent visual world.
            Every delivered image must pass identity and quality review.
          </p>
        </section>

        <section className="contact-sheet" aria-label="Theme preview gallery">
          {previews.slice(0, 6).map((image, index) => (
            <div className={`contact-shot contact-shot-${index + 1}`} key={`${image}-${index}`}>
              <Image
                src={image}
                alt={`${theme.title_en} example ${index + 1}`}
                fill
                sizes="50vw"
                unoptimized={shouldBypassImageOptimization(image)}
              />
            </div>
          ))}
        </section>

        <section className="theme-facts">
          <div><strong>{theme.shot_count ?? 6}</strong><span>finished portraits</span></div>
          <div><strong>{theme.reference_min ?? 4}–{theme.reference_max ?? 6}</strong><span>reference photos</span></div>
          <div><strong>1</strong><span>hero preview first</span></div>
        </section>

        <section className="theme-assurance">
          <h2>Before the full set, you see whether it feels like you.</h2>
          <div>
            <p><Check size={16} /> Failed generations never use your credit</p>
            <p><Check size={16} /> One guided remake is included</p>
            <p><LockKeyhole size={16} /> Source photos stay private</p>
          </div>
        </section>
      </main>

      <div className="theme-sticky-action">
        <div><span>{theme.title_en}</span><small>Preview before purchase</small></div>
        <Link href={`/create?theme=${encodeURIComponent(theme.slug)}`}>
          Shoot this look <ArrowRight size={17} />
        </Link>
      </div>
    </div>
  );
}
