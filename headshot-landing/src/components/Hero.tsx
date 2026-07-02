import Image from "next/image";
import { ArrowRight } from "lucide-react";

export function Hero() {
  return (
    <section className="relative min-h-[100dvh] flex items-center pt-20">
      <div className="mx-auto max-w-7xl px-6 w-full">
        <div className="grid md:grid-cols-2 gap-12 md:gap-16 items-center">
          {/* Left: Copy */}
          <div className="max-w-lg">
            <p className="text-accent font-medium text-sm tracking-wide mb-4">
              FlashShot &middot; AI Portrait Studio
            </p>
            <h1 className="text-4xl md:text-5xl lg:text-[3.5rem] font-semibold tracking-tight leading-[1.1] text-stone-900">
              AI portraits that
              <br />
              actually look like you
            </h1>
            <p className="mt-5 text-lg text-stone-500 leading-relaxed max-w-[55ch]">
              Upload a few selfies and get a gallery of AI portraits that genuinely resemble you &mdash; Hanfu, Hong Kong retro, French, Japanese, cinematic, and more. Skip the photo studio and own your portrait collection. Every shot is verified for likeness so it really looks like you.
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <a
                href="/create"
                className="inline-flex items-center gap-2 h-12 px-7 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors"
              >
                Create your portraits
                <ArrowRight size={16} />
              </a>
              <a
                href="#gallery"
                className="inline-flex items-center h-12 px-7 rounded-full border border-stone-300 text-stone-700 font-medium text-sm hover:border-stone-400 transition-colors"
              >
                See samples
              </a>
            </div>
          </div>

          {/* Right: Before / After comparison */}
          <div className="relative">
            <div className="grid grid-cols-2 gap-3">
              {/* "Before" - casual style */}
              <div className="relative aspect-[3/4] rounded-2xl overflow-hidden">
                <Image
                  src="/images/lw_f_01.png"
                  alt="Everyday selfie"
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 50vw, 25vw"
                />
                <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-sm rounded-full px-3 py-1 text-xs font-medium text-stone-600">
                  Your selfie
                </div>
              </div>

              {/* "After" - AI portrait */}
              <div className="relative aspect-[3/4] rounded-2xl overflow-hidden shadow-2xl shadow-stone-900/10 ring-1 ring-stone-200/50">
                <Image
                  src="/images/gf_f_qipao.png"
                  alt="AI-generated Qipao portrait"
                  fill
                  className="object-cover"
                  sizes="(max-width: 768px) 50vw, 25vw"
                  priority
                />
                <div className="absolute bottom-3 left-3 bg-accent/90 backdrop-blur-sm rounded-full px-3 py-1 text-xs font-medium text-white">
                  AI portrait
                </div>
              </div>
            </div>

            {/* Decorative offset card behind */}
            <div className="absolute -z-10 top-6 -right-4 w-full h-full rounded-2xl bg-stone-200/40" />
          </div>
        </div>
      </div>
    </section>
  );
}
