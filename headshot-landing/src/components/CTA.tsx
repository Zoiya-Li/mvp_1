import { Mail } from "lucide-react";

export function CTA() {
  return (
    <section id="cta" className="py-24 md:py-32 bg-stone-900 text-white">
      <div className="mx-auto max-w-7xl px-6">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">
            A portrait collection shouldn&apos;t cost this much
          </h2>
          <p className="mt-4 text-stone-400 leading-relaxed max-w-md mx-auto">
            Cheaper than a photo studio, more professional than a selfie, more natural than a retouching app. Upload your selfies and start your AI portrait studio today.
          </p>

          <a
            href="/create"
            className="mt-8 inline-flex items-center justify-center h-12 px-8 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors"
          >
            Start your portraits
          </a>

          <div className="mt-10 flex flex-col items-center gap-3">
            <a
              href="mailto:support@flashshot.top"
              className="inline-flex items-center gap-3 bg-white/10 rounded-xl px-5 py-3"
            >
              <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center text-white">
                <Mail size={18} />
              </div>
              <div className="text-left">
                <p className="text-xs text-stone-400">Questions?</p>
                <p className="font-semibold text-white text-sm">
                  support@flashshot.top
                </p>
              </div>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
