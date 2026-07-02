import { Upload, Sparkles, Download } from "lucide-react";

const STEPS = [
  {
    icon: Upload,
    title: "Upload selfies",
    body: "Upload 3–8 everyday photos — quick phone snaps are fine. We automatically learn your facial features from them.",
  },
  {
    icon: Sparkles,
    title: "Pick a portrait theme",
    body: "Choose from Chinese style, Hong Kong style, French, Japanese, Korean, cinematic and more. AI generates multiple artistic portraits that look like you for each theme.",
  },
  {
    icon: Download,
    title: "Download your portraits",
    body: "Download HD portraits, with free revisions if you're not satisfied. No studio appointment needed — get your portrait collection in ten minutes.",
  },
];

export function Workflow() {
  return (
    <section id="workflow" className="py-24 md:py-32">
      <div className="mx-auto max-w-7xl px-6">
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-center">
          Three steps to your portrait collection
        </h2>
        <p className="mt-3 text-stone-500 text-center max-w-md mx-auto">
          No photo studio, no makeup, no waiting for retouching. Get your own artistic portraits in ten minutes.
        </p>

        <div className="mt-16 grid md:grid-cols-3 gap-8 md:gap-12">
          {STEPS.map((step, i) => (
            <div key={step.title} className="relative text-center md:text-left">
              {/* Connector line (desktop only) */}
              {i < STEPS.length - 1 && (
                <div className="hidden md:block absolute top-7 left-[calc(50%+2rem)] w-[calc(100%-4rem)] h-px border-t border-dashed border-stone-300" />
              )}

              <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-accent-light text-accent mb-5">
                <step.icon size={26} />
              </div>

              <h3 className="text-lg font-semibold">{step.title}</h3>
              <p className="mt-2 text-stone-500 leading-relaxed text-sm">
                {step.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
