"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";

const ITEMS = [
  {
    q: "Will the portraits really look like me?",
    a: "We use an image model built specifically for facial consistency, and an in-house likeness verification system scores every portrait. If a shot doesn't look like you, the AI automatically re-renders it until it passes the likeness check.",
  },
  {
    q: "How is this different from a photo studio?",
    a: "No booking, no makeup, no waiting on retouching. Just upload a few selfies and get multiple art portraits across different styles in about ten minutes &mdash; studio-quality results at a fraction of the price.",
  },
  {
    q: "What kind of photos should I upload?",
    a: "Everyday phone selfies work great. We recommend uploading 3&ndash;8 photos with clear lighting and varied angles; ideally a mix of front and side views. No need to pose or apply makeup.",
  },
  {
    q: "How long does generation take?",
    a: "A single portrait takes about 30&ndash;60 seconds, and a full collection usually finishes within a few minutes. Same-day delivery is included with every package.",
  },
  {
    q: "What portrait themes are available?",
    a: "Traditional Chinese (Hanfu, Qipao), Hong Kong retro, French, Japanese, Korean, cinematic, editorial, natural, and cyberpunk &mdash; 9 themes and 25+ styles. Mix and match freely, just like choosing a studio package.",
  },
  {
    q: "Can I get revisions if I don't like a shot?",
    a: "The Standard package includes 2 free revisions, and the Pro package includes 3. You can adjust expression, background, outfit, and overall mood. Still not happy? We'll regenerate it.",
  },
  {
    q: "Are my photos safe?",
    a: "Your photos are used only to generate your portraits. Transfer and storage are encrypted, they're never used to train models, and never shown publicly. They're auto-deleted 7 days after delivery, and you can request immediate deletion at any time.",
  },
];

export function FAQ() {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  return (
    <section id="faq" className="py-24 md:py-32">
      <div className="mx-auto max-w-3xl px-6">
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-center">
          Frequently asked questions
        </h2>

        <div className="mt-14 divide-y divide-stone-200">
          {ITEMS.map((item, i) => {
            const isOpen = openIndex === i;
            return (
              <div key={i}>
                <button
                  onClick={() => setOpenIndex(isOpen ? null : i)}
                  className="w-full flex items-center justify-between py-5 text-left"
                >
                  <span className="font-medium text-stone-900 pr-4">
                    {item.q}
                  </span>
                  <ChevronDown
                    size={20}
                    className={`shrink-0 text-stone-400 transition-transform duration-300 ${
                      isOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>
                <div
                  className={`overflow-hidden transition-all duration-300 ${
                    isOpen ? "max-h-60 pb-5" : "max-h-0"
                  }`}
                >
                  <p className="text-stone-500 text-sm leading-relaxed">
                    {item.a}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
