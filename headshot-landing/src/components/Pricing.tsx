import { Check } from "lucide-react";

const PLANS = [
  {
    name: "Standard",
    price: "5",
    description: "Perfect for your first run — pick up to 2 portrait themes.",
    features: [
      "Upload 4–6 selfies",
      "Choose up to 2 portrait themes",
      "Multiple retouched portraits per theme",
      "2 free revisions",
      "ID photo + background replace included",
      "Same-day delivery",
    ],
    cta: "Choose Standard",
    featured: false,
  },
  {
    name: "Pro",
    price: "10",
    description: "More themes, HD downloads & priority queue.",
    features: [
      "Upload 4–6 selfies",
      "Choose up to 2 portrait themes",
      "Multiple retouched portraits per theme",
      "3 free revisions",
      "ID photo + business headshot included",
      "HD 2× download + priority queue",
    ],
    cta: "Go Pro",
    featured: true,
  },
];

export function Pricing() {
  return (
    <section id="pricing" className="py-24 md:py-32">
      <div className="mx-auto max-w-7xl px-6">
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight text-center">
          Portrait packages, priced clearly
        </h2>
        <p className="mt-3 text-stone-500 text-center max-w-md mx-auto">
          No hidden fees. Upload your selfies, pick your themes, and get your portrait collection. Not happy? Free revisions.
        </p>

        <div className="mt-14 grid md:grid-cols-2 gap-6 items-start max-w-3xl mx-auto">
          {PLANS.map((plan) => (
            <div
              key={plan.name}
              className={`relative rounded-2xl p-8 ${
                plan.featured
                  ? "bg-stone-900 text-white ring-2 ring-accent shadow-xl shadow-stone-900/10"
                  : "bg-white border border-stone-200"
              }`}
            >
              {plan.featured && (
                <span className="absolute -top-3 left-8 bg-accent text-white text-xs font-semibold px-3 py-1 rounded-full">
                  Best value
                </span>
              )}

              <h3 className="text-lg font-semibold">{plan.name}</h3>
              <p
                className={`mt-1 text-sm ${
                  plan.featured ? "text-stone-400" : "text-stone-500"
                }`}
              >
                {plan.description}
              </p>

              <div className="mt-6 flex items-baseline gap-1">
                <span className="text-4xl font-bold tracking-tight">
                  ${plan.price}
                </span>
                <span
                  className={`text-sm ${
                    plan.featured ? "text-stone-400" : "text-stone-500"
                  }`}
                >
                  one-time
                </span>
              </div>

              <ul className="mt-6 space-y-3">
                {plan.features.map((feat) => (
                  <li key={feat} className="flex items-start gap-2 text-sm">
                    <Check
                      size={18}
                      className="text-accent mt-0.5 shrink-0"
                    />
                    <span
                      className={
                        plan.featured ? "text-stone-300" : "text-stone-600"
                      }
                    >
                      {feat}
                    </span>
                  </li>
                ))}
              </ul>

              <a
                href="/create"
                className={`mt-8 block text-center h-11 leading-[2.75rem] rounded-full text-sm font-medium transition-colors ${
                  plan.featured
                    ? "bg-accent text-white hover:bg-accent-hover"
                    : "bg-stone-100 text-stone-900 hover:bg-stone-200"
                }`}
              >
                {plan.cta}
              </a>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
