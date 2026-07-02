import { ShieldCheck, EyeOff, Trash2 } from "lucide-react";

const GUARANTEES = [
  {
    icon: ShieldCheck,
    title: "Used only for generation",
    body: "Your photos are used solely to generate your portraits — never for model training or anything else."
  },
  {
    icon: EyeOff,
    title: "Never public",
    body: "Without your explicit consent, your photos and generated portraits will never appear anywhere public."
  },
  {
    icon: Trash2,
    title: "Delete anytime",
    body: "Original photos are auto-deleted 7 days after delivery. You can also request immediate deletion of all your data at any time."
  },
];

export function Privacy() {
  return (
    <section className="py-24 md:py-32 bg-white">
      <div className="mx-auto max-w-7xl px-6">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">
            Your face, your privacy
          </h2>
          <p className="mt-3 text-stone-500 max-w-lg mx-auto">
            Your face is sensitive personal data. We&apos;ve treated it as our highest-priority privacy concern from day one of the design.
          </p>
        </div>

        <div className="mt-14 grid md:grid-cols-3 gap-8">
          {GUARANTEES.map((g) => (
            <div key={g.title} className="text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-accent-light text-accent mb-4">
                <g.icon size={24} />
              </div>
              <h3 className="font-semibold">{g.title}</h3>
              <p className="mt-2 text-sm text-stone-500 leading-relaxed">
                {g.body}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
