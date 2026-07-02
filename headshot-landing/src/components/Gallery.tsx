import Image from "next/image";

const STYLES = [
  {
    category: "Traditional Chinese Studio",
    items: [
      { src: "/images/gf_m_hanfu.png", label: "Hanfu &middot; Refined Scholar" },
      { src: "/images/gf_f_qipao.png", label: "Qipao &middot; Republican Era" },
      { src: "/images/ac_m_01.png", label: "Intellectual &middot; Men" },
      { src: "/images/ac_f_01.png", label: "Intellectual &middot; Women" },
    ],
  },
  {
    category: "Japanese & Korean",
    items: [
      { src: "/images/jp_m_fresh.png", label: "Japanese Fresh &middot; Boyish" },
      { src: "/images/jp_f_fresh.png", label: "Japanese Fresh &middot; Airy" },
      { src: "/images/kr_m_minimal.png", label: "Korean Minimal &middot; Muted Grey" },
      { src: "/images/kr_f_elegant.png", label: "Korean Elegant &middot; Silk Texture" },
    ],
  },
  {
    category: "Retro & Cinematic",
    items: [
      { src: "/images/cr_m_retro.png", label: "Vintage Film &middot; Hong Kong Retro" },
      { src: "/images/film_f_cinematic.png", label: "Cinematic &middot; Storytelling Light" },
      { src: "/images/film_m_cyber.png", label: "Cyberpunk &middot; Neon Nights" },
      { src: "/images/film_f_dark.png", label: "Dark Gothic &middot; Mysterious" },
    ],
  },
  {
    category: "Lifestyle & Editorial",
    items: [
      { src: "/images/lw_f_01.png", label: "Cream Knit &middot; Soft Light" },
      { src: "/images/social_f_french.png", label: "French Effortless &middot; Relaxed Elegance" },
      { src: "/images/fz_f_editorial.png", label: "Editorial &middot; Cool Tones" },
      { src: "/images/fz_m_editorial.png", label: "Editorial &middot; High Contrast" },
    ],
  },
  {
    category: "Natural & Creative",
    items: [
      { src: "/images/cr_f_forest.png", label: "Natural &middot; Forest Fresh" },
      { src: "/images/cr_m_sport.png", label: "Sporty &middot; Sunlit" },
      { src: "/images/social_m_street.png", label: "Street Style &middot; Trendy" },
      { src: "/images/lw_m_02.png", label: "Knit Sweater &middot; Cafe Mood" },
    ],
  },
  {
    category: "Business & ID Photos",
    items: [
      { src: "/images/bf_m_01.png", label: "Urban Pro &middot; Navy Suit" },
      { src: "/images/bf_f_01.png", label: "Urban Pro &middot; Beige Suit" },
      { src: "/images/id_m_red.png", label: "Standard ID Photo &middot; Red Backdrop" },
      { src: "/images/id_f_white.png", label: "Standard ID Photo &middot; White Backdrop" },
    ],
  },
];

export function Gallery() {
  return (
    <section id="gallery" className="py-24 md:py-32 bg-white">
      <div className="mx-auto max-w-7xl px-6">
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">
          Popular portrait themes
        </h2>
        <p className="mt-3 text-stone-500 max-w-md">
          Traditional Chinese, Hong Kong retro, French, Japanese, Korean, cinematic, editorial &mdash; 25+ portrait themes, each generated just for you.
        </p>

        <div className="mt-14 space-y-14">
          {STYLES.map((group) => (
            <div key={group.category}>
              <h3 className="text-sm font-semibold text-accent tracking-wide mb-5">
                {group.category}
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {group.items.map((item) => (
                  <div
                    key={item.src}
                    className="group relative aspect-[3/4] rounded-2xl overflow-hidden bg-stone-100 cursor-pointer"
                  >
                    <Image
                      src={item.src}
                      alt={item.label}
                      fill
                      className="object-cover transition-transform duration-500 group-hover:scale-105"
                      sizes="(max-width: 768px) 50vw, 25vw"
                    />
                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-stone-900/60 to-transparent p-3 pt-8 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <span className="text-white text-xs font-medium">
                        {item.label}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
