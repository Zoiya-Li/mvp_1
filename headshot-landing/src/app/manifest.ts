import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "FlashShot AI Portrait Studio",
    short_name: "FlashShot",
    description: "Create private, identity-consistent AI portrait stories from a look you love.",
    start_url: "/",
    display: "standalone",
    background_color: "#f7f7f3",
    theme_color: "#171716",
    orientation: "portrait-primary",
    categories: ["photo", "lifestyle", "entertainment"],
    icons: [
      { src: "/icons/flashshot-192.png", sizes: "192x192", type: "image/png" },
      { src: "/icons/flashshot-512.png", sizes: "512x512", type: "image/png" },
      { src: "/icons/flashshot-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
