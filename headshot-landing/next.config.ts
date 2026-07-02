import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    formats: ["image/avif", "image/webp"],
  },
  // The git repo root is the user's home dir, which also has a stray
  // package-lock.json. Pin Turbopack's workspace root to THIS project so it
  // doesn't infer the home dir (and watch all of it). __dirname is available
  // here because the config is transpiled to CommonJS (no "type": "module").
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
      {
        source: "/ws/:path*",
        destination: "http://localhost:8000/ws/:path*",
      },
    ];
  },
};

export default nextConfig;
