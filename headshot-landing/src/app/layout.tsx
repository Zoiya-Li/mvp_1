import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FlashShot — AI Portrait Studio · Artistic portraits that look like you",
  description:
    "Upload a few selfies and generate multiple artistic portrait collections that look like you. Hanfu, Hong Kong style, French, Japanese, Korean, cinematic and more — get your own portrait collection without going to a photo studio.",
  openGraph: {
    title: "FlashShot — AI Portrait Studio · Artistic portraits that look like you",
    description:
      "Upload a few selfies and generate multiple artistic portrait collections that look like you. Hanfu, Hong Kong style, French, Japanese, Korean, cinematic and more — get your own portrait collection without going to a photo studio.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-stone-50 text-stone-900 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
