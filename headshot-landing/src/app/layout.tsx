import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { PwaRegistration } from "@/components/portrait/PwaRegistration";
import "./globals.css";

export const metadata: Metadata = {
  applicationName: "FlashShot",
  title: "FlashShot — Your private AI portrait studio",
  description:
    "Bring a portrait reference you love and step into a complete photo story that still looks unmistakably like you.",
  openGraph: {
    title: "FlashShot — Your private AI portrait studio",
    description:
      "Bring a portrait reference you love and step into a complete photo story that still looks unmistakably like you.",
    type: "website",
  },
  manifest: "/manifest.webmanifest",
  icons: {
    icon: [
      { url: "/flashshot-mark.svg", type: "image/svg+xml" },
      { url: "/icons/flashshot-192.png", sizes: "192x192", type: "image/png" },
    ],
    apple: [{ url: "/icons/flashshot-180.png", sizes: "180x180", type: "image/png" }],
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "default",
    title: "FlashShot",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={GeistSans.variable}>
      <body>
        {children}
        <PwaRegistration />
      </body>
    </html>
  );
}
