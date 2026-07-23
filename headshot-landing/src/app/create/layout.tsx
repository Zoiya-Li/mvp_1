import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Create — FlashShot AI Portrait Studio",
};

export default function CreateLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
