import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Create — FlashShot AI Portrait Studio",
};

export default function CreateLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-stone-50">
      {/* Simplified header */}
      <header className="fixed top-0 inset-x-0 z-40 bg-stone-50/80 backdrop-blur-md border-b border-stone-200/60">
        <nav className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          <Link href="/" className="text-lg font-semibold tracking-tight">
            Flash<span className="text-accent">Shot</span>
            <span className="ml-2 text-xs font-normal text-stone-400">AI Portrait Studio</span>
          </Link>
          <Link
            href="/"
            className="text-sm text-stone-500 hover:text-stone-700 transition-colors"
          >
            Back to home
          </Link>
        </nav>
      </header>
      <main className="pt-14">{children}</main>
    </div>
  );
}
