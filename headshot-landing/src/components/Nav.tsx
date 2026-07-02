"use client";

import { useState } from "react";
import { Menu, X } from "lucide-react";

const NAV_LINKS = [
  { label: "How it works", href: "#workflow" },
  { label: "Examples", href: "#gallery" },
  { label: "Pricing", href: "#pricing" },
  { label: "FAQ", href: "#faq" },
];

export function Nav() {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed top-0 inset-x-0 z-40 bg-stone-50/80 backdrop-blur-md border-b border-stone-200/60">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <a href="#" className="text-lg font-semibold tracking-tight">
          Flash<span className="text-accent">Shot</span>
        </a>

        {/* Desktop nav */}
        <ul className="hidden md:flex items-center gap-8 text-sm text-stone-500">
          {NAV_LINKS.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="hover:text-stone-900 transition-colors"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        <a
          href="/create"
          className="hidden md:inline-flex items-center h-10 px-5 rounded-full bg-stone-900 text-white text-sm font-medium hover:bg-stone-800 transition-colors"
        >
          Get started
        </a>

        {/* Mobile toggle */}
        <button
          onClick={() => setOpen(!open)}
          className="md:hidden p-2 text-stone-600"
          aria-label={open ? "Close menu" : "Open menu"}
        >
          {open ? <X size={24} /> : <Menu size={24} />}
        </button>
      </nav>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden border-t border-stone-200/60 bg-stone-50/95 backdrop-blur-md">
          <ul className="flex flex-col px-6 py-4 gap-4 text-sm text-stone-600">
            {NAV_LINKS.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="block py-1"
                >
                  {link.label}
                </a>
              </li>
            ))}
            <li>
              <a
                href="/create"
                onClick={() => setOpen(false)}
                className="inline-flex items-center h-10 px-5 rounded-full bg-stone-900 text-white text-sm font-medium"
              >
                Get started
              </a>
            </li>
          </ul>
        </div>
      )}
    </header>
  );
}
