"use client";

import Link from "next/link";
import { Aperture, Images, Search, Sparkles, UserRound } from "lucide-react";
import { usePathname } from "next/navigation";

const items = [
  { href: "/", label: "Discover", icon: Search },
  { href: "/inspiration", label: "Create", icon: Sparkles },
  { href: "/library", label: "Library", icon: Images },
  { href: "/account", label: "You", icon: UserRound },
];

export function PortalHeader({ light = false }: { light?: boolean }) {
  return (
    <header className={`portal-header ${light ? "portal-header-light" : ""}`}>
      <Link href="/" className="portal-brand" aria-label="FlashShot home">
        <Aperture aria-hidden="true" size={20} strokeWidth={1.8} />
        <span>FlashShot</span>
      </Link>
      <nav className="portal-desktop-nav" aria-label="Primary navigation">
        <Link href="/">Discover</Link>
        <Link href="/inspiration">Shoot a reference</Link>
        <Link href="/library">My portraits</Link>
      </nav>
      <Link href="/inspiration" className="portal-header-action">
        <Sparkles aria-hidden="true" size={16} />
        Create
      </Link>
    </header>
  );
}

export function PortalBottomNav() {
  const pathname = usePathname();
  return (
    <nav className="portal-bottom-nav" aria-label="Mobile navigation">
      {items.map(({ href, label, icon: Icon }) => {
        const active = href === "/"
          ? pathname === "/"
          : href === "/inspiration"
            ? pathname.startsWith("/inspiration") || pathname.startsWith("/create")
            : pathname.startsWith(href);
        return (
          <Link key={href} href={href} className={active ? "is-active" : undefined}>
            <Icon aria-hidden="true" size={20} strokeWidth={active ? 2.2 : 1.7} />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
