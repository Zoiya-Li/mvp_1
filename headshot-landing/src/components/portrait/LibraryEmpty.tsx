import Link from "next/link";
import { ArrowRight, Images } from "lucide-react";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

export function LibraryEmpty() {
  return (
    <div className="simple-portal-page">
      <PortalHeader />
      <main className="empty-library">
        <Images size={29} strokeWidth={1.5} />
        <p>Your private portrait library</p>
        <h1>The stories you make will live here.</h1>
        <span>Saved sets stay private until you choose to share one.</span>
        <Link href="/" className="dark-command-button">Find a story <ArrowRight size={17} /></Link>
      </main>
      <PortalBottomNav />
    </div>
  );
}

