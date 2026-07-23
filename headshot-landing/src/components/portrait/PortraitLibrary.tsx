"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowRight, Images, LoaderCircle, Plus, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  deletePortraitProject, getPortraitPhotoSet, getTheme, listPortraitProjects,
  loadPortraitAsset, PortraitProject, PortraitTheme, shouldBypassImageOptimization,
} from "@/lib/portrait-v2";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

type LibraryItem = {
  project: PortraitProject;
  theme: PortraitTheme | null;
  coverUrl: string | null;
};

const STATUS_LABELS: Record<string, string> = {
  draft: "Direction saved",
  awaiting_references: "Add identity photos",
  ready: "Ready for preview",
  preview_generating: "Creating first look",
  preview_ready: "First look ready",
  set_generating: "Creating complete shoot",
  delivered: "Complete shoot",
  failed: "Needs attention",
};

export function PortraitLibrary() {
  const [items, setItems] = useState<LibraryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const privateUrls = useRef<string[]>([]);

  async function removeProject(projectId: string) {
    if (!window.confirm("Delete this project, its identity photos, inspiration, and generated portraits? This cannot be undone.")) return;
    try {
      await deletePortraitProject(projectId);
      setItems((current) => current?.filter((item) => item.project.project_id !== projectId) ?? []);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not delete this project");
    }
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const projects = await listPortraitProjects();
        const resolved = await Promise.all(projects.map(async (project) => {
          const theme = project.theme_id ? await getTheme(project.theme_id).catch(() => null) : null;
          let coverUrl: string | null = null;
          if (project.status === "delivered" && project.photo_set_id) {
            try {
              const set = await getPortraitPhotoSet(project.project_id, project.photo_set_id);
              if (set.cover_asset_id) {
                coverUrl = await loadPortraitAsset(project.project_id, set.cover_asset_id);
                privateUrls.current.push(coverUrl);
              }
            } catch {
              // Fall back to the direction cover if a retained asset has expired.
            }
          }
          return { project, theme, coverUrl };
        }));
        if (!cancelled) setItems(resolved);
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : "Could not open your library");
          setItems([]);
        }
      }
    })();
    return () => {
      cancelled = true;
      privateUrls.current.forEach((url) => URL.revokeObjectURL(url));
      privateUrls.current = [];
    };
  }, []);

  return (
    <div className="simple-portal-page">
      <PortalHeader />
      <main className="portrait-library">
        <div className="library-heading">
          <div><p className="step-count">Private by default</p><h1>Your portrait stories</h1></div>
          <Link href="/" className="library-new"><Plus size={17} /> New shoot</Link>
        </div>
        {items === null && <div className="library-loading"><LoaderCircle className="spin" size={24} /> Opening your studio</div>}
        {items?.length === 0 && (
          <section className="empty-library library-empty-inline">
            <Images size={29} strokeWidth={1.5} />
            <p>Your private portrait library</p>
            <h1>The stories you make will live here.</h1>
            <span>{error ?? "Saved sets stay private until you choose to share one."}</span>
            <Link href="/" className="dark-command-button">Find a story <ArrowRight size={17} /></Link>
          </section>
        )}
        {items && items.length > 0 && <div className="library-grid">
          {items.map(({ project, theme, coverUrl }, index) => (
            <article className="library-project" key={project.project_id}>
              <Link href={`/create?project=${project.project_id}`} className="library-project-image">
                {coverUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={coverUrl} alt="" />
                ) : (
                  <Image
                    src={theme?.cover_image ?? "/images/film_f_cinematic.png"}
                    alt=""
                    fill
                    priority={index < 3}
                    sizes="(max-width: 680px) 100vw, 33vw"
                    unoptimized={shouldBypassImageOptimization(theme?.cover_image ?? "/images/film_f_cinematic.png")}
                  />
                )}
                <span>{project.source === "private_inspiration" ? "Your reference" : theme?.category ?? "Portrait"}</span>
              </Link>
              <div className="library-project-copy">
                <Link href={`/create?project=${project.project_id}`}><h2>{theme?.title_en ?? "Inspired portrait"}</h2><p>{STATUS_LABELS[project.status] ?? project.status}</p></Link>
                <div><Link href={`/create?project=${project.project_id}`} aria-label="Continue project"><ArrowRight size={18} /></Link><button title="Delete project" aria-label="Delete project" onClick={() => removeProject(project.project_id)}><Trash2 size={16} /></button></div>
              </div>
            </article>
          ))}
        </div>}
      </main>
      <PortalBottomNav />
    </div>
  );
}
