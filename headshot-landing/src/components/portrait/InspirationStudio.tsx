"use client";

import Image from "next/image";
import Link from "next/link";
import { ArrowLeft, ArrowRight, Check, ImagePlus, LoaderCircle, LockKeyhole, X } from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useState } from "react";

import { createPortraitProject, uploadInspiration } from "@/lib/portrait-v2";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

type Phase = "choose" | "review" | "analyzing" | "result";

export function InspirationStudio() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("choose");
  const [rights, setRights] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [spec, setSpec] = useState<Record<string, unknown> | null>(null);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const detailRows = useMemo(() => {
    if (!spec) return [];
    return [
      ["Scene", spec.scene], ["Wardrobe", spec.wardrobe],
      ["Light", spec.lighting], ["Composition", spec.composition],
      ["Mood", spec.mood],
    ].filter((row) => typeof row[1] === "string");
  }, [spec]);

  function choose(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setPhase("review");
    setError(null);
  }

  async function analyze() {
    if (!file || !rights) return;
    setPhase("analyzing");
    setError(null);
    try {
      const project = await createPortraitProject({ source: "private_inspiration" });
      setProjectId(project.project_id);
      const result = await uploadInspiration(project.project_id, file);
      setSpec(result.inspiration_spec);
      setPhase("result");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not read this reference");
      setPhase("review");
    }
  }

  return (
    <div className="inspiration-studio-page">
      <PortalHeader />
      <main className="inspiration-studio-main">
        <Link href="/" className="inline-back"><ArrowLeft size={18} /> Discover</Link>

        {phase === "choose" && (
          <section className="inspiration-chooser">
            <p className="step-count">01 / Bring a reference</p>
            <h1>What would you love to step into?</h1>
            <p className="inspiration-lead">
              Choose one photo for its scene, light, wardrobe, and composition.
              The person in it is never copied.
            </p>
            <label className="inspiration-dropzone">
              <input type="file" accept="image/jpeg,image/png,image/webp,image/heic" onChange={choose} />
              <ImagePlus size={28} strokeWidth={1.6} />
              <strong>Choose from your photos</strong>
              <span>JPG, PNG, WEBP, or HEIC · up to 10 MB</span>
            </label>
            <div className="reference-guidance">
              <span><Check size={15} /> One clear subject</span>
              <span><Check size={15} /> Visible scene and clothing</span>
              <span><X size={15} /> No explicit or private images</span>
            </div>
          </section>
        )}

        {(phase === "review" || phase === "analyzing") && preview && (
          <section className="inspiration-review">
            <div className="inspiration-review-image">
              <Image src={preview} alt="Selected inspiration" fill unoptimized sizes="(max-width: 760px) 100vw, 50vw" />
            </div>
            <div className="inspiration-review-copy">
              <p className="step-count">02 / Private style analysis</p>
              <h1>We borrow the photography, not the person.</h1>
              <div className="privacy-rule"><LockKeyhole size={19} /><p>This image stays private and is never published as a template.</p></div>
              <label className="rights-check">
                <input type="checkbox" checked={rights} onChange={(event) => setRights(event.target.checked)} />
                <span>I may use this image as a private style reference.</span>
              </label>
              {error && <p className="form-error">{error}</p>}
              <button className="dark-command-button" disabled={!rights || phase === "analyzing"} onClick={analyze}>
                {phase === "analyzing" ? <LoaderCircle className="spin" size={18} /> : <ArrowRight size={18} />}
                {phase === "analyzing" ? "Reading the visual language" : "Read this reference"}
              </button>
              <button className="text-command" onClick={() => setPhase("choose")} disabled={phase === "analyzing"}>Choose another photo</button>
            </div>
          </section>
        )}

        {phase === "result" && preview && (
          <section className="inspiration-result">
            <div className="inspiration-result-heading">
              <p className="step-count">03 / Your shoot direction</p>
              <h1>We found the visual language.</h1>
              <p>{spec ? "Review the direction, then add photos of yourself." : "Your reference is private and queued for visual analysis."}</p>
            </div>
            <div className="inspiration-result-layout">
              <div className="inspiration-result-image"><Image src={preview} alt="Analyzed inspiration" fill unoptimized sizes="50vw" /></div>
              <div className="shoot-direction">
                {detailRows.length ? detailRows.map(([label, value]) => (
                  <div key={String(label)}><span>{String(label)}</span><strong>{String(value)}</strong></div>
                )) : <p className="analysis-pending">Analysis will finish in your private workspace. You can add your identity photos now.</p>}
                <Link href={`/create?project=${projectId ?? ""}`} className="dark-command-button">
                  Add photos of me <ArrowRight size={17} />
                </Link>
              </div>
            </div>
          </section>
        )}
      </main>
      <PortalBottomNav />
    </div>
  );
}

