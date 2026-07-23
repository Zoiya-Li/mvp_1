"use client";

import Image from "next/image";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  ArrowLeft, ArrowRight, Camera, Check, Download, ImagePlus, Images,
  LoaderCircle, LockKeyhole, RefreshCcw, Share2, ShieldCheck, Sparkles, UserRoundCheck, X,
} from "lucide-react";
import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  createPortraitProject,
  createSharedRecipe,
  FALLBACK_THEMES,
  getPortraitPhotoSet,
  getPortraitProject,
  getSharedRecipe,
  getTheme,
  loadPortraitHero,
  loadPortraitAsset,
  PortraitPhotoSet,
  PortraitProject,
  ReferenceRoleQuality,
  retryPortraitPreview,
  SharedRecipe,
  shouldBypassImageOptimization,
  PortraitTheme,
  startPortraitPreview,
  uploadIdentityReferences,
} from "@/lib/portrait-v2";
import { PortalBottomNav, PortalHeader } from "./PortalNav";

type Phase = "direction" | "references" | "checking" | "ready" | "generating" | "preview" | "set_generating" | "delivered" | "failed";
type Presentation = "female" | "male";
type LocalPhotoCue = { tone: "reading" | "encouraging" | "gentle"; message: string };

const SLOTS = ["Front", "Smile", "Left 45°", "Right 45°", "Lifestyle", "Profile"];

export function PortraitCreateStudio() {
  const params = useSearchParams();
  const requestedTheme = params.get("theme");
  const requestedProject = params.get("project");
  const requestedRecipe = params.get("recipe");
  const [phase, setPhase] = useState<Phase>("direction");
  const [theme, setTheme] = useState<PortraitTheme | null>(null);
  const [project, setProject] = useState<PortraitProject | null>(null);
  const [sharedRecipe, setSharedRecipe] = useState<SharedRecipe | null>(null);
  const [presentation, setPresentation] = useState<Presentation>("female");
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [localPhotoCues, setLocalPhotoCues] = useState<LocalPhotoCue[]>([]);
  const [faceConsent, setFaceConsent] = useState(false);
  const [adultConfirmed, setAdultConfirmed] = useState(false);
  const [qualityIssues, setQualityIssues] = useState<string[]>([]);
  const [qualityFeedback, setQualityFeedback] = useState<ReferenceRoleQuality[]>([]);
  const [heroUrl, setHeroUrl] = useState<string | null>(null);
  const [photoSet, setPhotoSet] = useState<PortraitPhotoSet | null>(null);
  const [setUrls, setSetUrls] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [shareOpen, setShareOpen] = useState(false);
  const [sharePortrait, setSharePortrait] = useState(false);
  const [shareLink, setShareLink] = useState<string | null>(null);
  const [retryOpen, setRetryOpen] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [comparisonReveal, setComparisonReveal] = useState(62);
  const urls = useRef<Set<string>>(new Set());
  const activeProjectId = project?.project_id;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        if (requestedProject) {
          const existing = await getPortraitProject(requestedProject);
          if (!cancelled) {
            setProject(existing);
            if (existing.gender === "female" || existing.gender === "male") {
              setPresentation(existing.gender);
            }
            if (existing.status === "delivered") setPhase("delivered");
            else if (existing.status === "set_generating") setPhase("set_generating");
            else if (existing.status === "failed") setPhase("failed");
            else if (existing.status === "preview_ready") setPhase("preview");
            else if (existing.status === "preview_generating") setPhase("generating");
            else if (existing.status === "ready") setPhase("ready");
            else if (existing.status === "awaiting_references") setPhase("references");
          }
          if (existing.theme_id) {
            const selected = await getTheme(existing.theme_id);
            if (!cancelled) setTheme(selected);
          }
        } else if (requestedRecipe) {
          const shared = await getSharedRecipe(requestedRecipe);
          const selected = await getTheme(shared.theme_id || shared.theme_slug || FALLBACK_THEMES[0].slug);
          if (!cancelled) { setSharedRecipe(shared); setTheme(selected); }
        } else {
          const selected = await getTheme(requestedTheme || FALLBACK_THEMES[0].slug);
          if (!cancelled) setTheme(selected);
        }
      } catch (cause) {
        if (!cancelled) {
          setTheme(FALLBACK_THEMES.find((item) => item.slug === requestedTheme) ?? FALLBACK_THEMES[0]);
          setError(cause instanceof Error ? cause.message : "Could not open this direction");
        }
      }
    })();
    return () => { cancelled = true; };
  }, [requestedProject, requestedRecipe, requestedTheme]);

  useEffect(() => {
    if (phase !== "preview" || !project || heroUrl) return;
    let cancelled = false;
    loadPortraitHero(project.project_id).then((objectUrl) => {
      if (cancelled) return URL.revokeObjectURL(objectUrl);
      urls.current.add(objectUrl);
      setHeroUrl(objectUrl);
    }).catch((cause) => {
      if (!cancelled) setError(cause instanceof Error ? cause.message : "Could not open your preview");
    });
    return () => { cancelled = true; };
  }, [heroUrl, phase, project]);

  useEffect(() => {
    if (phase !== "generating" || !activeProjectId) return;
    let cancelled = false;
    const poll = window.setInterval(async () => {
      try {
        const latest = await getPortraitProject(activeProjectId);
        if (cancelled) return;
        setProject(latest);
        if (latest.status === "preview_ready") {
          const objectUrl = await loadPortraitHero(latest.project_id);
          if (cancelled) return URL.revokeObjectURL(objectUrl);
          urls.current.add(objectUrl);
          setHeroUrl((current) => {
            if (current) { URL.revokeObjectURL(current); urls.current.delete(current); }
            return objectUrl;
          });
          setPhase("preview");
          window.clearInterval(poll);
        }
      } catch {
        // A queued preview may take several minutes; keep the studio polling.
      }
    }, 4000);
    return () => { cancelled = true; window.clearInterval(poll); };
  }, [activeProjectId, phase]);

  useEffect(() => {
    if (phase !== "set_generating" || !activeProjectId) return;
    let cancelled = false;
    const poll = window.setInterval(async () => {
      try {
        const latest = await getPortraitProject(activeProjectId);
        if (cancelled) return;
        setProject(latest);
        if (latest.status === "delivered") {
          setPhase("delivered");
          window.clearInterval(poll);
        } else if (latest.status === "failed") {
          setPhase("failed");
          window.clearInterval(poll);
        }
      } catch {
        // Keep the studio connected while the six-shot batch is running.
      }
    }, 5000);
    return () => { cancelled = true; window.clearInterval(poll); };
  }, [activeProjectId, phase]);

  useEffect(() => {
    if (phase !== "delivered" || !project?.photo_set_id || photoSet) return;
    let cancelled = false;
    (async () => {
      try {
        const delivered = await getPortraitPhotoSet(project.project_id, project.photo_set_id!);
        const objectUrls = await Promise.all(
          delivered.assets.map((asset) => loadPortraitAsset(project.project_id, asset.asset_id)),
        );
        if (cancelled) {
          objectUrls.forEach((url) => URL.revokeObjectURL(url));
          return;
        }
        objectUrls.forEach((url) => urls.current.add(url));
        setPhotoSet(delivered);
        setSetUrls(objectUrls);
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : "Could not open your finished set");
      }
    })();
    return () => { cancelled = true; };
  }, [phase, photoSet, project]);

  useEffect(() => () => {
    urls.current.forEach((url) => URL.revokeObjectURL(url));
    urls.current.clear();
  }, []);

  const directionTitle = theme?.title_en ?? (project?.source === "private_inspiration" ? "Your inspiration" : "Portrait direction");
  const directionCover = theme?.cover_image ?? "/images/film_f_cinematic.png";
  const canUpload = files.length >= 4 && faceConsent && adultConfirmed;
  const progress = useMemo(() => ({
    direction: 1, references: 2, checking: 2, ready: 3, generating: 3, preview: 4,
    set_generating: 4, delivered: 4, failed: 4,
  }[phase]), [phase]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [phase]);

  async function readLocalPhoto(file: File, index: number): Promise<LocalPhotoCue> {
    const objectUrl = URL.createObjectURL(file);
    try {
      const image = document.createElement("img");
      image.src = objectUrl;
      await image.decode();
      if (Math.min(image.naturalWidth, image.naturalHeight) < 640) {
        return { tone: "gentle", message: "A larger photo will hold more of your detail" };
      }
      const canvas = document.createElement("canvas");
      canvas.width = 24;
      canvas.height = 24;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (context) {
        context.drawImage(image, 0, 0, 24, 24);
        const pixels = context.getImageData(0, 0, 24, 24).data;
        let luminance = 0;
        for (let offset = 0; offset < pixels.length; offset += 4) {
          luminance += 0.2126 * pixels[offset] + 0.7152 * pixels[offset + 1] + 0.0722 * pixels[offset + 2];
        }
        if (luminance / (pixels.length / 4) < 52) {
          return { tone: "gentle", message: "A brighter photo will reveal more of your expression" };
        }
      }
      const messages = [
        "The light here gives us a strong anchor",
        "This view adds beautiful shape and dimension",
        "Natural detail like this helps the portrait feel like you",
        "A clear supporting angle for a consistent set",
        "This gives the artist another honest view of you",
        "A useful final angle for your portrait session",
      ];
      return { tone: "encouraging", message: messages[index % messages.length] };
    } catch {
      return { tone: "gentle", message: "Choose another photo so we can read it clearly" };
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  }

  function addFiles(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []).filter(
      (file) => file.type.startsWith("image/") && file.size <= 10 * 1024 * 1024,
    );
    const next = [...files, ...selected].slice(0, 6);
    const additions = next.slice(files.length).map((file) => {
      const url = URL.createObjectURL(file);
      urls.current.add(url);
      return url;
    });
    setFiles(next);
    setPreviews((current) => [...current, ...additions].slice(0, next.length));
    setLocalPhotoCues((current) => [
      ...current,
      ...selected.slice(0, Math.max(0, 6 - files.length)).map(() => ({
        tone: "reading" as const,
        message: "Finding the light and detail",
      })),
    ].slice(0, next.length));
    const startingIndex = files.length;
    void Promise.all(selected.slice(0, Math.max(0, 6 - files.length)).map((file, offset) => (
      readLocalPhoto(file, startingIndex + offset)
    ))).then((cues) => {
      setLocalPhotoCues((current) => {
        const updated = [...current];
        cues.forEach((cue, offset) => { updated[startingIndex + offset] = cue; });
        return updated.slice(0, next.length);
      });
    });
    setQualityIssues([]);
    setQualityFeedback([]);
    event.target.value = "";
  }

  function removeFile(index: number) {
    const url = previews[index];
    if (url) { URL.revokeObjectURL(url); urls.current.delete(url); }
    setFiles((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setPreviews((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setLocalPhotoCues((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setQualityIssues([]);
    setQualityFeedback([]);
  }

  async function checkReferences() {
    if (!canUpload) return;
    setPhase("checking");
    setError(null);
    try {
      let activeProject = project;
      if (!activeProject) {
        activeProject = await createPortraitProject({
          theme_id: theme?.theme_id,
          source: sharedRecipe ? "shared_recipe" : "official_theme",
          gender: presentation,
          shared_recipe_id: sharedRecipe?.share_token,
        });
        setProject(activeProject);
      }
      const result = await uploadIdentityReferences({
        projectId: activeProject.project_id,
        files,
        gender: presentation,
      });
      const issues = result.reference_quality.issues ?? [];
      setQualityIssues(issues);
      setQualityFeedback(result.reference_quality.role_coverage ?? []);
      setPhase(result.reference_quality.pass ? "ready" : "references");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not check these photos");
      setPhase("references");
    }
  }

  async function generatePreview() {
    if (!project) return;
    setError(null);
    try {
      await startPortraitPreview(project.project_id);
      setPhase("generating");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start your preview");
    }
  }

  async function retryCloserMatch(reason: "identity" | "expression" | "overall") {
    if (!project || retrying) return;
    setRetrying(true);
    setError(null);
    try {
      const result = await retryPortraitPreview(project.project_id, reason);
      setProject((current) => current ? {
        ...current,
        status: result.status,
        hero_asset_id: null,
        preview_retries_used: 1,
        preview_retries_remaining: result.retries_remaining,
      } : current);
      setRetryOpen(false);
      setPhase("generating");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not create a closer match");
    } finally {
      setRetrying(false);
    }
  }

  async function shareProject() {
    if (!project) return;
    setError(null);
    try {
      const shared = await createSharedRecipe(project.project_id, sharePortrait);
      const url = `${window.location.origin}/s/${shared.share_token}`;
      setShareLink(url);
      if (navigator.share) {
        await navigator.share({ title: "Shoot this portrait look", url });
      } else {
        await navigator.clipboard?.writeText(url);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not create this share link");
    }
  }

  return (
    <div className="portrait-create-page">
      <PortalHeader />
      <main className="portrait-create-main">
        <div className="studio-topline">
          <Link href={sharedRecipe ? `/s/${sharedRecipe.share_token}` : theme ? `/themes/${theme.slug}` : "/inspiration"} className="inline-back">
            <ArrowLeft size={18} /> Back
          </Link>
          <div className="studio-progress" aria-label={`Step ${progress} of 4`}>
            {[1, 2, 3, 4].map((step) => <span key={step} className={step <= progress ? "active" : ""} />)}
          </div>
        </div>

        {phase === "direction" && (
          <section className="studio-direction">
            <div className="studio-direction-image"><Image src={directionCover} alt="" fill sizes="(max-width: 760px) 100vw, 50vw" priority unoptimized={shouldBypassImageOptimization(directionCover)} /></div>
            <div className="studio-direction-copy">
              <p className="step-count">01 / Shoot direction</p>
              <h1>{directionTitle}</h1>
              <p>{sharedRecipe ? "A shared visual recipe, rebuilt around your identity." : project?.source === "private_inspiration" ? "Your reference remains private. We transfer its visual language, never its person." : theme?.tagline}</p>
              <div className="presentation-control">
                <span>Wardrobe presentation</span>
                <div role="group" aria-label="Wardrobe presentation">
                  {(["female", "male"] as Presentation[]).map((value) => (
                    <button key={value} className={presentation === value ? "active" : ""} onClick={() => setPresentation(value)}>
                      {value === "female" ? "Women" : "Men"}
                    </button>
                  ))}
                </div>
              </div>
              <button className="dark-command-button" onClick={() => setPhase("references")}>Add photos of me <ArrowRight size={18} /></button>
            </div>
          </section>
        )}

        {(phase === "references" || phase === "checking") && (
          <section className="studio-references">
            <div className="studio-section-heading">
              <p className="step-count">02 / Your identity</p>
              <h1>Show us the moments that feel most like you.</h1>
              <p>Choose four to six clear views. Natural expressions give the portrait artist more of your character to hold onto.</p>
            </div>
            <div className="reference-grid">
              {Array.from({ length: 6 }, (_, index) => previews[index] ? (
                <div className="reference-item" key={previews[index]}>
                  <div className={`reference-slot filled ${qualityFeedback[index] && !qualityFeedback[index].pass ? "needs-replacement" : qualityFeedback[index]?.pass ? "quality-pass" : ""}`}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={previews[index]} alt={SLOTS[index]} />
                    <span>{SLOTS[index]}</span>
                    {qualityFeedback[index] && (
                      <div className="slot-quality" aria-label={`${SLOTS[index]} ${qualityFeedback[index].pass ? "passed" : "needs replacement"}`}>
                        {qualityFeedback[index].pass ? <Check size={15} /> : <X size={15} />}
                      </div>
                    )}
                    <button aria-label={`Remove ${SLOTS[index]}`} onClick={() => removeFile(index)}><X size={15} /></button>
                    {localPhotoCues[index]?.tone === "reading" && <span className="slot-scan" aria-hidden="true" />}
                  </div>
                  {localPhotoCues[index] && (
                    <p className={`reference-cue ${localPhotoCues[index].tone}`}>
                      {localPhotoCues[index].tone === "reading" ? <LoaderCircle className="spin" size={12} /> : <Check size={12} />}
                      {localPhotoCues[index].message}
                    </p>
                  )}
                </div>
              ) : (
                <div className="reference-item" key={SLOTS[index]}>
                  <label className="reference-slot">
                    {index === files.length && <input type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={addFiles} />}
                    <ImagePlus size={21} />
                    <span>{SLOTS[index]}</span>
                    <small>{index < 4 ? "Needed" : "Helpful"}</small>
                  </label>
                </div>
              ))}
            </div>
            <div className="reference-notes">
              <span><Check size={15} /> Just you</span><span><Check size={15} /> Natural light</span><span><Check size={15} /> Eyes easy to see</span>
            </div>
            {qualityIssues.length > 0 && (
              <div className="quality-feedback" role="alert">
                <strong>A few photos need a gentler second take.</strong>
                {qualityFeedback.map((item, index) => !item.pass ? (
                    <div className="quality-feedback-item" key={item.role}>
                      <b>{SLOTS[index] ?? item.role.replaceAll("_", " ")}</b>
                      <span>{item.headline ?? "This view needs another photo"}</span>
                      {item.guidance && <p>{item.guidance}</p>}
                    </div>
                  ) : null)}
                {qualityIssues.includes("reference_identity_mismatch") && (
                  <div className="quality-feedback-item">
                    <b>Same person check</b>
                    <span>These photos may show different people.</span>
                    <p>Use four to six photos of the same adult person.</p>
                  </div>
                )}
              </div>
            )}
            <div className="consent-stack">
              <h2>A private promise</h2>
              <label><input type="checkbox" checked={faceConsent} onChange={(event) => setFaceConsent(event.target.checked)} /><ShieldCheck size={20} /><span><strong>Keep this shoot private</strong><small>I consent to using these photos only to create this portrait set. They are not used for training by default.</small></span></label>
              <label><input type="checkbox" checked={adultConfirmed} onChange={(event) => setAdultConfirmed(event.target.checked)} /><UserRoundCheck size={20} /><span><strong>A portrait session for adults</strong><small>I confirm the person shown is me and is 18 or older.</small></span></label>
              <p><LockKeyhole size={13} /> Source photos are never posted publicly and are removed within seven days.</p>
            </div>
            {error && <p className="form-error">{error}</p>}
            <button className="dark-command-button" disabled={!canUpload || phase === "checking"} onClick={checkReferences}>
              {phase === "checking" ? <LoaderCircle className="spin" size={18} /> : <Camera size={18} />}
              {phase === "checking" ? "Finding your strongest views" : `Check ${files.length || "my"} photos`}
            </button>
          </section>
        )}

        {phase === "ready" && (
          <section className="studio-ready">
            <div className="ready-mark"><Check size={24} /></div>
            <p className="step-count">03 / Ready to create</p>
            <h1>Your identity pack is strong.</h1>
            <p>We will make one close portrait first. You decide whether it feels like you before a full set is created.</p>
            <div className="preview-promise"><Sparkles size={20} /><div><strong>One complimentary hero preview</strong><span>Identity and final-quality gates run before it is shown.</span></div></div>
            {error && <p className="form-error">{error}</p>}
            <button className="dark-command-button" onClick={generatePreview}><Sparkles size={18} /> Create my first portrait</button>
          </section>
        )}

        {phase === "generating" && (
          <section className="studio-darkroom">
            <div className="darkroom-visual" aria-hidden="true">
              <div className="darkroom-print">
                {/* Object URLs are already-local selected photos and cannot use the Next image optimizer. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {(heroUrl || previews[0]) && <img src={heroUrl || previews[0]} alt="" />}
                <span />
              </div>
            </div>
            <div className="darkroom-copy">
              <p className="step-count">{heroUrl ? "A closer direction" : "Your first exposure"}</p>
              <h1>{heroUrl ? "Listening closely to what felt off." : "Setting the light around your expression."}</h1>
              <p>{heroUrl ? "Your first portrait stays safe while we bring your familiar features forward." : "We are directing a frame that feels photographed, personal, and still unmistakably you."}</p>
              <div className="darkroom-notes"><span>Reading your expression</span><span>Shaping the studio light</span><span>Reviewing before the reveal</span></div>
              <small>You can leave at any time. We will keep working in your Library.</small>
            </div>
          </section>
        )}

        {phase === "preview" && (
          <section className="studio-preview">
            <div className="studio-preview-image">
              {heroUrl && previews[0] ? <div className="portrait-comparison">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previews[0]} alt="Your source photo" />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img className="portrait-layer" style={{ clipPath: `inset(0 ${100 - comparisonReveal}% 0 0)` }} src={heroUrl} alt="Your FlashShot portrait" />
                <span className="comparison-divider" style={{ left: `${comparisonReveal}%` }}><ArrowRight size={16} /></span>
                <b className="comparison-label portrait-label">PORTRAIT</b><b className="comparison-label source-label">SOURCE</b>
                <input aria-label="Compare source and portrait" type="range" min="5" max="95" value={comparisonReveal} onChange={(event) => setComparisonReveal(Number(event.target.value))} />
              </div> : heroUrl ? <Image src={heroUrl} alt="Your FlashShot portrait preview" fill unoptimized sizes="(max-width: 760px) 100vw, 55vw" /> : <div className="preview-loading"><LoaderCircle className="spin" size={25} /><span>Opening your preview</span></div>}
            </div>
            <div className="studio-preview-copy">
              <p className="step-count">Your first portrait is ready</p>
              <h1>There you are.</h1>
              <p>Drag across the portrait and meet the version of you we found in a different light.</p>
              {heroUrl && <div className="bundle-peek">
                <div><strong>Five more frames are waiting for their light</strong><span>Each composition is created fresh after purchase.</span></div>
                <div className="bundle-peek-row">{["Close", "Half", "Full", "Seated", "Scene"].map((label, index) => <figure key={label}><span><Image src={heroUrl} alt="" fill unoptimized sizes="90px" style={{ objectPosition: `${35 + index * 8}% center` }} /><LockKeyhole size={13} /></span><figcaption>{label}</figcaption></figure>)}</div>
              </div>}
              {(project?.preview_retries_remaining ?? 1) > 0 ? (
                <>
                  <button className="text-command" onClick={() => setRetryOpen((value) => !value)}>
                    <RefreshCcw size={15} /> Guide it closer to me
                  </button>
                  {retryOpen && (
                    <div className="preview-retry-panel">
                      <strong>Let&apos;s make the first portrait earn your trust.</strong>
                      <p>Your one complimentary closer-match retry does not use another preview credit. What feels most off?</p>
                      <div>
                        <button disabled={retrying} onClick={() => retryCloserMatch("identity")}>Facial likeness</button>
                        <button disabled={retrying} onClick={() => retryCloserMatch("expression")}>Expression</button>
                        <button disabled={retrying} onClick={() => retryCloserMatch("overall")}>Overall feeling</button>
                      </div>
                      {retrying && <span><LoaderCircle className="spin" size={14} /> Directing a closer match</span>}
                    </div>
                  )}
                </>
              ) : (
                <p className="preview-retry-used">We used your closer-match direction. Continue only if this portrait now feels right.</p>
              )}
              <Link href={`/checkout?project=${project?.project_id ?? ""}`} className="dark-command-button">Complete my portrait story <ArrowRight size={18} /></Link>
              <button className="text-command" onClick={() => setShareOpen((value) => !value)}><Share2 size={15} /> Share this look</button>
              {shareOpen && <div className="share-controls">
                <label><input type="checkbox" checked={sharePortrait} onChange={(event) => setSharePortrait(event.target.checked)} /><span>Show my finished preview on the public link. Leave off to share only the reusable visual recipe.</span></label>
                <button onClick={shareProject}><Share2 size={15} /> {shareLink ? "Copy or share again" : "Create share link"}</button>
                {shareLink && <a href={shareLink}>{shareLink}</a>}
              </div>}
              {!shareOpen && <p className="preview-private"><LockKeyhole size={14} /> Private until you choose otherwise</p>}
              {error && <p className="form-error">{error}</p>}
            </div>
          </section>
        )}

        {phase === "set_generating" && (
          <section className="studio-generating set-generating">
            <LoaderCircle className="spin" size={30} />
            <p className="step-count">Complete shoot / In progress</p>
            <h1>Six portraits. One visual story.</h1>
            <p>Each frame is generated separately, checked against your identity pack, and held back unless it clears final QA.</p>
            <div className="generation-stages"><span className="done">Direction locked</span><span className="active">Six-shot sequence</span><span>Private delivery</span></div>
            <Link href="/library" className="text-command"><Images size={16} /> Continue in the background</Link>
          </section>
        )}

        {phase === "delivered" && (
          <section className="delivered-set">
            <div className="delivered-heading">
              <div><p className="step-count">Complete shoot / Delivered</p><h1>{photoSet?.title ?? "Your portrait story"}</h1></div>
              <div className="delivered-actions">
                <button className="text-command" onClick={() => setShareOpen((value) => !value)}><Share2 size={16} /> Share the look</button>
                <Link href="/library" className="text-command"><Images size={16} /> Library</Link>
              </div>
            </div>
            {setUrls.length === 0 ? (
              <div className="set-loading"><LoaderCircle className="spin" size={25} /><span>Opening six finished portraits</span></div>
            ) : (
              <div className="delivered-grid">
                {setUrls.map((url, index) => (
                  <figure key={photoSet?.assets[index]?.asset_id ?? url}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={url} alt={`Finished portrait ${index + 1}`} />
                    <figcaption><span>{String(index + 1).padStart(2, "0")}</span><a href={url} download={`flashshot-portrait-${index + 1}.png`} aria-label={`Download portrait ${index + 1}`}><Download size={17} /></a></figcaption>
                  </figure>
                ))}
              </div>
            )}
            {shareOpen && <div className="share-controls delivered-share">
              <label><input type="checkbox" checked={sharePortrait} onChange={(event) => setSharePortrait(event.target.checked)} /><span>Show one portrait on the public page. Leave off to share only the reusable visual recipe.</span></label>
              <button onClick={shareProject}><Share2 size={15} /> {shareLink ? "Copy or share again" : "Create share link"}</button>
              {shareLink && <a href={shareLink}>{shareLink}</a>}
            </div>}
            {error && <p className="form-error">{error}</p>}
          </section>
        )}

        {phase === "failed" && (
          <section className="studio-ready studio-failed">
            <div className="ready-mark"><X size={24} /></div>
            <p className="step-count">Quality hold</p>
            <h1>We stopped before delivering a weak set.</h1>
            <p>At least one portrait did not clear the final identity or quality gate. Your source photos and unfinished results remain private.</p>
            <div className="failed-actions"><Link href="/library" className="dark-command-button">Return to library</Link><Link href="/" className="text-command">Start a new direction</Link></div>
          </section>
        )}
      </main>
      <PortalBottomNav />
    </div>
  );
}
