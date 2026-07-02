"use client";

import { useState, useEffect, useCallback } from "react";
import type {
  SessionResponse,
  JobResponse,
  GeneratedImage,
  StyleKey,
  Gender,
  WSMessage,
  GenerationProgress,
} from "@/lib/types";
import {
  createSession,
  uploadPhotos,
  startHeroPreview,
  unlockFullSet,
  startMultiStyleGeneration,
  submitRevision,
  getSession,
  getJobs,
  deleteSession,
} from "@/lib/api";
import { useWebSocket } from "@/lib/websocket";
import { StepGuide } from "@/components/create/StepGuide";
import { StepUpload } from "@/components/create/StepUpload";
import { StepStyle } from "@/components/create/StepStyle";
import { StepGenerating } from "@/components/create/StepGenerating";
import { StepHeroPreview } from "@/components/create/StepHeroPreview";
import { StepResults } from "@/components/create/StepResults";

type Step =
  | "guide"
  | "upload"
  | "style"
  | "hero_generating"
  | "hero_preview"
  | "full_set_generating"
  | "results";

const PERSIST_KEY = "flashshot_create_state";

interface PersistedState {
  step: Step;
  sessionId: string | null;
  style: StyleKey | null;
  gender: Gender | null;
}

function loadPersisted(): PersistedState | null {
  try {
    const raw = localStorage.getItem(PERSIST_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedState;
  } catch {
    return null;
  }
}

function persist(state: PersistedState) {
  try {
    localStorage.setItem(PERSIST_KEY, JSON.stringify(state));
  } catch {
    // ignore storage errors
  }
}

function clearPersisted() {
  try {
    localStorage.removeItem(PERSIST_KEY);
  } catch {
    // ignore
  }
}

export default function CreatePage() {
  const [initialPersisted] = useState<PersistedState | null>(() => {
    if (typeof window === "undefined") return null;
    return loadPersisted();
  });

  const [step, setStep] = useState<Step>(initialPersisted?.step ?? "guide");
  const [sessionId, setSessionId] = useState<string | null>(
    initialPersisted?.sessionId ?? null
  );
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [jobs, setJobs] = useState<JobResponse[]>([]);
  const [progressByJob, setProgressByJob] = useState<
    Record<string, GenerationProgress>
  >({});
  const [files, setFiles] = useState<File[]>([]);
  const [faceProcessingConsent, setFaceProcessingConsent] = useState(false);
  const [adultSubjectConfirmed, setAdultSubjectConfirmed] = useState(false);
  const [style, setStyle] = useState<StyleKey | null>(
    initialPersisted?.style ?? null
  );
  const [gender, setGender] = useState<Gender | null>(
    initialPersisted?.gender ?? null
  );
  const [error, setError] = useState<string | null>(null);

  const handleWsMessage = useCallback(
    (msg: WSMessage) => {
      if (!sessionId) return;

      if (msg.type === "image_ready") {
        getSession(sessionId).then(setSession).catch(() => {});
        setJobs((prev) =>
          prev.map((j) =>
            j.job_id === (msg.job_id as string)
              ? {
                  ...j,
                  status: "completed",
                  result_image: msg.image as GeneratedImage,
                }
              : j
          )
        );
        setProgressByJob((prev) => {
          if (!(msg.job_id as string)) return prev;
          const next = { ...prev };
          delete next[msg.job_id as string];
          return next;
        });
      } else if (msg.type === "generation_progress") {
        const jobId = msg.job_id as string;
        if (!jobId) return;
        setProgressByJob((prev) => ({
          ...prev,
          [jobId]: {
            iteration: Number(msg.iteration ?? 0),
            max_iterations: Number(msg.max_iterations ?? 0),
            phase: msg.phase as GenerationProgress["phase"],
            detail: String(msg.detail ?? ""),
            shot_spec: (msg.shot_spec as GenerationProgress["shot_spec"]) ?? null,
          },
        }));
      } else if (msg.type === "job_started") {
        setJobs((prev) =>
          prev.map((j) =>
            j.job_id === (msg.job_id as string)
              ? {
                  ...j,
                  status: "processing",
                  shot_spec:
                    (msg.shot_spec as JobResponse["shot_spec"]) ?? j.shot_spec,
                }
              : j
          )
        );
      } else if (msg.type === "job_failed") {
        setJobs((prev) =>
          prev.map((j) =>
            j.job_id === (msg.job_id as string)
              ? { ...j, status: "failed", error: msg.error as string }
              : j
          )
        );
        setProgressByJob((prev) => {
          if (!(msg.job_id as string)) return prev;
          const next = { ...prev };
          delete next[msg.job_id as string];
          return next;
        });
        getSession(sessionId).then(setSession).catch(() => {});
      }
    },
    [sessionId]
  );

  const { status: wsStatus } = useWebSocket(sessionId, handleWsMessage);

  // Restore session data when sessionId becomes known
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    (async () => {
      try {
        const [sess, existingJobs] = await Promise.all([
          getSession(sessionId),
          getJobs(sessionId),
        ]);
        if (cancelled) return;
        setSession(sess);

        if (existingJobs.length > 0) setJobs(existingJobs);

        const hasHero = sess.hero_preview_image_id != null;
        const hasImages = sess.generated_images && sess.generated_images.length > 0;
        const stillWorking = existingJobs.some(
          (j) => j.status === "queued" || j.status === "processing"
        );

        if (stillWorking) {
          // Determine which generation phase we're in
          const heroJob = existingJobs.find(
            (j) => j.job_type === "hero_preview" && j.status === "processing"
          );
          const fullSetJobs = existingJobs.filter(
            (j) => j.job_type === "full_set" || j.job_type === "generate"
          );
          const fullSetWorking = fullSetJobs.some(
            (j) => j.status === "queued" || j.status === "processing"
          );
          if (heroJob) {
            setStep("hero_generating");
          } else if (fullSetWorking) {
            setStep("full_set_generating");
          } else {
            setStep("hero_generating");
          }
        } else if (hasImages && sess.unlocked) {
          setStep("results");
        } else if (hasHero) {
          setStep("hero_preview");
        } else if (hasImages) {
          // Legacy: images exist but no hero flag (backward compat)
          setStep("results");
        }
      } catch {
        if (cancelled) return;
        setSessionId(null);
        setSession(null);
        setJobs([]);
        setStep("guide");
        clearPersisted();
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // Persist step & sessionId changes
  useEffect(() => {
    persist({ step, sessionId, style, gender });
  }, [step, sessionId, style, gender]);

  // Auto-advance when jobs complete
  useEffect(() => {
    if (jobs.length === 0) return;
    const allDone = jobs.every(
      (j) => j.status === "completed" || j.status === "failed"
    );
    if (!allDone) return;

    if (step === "hero_generating") {
      const t = setTimeout(() => setStep("hero_preview"), 600);
      return () => clearTimeout(t);
    }
    if (step === "full_set_generating") {
      const t = setTimeout(() => setStep("results"), 600);
      return () => clearTimeout(t);
    }
  }, [jobs, step]);

  // Step transitions
  const handleFilesSelected = useCallback(
    (selected: File[], consentConfirmed: boolean, adultConfirmed: boolean) => {
      setFiles(selected);
      setFaceProcessingConsent(consentConfirmed);
      setAdultSubjectConfirmed(adultConfirmed);
      setStep("style");
    },
    []
  );

  const handleStyleConfirm = useCallback(
    async (s: StyleKey, g: Gender) => {
      setStyle(s);
      setGender(g);
      setError(null);

      try {
        const sess = await createSession(s, g);
        setSessionId(sess.session_id);
        setSession(sess);

        const updated = await uploadPhotos(
          sess.session_id,
          files,
          faceProcessingConsent,
          adultSubjectConfirmed,
        );
        setSession(updated);

        // Start hero preview instead of full generation
        const heroJobs = await startHeroPreview(sess.session_id);
        setJobs(heroJobs);
        setStep("hero_generating");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
      }
    },
    [adultSubjectConfirmed, faceProcessingConsent, files]
  );

  const handleMultiStyleConfirm = useCallback(
    async (styles: StyleKey[], g: Gender) => {
      setStyle(styles[0]);
      setGender(g);
      setError(null);

      try {
        const sess = await createSession(styles[0], g);
        setSessionId(sess.session_id);
        setSession(sess);

        const updated = await uploadPhotos(
          sess.session_id,
          files,
          faceProcessingConsent,
          adultSubjectConfirmed,
        );
        setSession(updated);

        // Multi-style: still generate a hero preview with the first selected style
        // so the user gets an Aha Moment before deciding to unlock the full bundle.
        const heroJobs = await startHeroPreview(sess.session_id, styles[0]);
        setJobs(heroJobs);
        setStep("hero_generating");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong");
      }
    },
    [adultSubjectConfirmed, faceProcessingConsent, files]
  );

  const handleUnlock = useCallback(async () => {
    if (!sessionId) return;
    setError(null);
    try {
      const fullSetJobs = await unlockFullSet(sessionId);
      setJobs(fullSetJobs);
      setStep("full_set_generating");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unlock failed");
    }
  }, [sessionId]);

  const handleRevise = useCallback(
    async (imageId: string, instruction: string) => {
      if (!sessionId) return;
      const job = await submitRevision(sessionId, imageId, instruction);
      setJobs((prev) => [...prev, job]);
      setStep("hero_generating");
    },
    [sessionId]
  );

  const handleTryAnotherStyle = useCallback(() => {
    setStep("style");
    setJobs([]);
    setError(null);
  }, []);

  const handleGuideContinue = useCallback(() => {
    setStep("upload");
  }, []);

  const handleStartOver = useCallback(() => {
    const sid = sessionId;
    setStep("guide");
    setSessionId(null);
    setSession(null);
    setJobs([]);
    setFiles([]);
    setFaceProcessingConsent(false);
    setStyle(null);
    setGender(null);
    setError(null);
    clearPersisted();
    if (sid) {
      deleteSession(sid).catch(() => {});
    }
  }, [sessionId]);

  const handleRefreshSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      const sess = await getSession(sessionId);
      setSession(sess);
    } catch {
      // ignore
    }
  }, [sessionId]);

  // Step labels for the indicator
  const stepLabels: Record<Step, string> = {
    guide: "Guide",
    upload: "Upload",
    style: "Style",
    hero_generating: "Preview",
    hero_preview: "Preview",
    full_set_generating: "Generating",
    results: "Results",
  };

  const stepOrder: Step[] = [
    "guide",
    "upload",
    "style",
    "hero_generating",
    "hero_preview",
    "full_set_generating",
    "results",
  ];

  const currentStepIndex = stepOrder.indexOf(step);

  return (
    <div className="py-16 px-6" suppressHydrationWarning>
      {/* Step indicator */}
      <div className="max-w-2xl mx-auto mb-12">
        <div className="flex items-center justify-center gap-2">
          {stepOrder.map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium ${
                  step === s
                    ? "bg-accent text-white"
                    : currentStepIndex > i
                    ? "bg-accent/20 text-accent"
                    : "bg-stone-200 text-stone-500"
                }`}
              >
                {i + 1}
              </div>
              {i < stepOrder.length - 1 && (
                <div
                  className={`w-12 h-0.5 ${
                    currentStepIndex > i ? "bg-accent" : "bg-stone-200"
                  }`}
                />
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-center gap-6 mt-2 text-xs text-stone-400">
          {stepOrder.map((s) => (
            <span key={s}>{stepLabels[s]}</span>
          ))}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="max-w-2xl mx-auto mb-6 p-4 rounded-xl bg-red-50 text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Steps */}
      {step === "guide" && <StepGuide onContinue={handleGuideContinue} />}

      {step === "upload" && (
        <StepUpload
          onFilesSelected={handleFilesSelected}
          onViewGuide={() => setStep("guide")}
        />
      )}

      {step === "style" && (
        <StepStyle
          onConfirm={handleStyleConfirm}
          onMultiConfirm={handleMultiStyleConfirm}
          onBack={() => setStep("upload")}
        />
      )}

      {(step === "hero_generating" || step === "full_set_generating") && (
        <StepGenerating
          jobs={jobs}
          progressByJob={progressByJob}
          connectionStatus={wsStatus}
          phase={step === "hero_generating" ? "hero" : "full_set"}
        />
      )}

      {step === "hero_preview" && session && (
        <StepHeroPreview
          session={session}
          onUnlock={handleUnlock}
          onRevise={handleRevise}
          onTryAnotherStyle={handleTryAnotherStyle}
          onStartOver={handleStartOver}
        />
      )}

      {step === "results" && session && (
        <StepResults
          session={session}
          onRevise={handleRevise}
          onStartOver={handleStartOver}
          onRefreshSession={handleRefreshSession}
        />
      )}
    </div>
  );
}
