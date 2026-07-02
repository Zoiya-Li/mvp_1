"use client";

import { Loader2, WifiOff, RefreshCw, ScanFace, Check } from "lucide-react";
import type { JobResponse, GenerationPhase, GenerationProgress } from "@/lib/types";
import type { WSConnectionStatus } from "@/lib/websocket";

interface Props {
  jobs: JobResponse[];
  /** Live resemblance-agent progress per job, keyed by job_id. */
  progressByJob?: Record<string, GenerationProgress>;
  /** Live WebSocket connection state — used to surface network trouble. */
  connectionStatus?: WSConnectionStatus;
  /** Which generation phase we're in. */
  phase?: "hero" | "full_set";
}

// Production-pipeline phase → chip label + tone. New jobs emit the controlled
// candidate pipeline phases; legacy resemblance-loop phases are kept so older
// in-flight jobs still display correctly.
const PHASE_CHIP: Record<
  GenerationPhase,
  { label: string; tone: "active" | "warn" | "ok" | "muted" }
> = {
  candidate_generating: { label: "Candidate", tone: "active" },
  candidate_judging: { label: "QA scoring", tone: "active" },
  repairing: { label: "Repairing", tone: "warn" },
  generating: { label: "Initial", tone: "active" },
  judging: { label: "Scoring", tone: "active" },
  revising: { label: "Revising", tone: "warn" },
  accepted: { label: "Accepted", tone: "ok" },
  max_reached: { label: "Max reached", tone: "muted" },
};

function chipToneClass(tone: "active" | "warn" | "ok" | "muted") {
  switch (tone) {
    case "active":
      return "bg-accent/10 text-accent";
    case "warn":
      return "bg-amber-50 text-amber-600";
    case "ok":
      return "bg-emerald-50 text-emerald-600";
    default:
      return "bg-stone-200 text-stone-500";
  }
}

function jobTitle(job: JobResponse, progress?: GenerationProgress) {
  return (
    progress?.shot_spec?.shot_label ??
    job.shot_spec?.shot_label ??
    job.shot_spec?.shot_id ??
    job.prompt_id ??
    "Generating"
  );
}

function jobSubtitle(job: JobResponse) {
  const template = job.shot_spec?.template_label;
  const style = job.shot_spec?.style_label;
  if (template && style) return `${style} · ${template}`;
  return template ?? style ?? null;
}

// Human label for each non-OK connection state. "connected" renders nothing
// (it's the expected norm — no need to clutter the UI).
function connectionBanner(status: WSConnectionStatus) {
  switch (status) {
    case "connecting":
      return { text: "Connecting to server…", tone: "info" as const };
    case "reconnecting":
      return { text: "Connection lost — reconnecting…", tone: "warn" as const };
    case "disconnected":
      return { text: "Connection lost — please refresh", tone: "error" as const };
    default:
      return null;
  }
}

export function StepGenerating({ jobs, progressByJob = {}, connectionStatus, phase }: Props) {
  const completed = jobs.filter((j) => j.status === "completed").length;
  const total = jobs.length;
  const failed = jobs.filter((j) => j.status === "failed").length;
  const anyPipelineProgress = jobs.some(
    (j) => j.status === "processing" && progressByJob[j.job_id]
  );

  const banner =
    connectionStatus && connectionStatus !== "connected"
      ? connectionBanner(connectionStatus)
      : null;

  const isHero = phase === "hero";

  return (
    <div className="max-w-lg mx-auto text-center">
      <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-accent/10 text-accent mb-6">
        <Loader2 size={32} className="animate-spin" />
      </div>

      <h2 className="text-3xl font-semibold tracking-tight">
        {isHero
          ? "Generating your preview portrait"
          : "Generating your portrait collection"}
      </h2>
      <p className="mt-3 text-stone-500">
        {isHero
          ? "We're creating a single high-quality preview portrait optimized for identity accuracy. This usually takes under a minute."
          : anyPipelineProgress
          ? "We're generating multiple candidates, running automated quality checks and identity repair, then delivering only the best ones."
          : "Each portrait goes through candidate generation, quality scoring, and curated delivery — this usually takes a minute or two."}
      </p>

      {/* Connection status — only when something is off. */}
      {banner && (
        <div
          className={`mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs ${
            banner.tone === "error"
              ? "bg-red-50 text-red-600"
              : banner.tone === "warn"
              ? "bg-amber-50 text-amber-600"
              : "bg-stone-100 text-stone-500"
          }`}
        >
          {banner.tone === "error" ? (
            <WifiOff size={13} />
          ) : (
            <RefreshCw size={13} className="animate-spin" />
          )}
          {banner.text}
        </div>
      )}

      {/* Progress */}
      <div className="mt-8 space-y-3">
        {jobs.map((job) => {
          const prog = progressByJob[job.job_id];
          const showResemblance = job.status === "processing" && prog;
          const chip = prog ? PHASE_CHIP[prog.phase] : null;
          const subtitle = jobSubtitle(job);
          return (
            <div
              key={job.job_id}
              className="px-4 py-3 rounded-xl bg-stone-50"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {job.status === "processing" && (
                    <Loader2 size={16} className="animate-spin text-accent" />
                  )}
                  {job.status === "completed" && (
                    <span className="text-accent text-sm">✓</span>
                  )}
                  {job.status === "failed" && (
                    <span className="text-red-500 text-sm">✗</span>
                  )}
                  {job.status === "queued" && (
                    <span className="text-stone-400 text-sm">○</span>
                  )}
                  <span className="min-w-0 text-left">
                    <span className="block truncate text-sm text-stone-700">
                      {jobTitle(job, prog)}
                    </span>
                    {subtitle ? (
                      <span className="block truncate text-[11px] text-stone-400">
                        {subtitle}
                      </span>
                    ) : null}
                  </span>
                </div>
                <span
                  className={`text-xs ${
                    job.status === "processing"
                      ? "text-accent"
                      : job.status === "failed"
                      ? "text-red-500"
                      : "text-stone-400"
                  }`}
                >
                  {job.status === "processing"
                    ? "Generating…"
                    : job.status === "queued"
                    ? (job.position_in_queue ? `In queue (#${job.position_in_queue})` : "In queue")
                    : job.status === "failed"
                    ? "Failed"
                    : "Done"}
                </span>
              </div>

              {/* Production-pipeline status — only while this job is actively
                  processing and we have a progress frame for it. Shows the
                  backend phase plus the detail string. */}
              {showResemblance && prog && chip && (
                <div className="mt-2 flex items-center gap-2 text-xs text-stone-500">
                  <ScanFace size={13} className="text-accent shrink-0" />
                  {prog.phase === "accepted" ? (
                    <Check size={13} className="text-emerald-600 shrink-0" />
                  ) : null}
                  <span
                    className={`px-2 py-0.5 rounded-full font-medium ${chipToneClass(
                      chip.tone
                    )}`}
                  >
                    {chip.label}
                    {prog.iteration > 0
                      ? ` · ${prog.iteration}/${prog.max_iterations}`
                      : ""}
                  </span>
                  {prog.detail && (
                    <span className="truncate">{prog.detail}</span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <p className="mt-4 text-sm text-stone-400">
        {completed} / {total} done
        {failed > 0 && <span className="text-red-500"> · {failed} failed</span>}
      </p>
    </div>
  );
}
