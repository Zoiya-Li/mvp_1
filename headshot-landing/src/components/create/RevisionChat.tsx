"use client";

import { useState } from "react";
import { Send, Loader2 } from "lucide-react";
import type { GeneratedImage } from "@/lib/types";

/* ------------------------------------------------------------------ */
/*  Quick revision presets                                             */
/* ------------------------------------------------------------------ */

const QUICK_REVISIONS = [
  { label: "More natural", prompt: "Make the result more natural and realistic, reducing the AI look" },
  { label: "Sharper", prompt: "Improve clarity and detail, make the photo crisper" },
  { label: "Relaxed expression", prompt: "Make the expression more relaxed and the smile more natural" },
  { label: "Less smoothing", prompt: "Reduce skin smoothing, keep more skin texture and realism" },
  { label: "More like me", prompt: "Restore the facial features from the reference photo to better match the person" },
  { label: "Brighter light", prompt: "Brighten the overall lighting so the face is more lit" },
];

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

interface Props {
  image: GeneratedImage;
  revisionsUsed: number;
  maxRevisions: number;
  onSubmit: (imageId: string, instruction: string) => Promise<void>;
  disabled?: boolean;
}

interface RevisionEntry {
  instruction: string;
  image: GeneratedImage | null;
  loading: boolean;
}

export function RevisionChat({
  image,
  revisionsUsed,
  maxRevisions,
  onSubmit,
  disabled,
}: Props) {
  const [entries, setEntries] = useState<RevisionEntry[]>([]);
  const [input, setInput] = useState("");
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const remaining = maxRevisions - revisionsUsed;

  const handleSubmit = async () => {
    if (!input.trim() || remaining <= 0 || disabled) return;
    const instruction = input.trim();
    setInput("");
    setActiveTag(null);

    const entry: RevisionEntry = {
      instruction,
      image: null,
      loading: true,
    };
    setEntries((prev) => [...prev, entry]);

    try {
      await onSubmit(image.image_id, instruction);
      // The parent will update generated_images via WebSocket;
      // we just show the instruction was sent.
      setEntries((prev) =>
        prev.map((e, i) =>
          i === prev.length - 1 ? { ...e, loading: false } : e
        )
      );
    } catch {
      setEntries((prev) =>
        prev.map((e, i) =>
          i === prev.length - 1 ? { ...e, loading: false } : e
        )
      );
    }
  };

  const handleTagClick = (tag: (typeof QUICK_REVISIONS)[number]) => {
    if (activeTag === tag.label) {
      // Deselect
      setInput("");
      setActiveTag(null);
    } else {
      setInput(tag.prompt);
      setActiveTag(tag.label);
    }
  };

  return (
    <div className="border-t border-stone-200 mt-3 pt-3">
      <p className="text-xs text-stone-400 mb-3">
        {remaining} {remaining === 1 ? "revision" : "revisions"} left
      </p>

      {/* Revision history */}
      {entries.map((entry, i) => (
        <div key={i} className="mb-2 flex items-start gap-2">
          <span className="text-xs text-stone-400 mt-0.5 shrink-0">
            Revision {i + 1}:
          </span>
          <span className="text-sm text-stone-600">{entry.instruction}</span>
          {entry.loading && (
            <Loader2 size={14} className="animate-spin text-accent shrink-0" />
          )}
        </div>
      ))}

      {/* Quick revision tags */}
      {remaining > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {QUICK_REVISIONS.map((tag) => (
            <button
              key={tag.label}
              onClick={() => handleTagClick(tag)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                activeTag === tag.label
                  ? "bg-accent text-white"
                  : "bg-stone-100 text-stone-600 hover:bg-stone-200"
              }`}
              disabled={disabled}
            >
              {tag.label}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      {remaining > 0 && (
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              // If user edits manually, deselect active tag
              if (activeTag) setActiveTag(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            placeholder="Pick a local retouch, or type clarity / light / expression"
            className="flex-1 h-9 px-3 rounded-lg border border-stone-200 text-sm focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent"
            disabled={disabled}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || disabled}
            className="h-9 px-3 rounded-lg bg-accent text-white flex items-center justify-center hover:bg-accent-hover transition-colors disabled:opacity-40"
          >
            <Send size={14} />
          </button>
        </div>
      )}
    </div>
  );
}
