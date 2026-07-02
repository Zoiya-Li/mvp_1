"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  Upload,
  X,
  ImagePlus,
  ChevronDown,
  ChevronUp,
  Sun,
  Smile,
  User,
  Users,
  Glasses,
  Sparkles,
  AlertTriangle,
  Check,
} from "lucide-react";

interface Props {
  onFilesSelected: (
    files: File[],
    faceProcessingConsent: boolean,
    adultSubjectConfirmed: boolean
  ) => void;
  onViewGuide?: () => void;
  disabled?: boolean;
}

const REFERENCE_SLOTS = [
  "Front",
  "Smile",
  "Left 45",
  "Right 45",
  "Lifestyle",
  "Profile",
];
const MAX_PHOTOS = 6;

const DO_TIPS = [
  {
    icon: Sun,
    title: "Good lighting",
    desc: "Face the light source. Avoid harsh shadows or backlighting.",
  },
  {
    icon: Smile,
    title: "Natural expression",
    desc: "Relaxed smile or neutral expression. Avoid forced poses.",
  },
  {
    icon: User,
    title: "Clear face view",
    desc: "Front and 45° angles show your facial features best.",
  },
  {
    icon: Check,
    title: "Plain background",
    desc: "Simple backgrounds help the AI focus on your face.",
  },
];

const DONT_TIPS = [
  {
    icon: Glasses,
    title: "No sunglasses",
    desc: "Eyewear that hides your eyes reduces identity accuracy.",
  },
  {
    icon: Users,
    title: "No group photos",
    desc: "Only upload photos with one person (you) in the frame.",
  },
  {
    icon: Sparkles,
    title: "No heavy filters",
    desc: "Avoid beauty filters that alter your face shape or skin texture.",
  },
  {
    icon: AlertTriangle,
    title: "No blurry shots",
    desc: "Out-of-focus or motion-blurred photos hurt generation quality.",
  },
];

export function StepUpload({ onFilesSelected, disabled }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [dragging, setDragging] = useState(false);
  const [faceProcessingConsent, setFaceProcessingConsent] = useState(false);
  const [adultSubjectConfirmed, setAdultSubjectConfirmed] = useState(false);
  const [showGuide, setShowGuide] = useState(false);

  const objectUrlsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const urls = objectUrlsRef.current;
    return () => {
      for (const u of urls) URL.revokeObjectURL(u);
      urls.clear();
    };
  }, []);

  const addFiles = useCallback(
    (incoming: FileList | File[]) => {
      const MAX_SIZE = 10 * 1024 * 1024;
      const valid = Array.from(incoming).filter(
        (f) => f.type.startsWith("image/") && f.size <= MAX_SIZE
      );
      if (valid.length === 0) return;

      const combinedFiles = [...files, ...valid].slice(0, MAX_PHOTOS);
      setFiles(combinedFiles);

      const alreadyHave = files.length;
      const newPreviews: string[] = [];
      for (let i = 0; i < combinedFiles.length - alreadyHave; i++) {
        const url = URL.createObjectURL(combinedFiles[alreadyHave + i]);
        objectUrlsRef.current.add(url);
        newPreviews.push(url);
      }
      setPreviews((prev) => [...prev, ...newPreviews].slice(0, combinedFiles.length));
    },
    [files]
  );

  const removeFile = (index: number) => {
    const url = previews[index];
    if (url) {
      URL.revokeObjectURL(url);
      objectUrlsRef.current.delete(url);
    }
    setFiles(files.filter((_, i) => i !== index));
    setPreviews(previews.filter((_, i) => i !== index));
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      addFiles(e.dataTransfer.files);
    },
    [addFiles, disabled]
  );

  const hasEnoughPhotos = files.length >= 4;
  const canProceed = hasEnoughPhotos && faceProcessingConsent && adultSubjectConfirmed;

  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-3xl font-semibold tracking-tight text-center">
        Upload your selfies to start your portrait studio
      </h2>
      <p className="mt-3 text-stone-500 text-center">
        Upload 4–6 everyday photos in this order: front, smile, left angle, right angle,
        then lifestyle or profile.
        <span className="text-amber-700 font-medium"> The first four must be clear</span>
        ; they become the main identity references.
      </p>

      {/* Photo quality guide — collapsible */}
      <div className="mt-4">
        <button
          onClick={() => setShowGuide(!showGuide)}
          className="mx-auto flex items-center gap-1 text-sm text-accent hover:text-accent/80 transition-colors"
        >
          {showGuide ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          {showGuide ? "Hide photo tips" : "How to take good reference photos"}
        </button>

        {showGuide && (
          <div className="mt-4 rounded-2xl border border-stone-200 bg-stone-50/50 p-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Do's */}
              <div>
                <h3 className="text-sm font-semibold text-emerald-700 mb-3 flex items-center gap-2">
                  <Check size={16} />
                  Do
                </h3>
                <div className="space-y-3">
                  {DO_TIPS.map((tip) => (
                    <div key={tip.title} className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-emerald-50 flex items-center justify-center shrink-0">
                        <tip.icon size={16} className="text-emerald-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-stone-800">{tip.title}</p>
                        <p className="text-xs text-stone-500">{tip.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Don'ts */}
              <div>
                <h3 className="text-sm font-semibold text-rose-700 mb-3 flex items-center gap-2">
                  <AlertTriangle size={16} />
                  Don&apos;t
                </h3>
                <div className="space-y-3">
                  {DONT_TIPS.map((tip) => (
                    <div key={tip.title} className="flex items-start gap-3">
                      <div className="w-8 h-8 rounded-lg bg-rose-50 flex items-center justify-center shrink-0">
                        <tip.icon size={16} className="text-rose-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-stone-800">{tip.title}</p>
                        <p className="text-xs text-stone-500">{tip.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`mt-6 border-2 border-dashed rounded-2xl p-10 text-center transition-colors ${
          dragging
            ? "border-accent bg-accent/5"
            : "border-stone-300 hover:border-stone-400"
        } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
      >
        <ImagePlus size={40} className="mx-auto text-stone-400 mb-4" />
        <p className="text-stone-600 font-medium">
          Drag and drop photos here, or click to browse
        </p>
        <p className="text-stone-400 text-sm mt-1">
          Supports JPG / PNG / WebP, up to 10MB each
        </p>
        <label className="mt-4 inline-flex items-center gap-2 h-10 px-5 rounded-full bg-stone-100 text-stone-700 text-sm font-medium cursor-pointer hover:bg-stone-200 transition-colors">
          <Upload size={16} />
          Select photos
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
            disabled={disabled}
          />
        </label>
      </div>

      {/* Thumbnails */}
      {previews.length > 0 && (
        <div className="mt-6 grid grid-cols-4 gap-3">
          {previews.map((src, i) => (
            <div
              key={src}
              className="relative aspect-square rounded-xl overflow-hidden bg-stone-100 group"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={src}
                alt={`Photo ${i + 1}`}
                className="object-cover w-full h-full"
              />
              <span className="absolute top-2 left-2 bg-accent text-white text-xs px-2 py-0.5 rounded-full">
                {REFERENCE_SLOTS[i] ?? `Photo ${i + 1}`}
              </span>
              {!disabled && (
                <button
                  onClick={() => removeFile(i)}
                  className="absolute top-2 right-2 w-6 h-6 rounded-full bg-stone-900/60 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X size={14} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Consent checkboxes */}
      <label
        className={`mt-5 flex items-start gap-3 rounded-lg border border-stone-200 bg-white p-4 text-sm text-stone-700 ${
          disabled ? "opacity-50" : "cursor-pointer"
        }`}
      >
        <input
          type="checkbox"
          checked={faceProcessingConsent}
          onChange={(e) => setFaceProcessingConsent(e.target.checked)}
          disabled={disabled}
          className="mt-1 h-4 w-4 rounded border-stone-300 text-accent focus:ring-accent"
        />
        <span>
          I confirm these photos are of me or I have permission to use them, and I
          agree that FlashShot may temporarily analyze facial features only for
          this portrait task.{" "}
          <a href="/privacy" target="_blank" className="text-accent underline">
            Privacy Policy
          </a>
          {" / "}
          <a href="/terms" target="_blank" className="text-accent underline">
            Terms of Service
          </a>
          .
        </span>
      </label>

      <label
        className={`mt-3 flex items-start gap-3 rounded-lg border border-stone-200 bg-white p-4 text-sm text-stone-700 ${
          disabled ? "opacity-50" : "cursor-pointer"
        }`}
      >
        <input
          type="checkbox"
          checked={adultSubjectConfirmed}
          onChange={(e) => setAdultSubjectConfirmed(e.target.checked)}
          disabled={disabled}
          className="mt-1 h-4 w-4 rounded border-stone-300 text-accent focus:ring-accent"
        />
        <span>
          I confirm the person in these photos is an adult and this session is
          for single-person static portraits only.
        </span>
      </label>

      {/* Count + CTA */}
      <div className="mt-6 flex items-center justify-between">
        <p className="text-sm text-stone-500">
          {files.length} / {MAX_PHOTOS} photos selected
        </p>
        <button
          onClick={() =>
            canProceed &&
            onFilesSelected(files, faceProcessingConsent, adultSubjectConfirmed)
          }
          disabled={!canProceed || disabled}
          className="h-11 px-7 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {hasEnoughPhotos
            ? faceProcessingConsent
              ? adultSubjectConfirmed
                ? "Next: choose a style"
                : "Confirm adult subject"
              : "Confirm photo authorization"
            : "Add at least 4 photos"}
        </button>
      </div>
    </div>
  );
}
