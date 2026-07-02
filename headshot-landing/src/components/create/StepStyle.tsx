"use client";

import { useState } from "react";
import { ArrowLeft, Check } from "lucide-react";
import type { StyleKey, Gender } from "@/lib/types";
import { TemplatePreviewModal } from "./TemplatePreviewModal";

/* ------------------------------------------------------------------ */
/*  Style data — portrait studio themes (mirrors prompts.json v4)                */
/* ------------------------------------------------------------------ */

interface Template {
  id: string;
  gender: Gender;
  label: string;
  image: string;
}

interface StyleGroup {
  key: StyleKey;
  category: string;
  desc: string;
  templates: Template[];
}

const STYLE_GROUPS: StyleGroup[] = [
  {
    key: "chinese_style",
    category: "Traditional Chinese",
    desc: "Hanfu, Qipao, neo-Chinese — Eastern aesthetics",
    templates: [
      { id: "gf_m_hanfu", gender: "male", label: "Hanfu · literati scholar", image: "/images/gf_m_hanfu.png" },
      { id: "gf_f_qipao", gender: "female", label: "Qipao · Republican-era charm", image: "/images/gf_f_qipao.png" },
    ],
  },
  {
    key: "jk_portrait",
    category: "Japanese · Korean",
    desc: "Fresh youth / minimalist premium",
    templates: [
      { id: "jp_m_fresh", gender: "male", label: "Japanese fresh · boyish charm", image: "/images/jp_m_fresh.png" },
      { id: "jp_f_fresh", gender: "female", label: "Japanese fresh · airy softness", image: "/images/jp_f_fresh.png" },
      { id: "kr_m_minimal", gender: "male", label: "Korean minimal · premium grey", image: "/images/kr_m_minimal.png" },
      { id: "kr_f_elegant", gender: "female", label: "Korean elegant · silk texture", image: "/images/kr_f_elegant.png" },
    ],
  },
  {
    key: "cinematic",
    category: "Cinematic",
    desc: "Cyberpunk, story lighting, dark gothic",
    templates: [
      { id: "film_m_cyber", gender: "male", label: "Cyberpunk · neon nightscape", image: "/images/film_m_cyber.png" },
      { id: "film_f_cinematic", gender: "female", label: "Cinematic · story lighting", image: "/images/film_f_cinematic.png" },
      { id: "film_f_dark", gender: "female", label: "Dark gothic · mysterious mood", image: "/images/film_f_dark.png" },
    ],
  },
  {
    key: "creative",
    category: "Retro · Creative",
    desc: "Hong Kong film, natural forest, sport energy",
    templates: [
      { id: "cr_m_retro", gender: "male", label: "Retro film · Hong Kong Retro", image: "/images/cr_m_retro.png" },
      { id: "cr_f_forest", gender: "female", label: "Natural · fresh outdoors", image: "/images/cr_f_forest.png" },
      { id: "cr_m_sport", gender: "male", label: "Sport energy · sunny athlete", image: "/images/cr_m_sport.png" },
    ],
  },
  {
    key: "social",
    category: "Everyday Lifestyle",
    desc: "French ease, street style, cafe moments",
    templates: [
      { id: "lw_m_01", gender: "male", label: "Urban rooftop · golden sunset", image: "/images/lw_m_01.png" },
      { id: "lw_m_02", gender: "male", label: "Knit sweater · cafe vibe", image: "/images/lw_m_02.png" },
      { id: "lw_f_01", gender: "female", label: "Cream knit · soft light", image: "/images/lw_f_01.png" },
      { id: "social_f_french", gender: "female", label: "French · effortless elegance", image: "/images/social_f_french.png" },
      { id: "social_m_street", gender: "male", label: "Street style · trend-forward", image: "/images/social_m_street.png" },
    ],
  },
  {
    key: "fashion",
    category: "Editorial",
    desc: "Magazine-cover, high contrast, cool tones",
    templates: [
      { id: "fz_m_editorial", gender: "male", label: "Editorial · high contrast", image: "/images/fz_m_editorial.png" },
      { id: "fz_f_editorial", gender: "female", label: "Editorial · cool premium", image: "/images/fz_f_editorial.png" },
    ],
  },
  {
    key: "business",
    category: "Business Elite",
    desc: "Professional headshots, polished business portraits",
    templates: [
      { id: "bf_m_01", gender: "male", label: "Classic business · navy suit", image: "/images/bf_m_01.png" },
      { id: "bf_m_03", gender: "male", label: "Black suit · gold-rim glasses", image: "/images/bf_m_03.png" },
      { id: "bf_m_tech", gender: "male", label: "Tech / startup · smart casual", image: "/images/bf_m_tech.png" },
      { id: "bf_m_04", gender: "male", label: "Modern business · emerald suit", image: "/images/bf_m_04.png" },
      { id: "bf_f_01", gender: "female", label: "Classic business · beige suit", image: "/images/bf_f_01.png" },
      { id: "bf_f_03", gender: "female", label: "Classic business · black dress", image: "/images/bf_f_03.png" },
      { id: "bf_f_tech", gender: "female", label: "Tech / startup · modern feel", image: "/images/bf_f_tech.png" },
      { id: "bf_f_04", gender: "female", label: "Modern business · pink suit", image: "/images/bf_f_04.png" },
    ],
  },
  {
    key: "academic",
    category: "Academic & Artistic",
    desc: "Scholarly, warm and intellectual portraits",
    templates: [
      { id: "ac_m_01", gender: "male", label: "Casual blazer · scholarly air", image: "/images/ac_m_01.png" },
      { id: "ac_f_01", gender: "female", label: "Cardigan · gentle and intellectual", image: "/images/ac_f_01.png" },
    ],
  },
  {
    key: "id_photo",
    category: "ID Photo",
    desc: "Red / blue / white background standard ID photos",
    templates: [
      { id: "id_m_red", gender: "male", label: "Red background · standard ID", image: "/images/id_m_red.png" },
      { id: "id_m_blue", gender: "male", label: "Blue background · standard ID", image: "/images/id_m_blue.png" },
      { id: "id_f_white", gender: "female", label: "White background · visa / passport", image: "/images/id_f_white.png" },
      { id: "id_f_red", gender: "female", label: "Red background · standard ID", image: "/images/id_f_red.png" },
    ],
  },
];

type Mode = "single" | "multi";

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

interface Props {
  onConfirm: (style: StyleKey, gender: Gender) => void;
  onMultiConfirm?: (styles: StyleKey[], gender: Gender) => void;
  onBack: () => void;
  disabled?: boolean;
}

export function StepStyle({ onConfirm, onMultiConfirm, onBack, disabled }: Props) {
  const [mode, setMode] = useState<Mode>("single");
  const [gender, setGender] = useState<Gender>("female");
  const [selectedStyle, setSelectedStyle] = useState<StyleKey | null>(null);
  const [multiSelected, setMultiSelected] = useState<StyleKey[]>([]);
  const [previewTemplate, setPreviewTemplate] = useState<{
    template: Template;
    group: StyleGroup;
  } | null>(null);

  const toggleMulti = (key: StyleKey) => {
    setMultiSelected((prev) => {
      if (prev.includes(key)) return prev.filter((k) => k !== key);
      if (prev.length >= 4) return prev; // max 4
      return [...prev, key];
    });
  };

  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={onBack}
        className="flex items-center gap-1 text-stone-500 text-sm hover:text-stone-700 mb-6"
      >
        <ArrowLeft size={16} />
        Back to upload
      </button>

      <h2 className="text-3xl font-semibold tracking-tight text-center">
        Choose a style
      </h2>
      <p className="text-center text-stone-400 text-sm mt-2">
        {mode === "single"
          ? "Pick a style you love and the AI will generate several artistic portraits in that theme."
          : "Pick 2–4 styles to bundle — the AI generates portraits for each so you can compare and choose."}
      </p>

      {/* Mode toggle */}
      <div className="mt-6 flex justify-center">
        <div className="inline-flex rounded-full bg-stone-100 p-1">
          {(["single", "multi"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m);
                setSelectedStyle(null);
                setMultiSelected([]);
              }}
              className={`px-5 py-1.5 rounded-full text-sm font-medium transition-colors ${
                mode === m
                  ? "bg-white text-stone-900 shadow-sm"
                  : "text-stone-500 hover:text-stone-700"
              }`}
            >
              {m === "single" ? "Single style" : "Style bundle"}
            </button>
          ))}
        </div>
      </div>

      {/* Gender toggle */}
      <div className="mt-4 flex justify-center">
        <div className="inline-flex rounded-full bg-stone-100 p-1">
          {(["female", "male"] as Gender[]).map((g) => (
            <button
              key={g}
              onClick={() => {
                setGender(g);
                setSelectedStyle(null);
              }}
              className={`px-5 py-1.5 rounded-full text-sm font-medium transition-colors ${
                gender === g
                  ? "bg-white text-stone-900 shadow-sm"
                  : "text-stone-500 hover:text-stone-700"
              }`}
            >
              {g === "female" ? "Women" : "Men"}
            </button>
          ))}
        </div>
      </div>

      {/* Style groups with template galleries */}
      <div className="mt-10 space-y-10">
        {STYLE_GROUPS.map((group) => {
          const filtered = group.templates.filter(
            (t) => t.gender === gender
          );
          if (filtered.length === 0) return null;

          const isActive = mode === "single" && selectedStyle === group.key;
          const isMultiChecked = multiSelected.includes(group.key);

          return (
            <div key={group.key}>
              {/* Group header */}
              <button
                onClick={() => {
                  if (mode === "single") {
                    setSelectedStyle(isActive ? null : group.key);
                  } else {
                    toggleMulti(group.key);
                  }
                }}
                className="w-full flex items-center justify-between group"
              >
                <div className="flex items-center gap-3">
                  <h3 className="text-lg font-semibold text-stone-900">
                    {group.category}
                  </h3>
                  <span className="text-xs text-stone-400">
                    {group.desc}
                  </span>
                </div>
                <div
                  className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                    (mode === "single" && isActive) || (mode === "multi" && isMultiChecked)
                      ? "border-accent bg-accent text-white"
                      : "border-stone-300 group-hover:border-stone-400"
                  }`}
                >
                  {(mode === "single" && isActive) && <Check size={12} />}
                  {mode === "multi" && isMultiChecked && (
                    <span className="text-xs font-bold">{multiSelected.indexOf(group.key) + 1}</span>
                  )}
                </div>
              </button>

              {/* Template gallery */}
              <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {filtered.map((tmpl) => (
                  <TemplateCard
                    key={tmpl.id}
                    template={tmpl}
                    isSelected={mode === "single" ? isActive : isMultiChecked}
                    onPreview={() => setPreviewTemplate({ template: tmpl, group })}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* CTA */}
      <div className="mt-10 flex justify-center">
        {mode === "single" ? (
          <button
            onClick={() => selectedStyle && onConfirm(selectedStyle, gender)}
            disabled={!selectedStyle || disabled}
            className="h-12 px-8 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Generate portraits
          </button>
        ) : (
          <button
            onClick={() => multiSelected.length >= 2 && onMultiConfirm?.(multiSelected, gender)}
            disabled={multiSelected.length < 2 || disabled}
            className="h-12 px-8 rounded-full bg-accent text-white font-medium text-sm hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {multiSelected.length >= 2 ? `Generate ${multiSelected.length} styles` : "Select at least 2 styles"}
          </button>
        )}
      </div>

      {/* Template preview modal */}
      {previewTemplate && (
        <TemplatePreviewModal
          template={previewTemplate.template}
          styleLabel={`${previewTemplate.group.category} · ${previewTemplate.group.desc}`}
          onConfirm={() => {
            if (mode === "single") {
              setSelectedStyle(previewTemplate.group.key);
            } else {
              toggleMulti(previewTemplate.group.key);
            }
            setPreviewTemplate(null);
          }}
          onClose={() => setPreviewTemplate(null)}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Template card — individual style preview                           */
/* ------------------------------------------------------------------ */

function TemplateCard({
  template,
  isSelected,
  onPreview,
}: {
  template: Template;
  isSelected: boolean;
  onPreview: () => void;
}) {
  return (
    <div
      onClick={onPreview}
      className={`relative aspect-[3/4] rounded-xl overflow-hidden transition-all cursor-pointer group/card ${
        isSelected ? "ring-2 ring-accent/30" : "hover:ring-2 hover:ring-stone-200"
      }`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={template.image}
        alt={template.label}
        className="object-cover w-full h-full transition-transform duration-300 group-hover/card:scale-105"
      />
      {/* Label overlay at bottom */}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent px-3 py-2">
        <p className="text-white text-xs font-medium truncate">
          {template.label}
        </p>
      </div>
    </div>
  );
}
