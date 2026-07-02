"use client";

import { X, Check } from "lucide-react";

interface Template {
  id: string;
  gender: string;
  label: string;
  image: string;
}

interface Props {
  template: Template;
  styleLabel: string;
  onConfirm: () => void;
  onClose: () => void;
}

export function TemplatePreviewModal({
  template,
  styleLabel,
  onConfirm,
  onClose,
}: Props) {
  return (
    <div
      className="fixed inset-0 z-50 bg-stone-900/70 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl max-w-lg w-full overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-stone-100">
          <div>
            <p className="text-xs text-stone-400">{styleLabel}</p>
            <p className="text-sm font-medium text-stone-900">
              {template.label}
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 rounded-full bg-stone-100 flex items-center justify-center hover:bg-stone-200 transition-colors"
          >
            <X size={14} className="text-stone-500" />
          </button>
        </div>

        {/* Large image */}
        <div className="aspect-[3/4] bg-stone-50">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={template.image}
            alt={template.label}
            className="w-full h-full object-cover"
          />
        </div>

        {/* Actions */}
        <div className="p-4 flex items-center justify-between">
          <p className="text-xs text-stone-400">
            This is a style reference. Your final portrait is generated from your selfie.
          </p>
          <button
            onClick={onConfirm}
            className="h-10 px-6 rounded-full bg-accent text-white text-sm font-medium hover:bg-accent-hover transition-colors flex items-center gap-2"
          >
            <Check size={15} />
            Use this style
          </button>
        </div>
      </div>
    </div>
  );
}
