"use client";

import { Check, X, Camera, Sun, User, Users, Sparkles } from "lucide-react";

interface Props {
  onContinue: () => void;
}

const DO_ITEMS = [
  {
    icon: Sun,
    title: "光线充足、面部无阴影",
    desc: "自然光或柔和室内光最佳，避免头顶强光或背光导致脸黑。",
  },
  {
    icon: User,
    title: "正面或轻微侧面，五官清晰",
    desc: "至少一张正面照；可补充 45° 侧脸，但避免过度侧脸。",
  },
  {
    icon: Camera,
    title: "表情自然",
    desc: "闭着嘴或自然微笑都可以，避免夸张表情或闭眼。",
  },
  {
    icon: Sparkles,
    title: "摘掉眼镜、帽子、口罩",
    desc: "遮挡会干扰 AI 对人脸特征的判断。",
  },
  {
    icon: Check,
    title: "背景简单",
    desc: "纯色或简单背景，避免杂乱物体或其他人脸。",
  },
  {
    icon: Users,
    title: "至少 4 张，最好 4–6 张",
    desc: "包含不同角度和表情，帮助 AI 更稳定地识别你的脸。",
  },
];

const DONT_ITEMS = [
  "过度美颜、滤镜或磨皮",
  "多人合影",
  "闭眼、低头或侧脸过大",
  "昏暗、背光或噪点过多",
  "戴墨镜、口罩、帽子",
  "翻拍屏幕或低分辨率图片",
];

export function StepGuide({ onContinue }: Props) {
  return (
    <div className="max-w-2xl mx-auto">
      <h2 className="text-3xl font-semibold tracking-tight text-center">
        拍出合格的自拍
      </h2>
      <p className="mt-3 text-stone-500 text-center">
        上传质量直接决定最终肖像是否像你。花 30 秒看完，能大幅提升成功率。
      </p>

      {/* Do's grid */}
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
        {DO_ITEMS.map((item) => (
          <div
            key={item.title}
            className="rounded-xl border border-stone-200 bg-white p-4"
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5 shrink-0 w-8 h-8 rounded-lg bg-accent/10 text-accent flex items-center justify-center">
                <item.icon size={18} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-stone-900">
                  {item.title}
                </h3>
                <p className="mt-1 text-sm text-stone-500">{item.desc}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Don'ts panel */}
      <div className="mt-4 rounded-xl border border-red-100 bg-red-50/50 p-5">
        <h3 className="text-sm font-semibold text-red-900 flex items-center gap-2">
          <X size={16} />
          这些会降低成功率
        </h3>
        <ul className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
          {DONT_ITEMS.map((text) => (
            <li key={text} className="flex items-start gap-2 text-sm text-red-800">
              <X size={14} className="mt-0.5 shrink-0" />
              {text}
            </li>
          ))}
        </ul>
      </div>

      {/* Example guidance */}
      <div className="mt-4 rounded-xl border border-stone-200 bg-stone-50 p-5 text-sm text-stone-600">
        <strong className="text-stone-900">推荐组合：</strong>
        <span className="ml-1">
          正面 neutral → 正面微笑 → 左侧 45° → 右侧 45° → 自然生活照 → 可选侧面。
        </span>
      </div>

      {/* CTA */}
      <div className="mt-8 flex justify-center">
        <button
          onClick={onContinue}
          className="h-12 px-8 rounded-full bg-stone-900 text-white text-sm font-medium hover:bg-stone-800 transition-colors"
        >
          我已了解，开始上传
        </button>
      </div>
    </div>
  );
}
