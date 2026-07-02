#!/usr/bin/env python3
"""A/B experiment: generation vs. editing for identity-preserving portraits.

Tests multiple OpenRouter image models under two prompt framings:
  - "generation": template first, user selfie second (current production style).
  - "editing":    user selfie first, template second (explicit edit instruction).

Each output is scored by a strict Chinese identity judge (the same prompt used
in production) so results are directly comparable to the live pipeline.

Run:
  cd headshot_pipeline
  python experiments/compare_identity_preservation.py

Outputs are written to experiments/output/ and a JSON/CSV report.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

# Load .env if present.
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "templates"
OUTPUT_DIR = ROOT / "experiments" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "report.json"
REPORT_CSV = OUTPUT_DIR / "report.csv"

# ── Judge prompt (identical to production) ──────────────────────────────
JUDGE_PROMPT = """\
你现在是严格的人脸身份审核系统。请像证件照比对或人脸识别一样，严格判断第一张图片（AI生成）中的人是否是第二张图片（用户本人参考照片）中的同一个人。

不要宽容、不要给面子。AI生成的图片常常会有"看起来像但仔细一看不是本人"的问题，你必须抓出来。

请逐项核对以下面部特征，任何一项有明显差异，评分就不能超过6分：
1. 脸型轮廓（圆脸/瓜子脸/方脸/长脸，下颌线角度）
2. 眼睛（大小、形状、单双眼皮、眼距、眼角形状）
3. 鼻子（鼻梁高度、鼻翼宽度、鼻头形状、鼻孔露出程度）
4. 嘴巴（嘴唇厚度、嘴角上扬/下垂、笑容弧度、牙齿露出情况）
5. 眉毛（粗细、弧度、眉峰位置、眉头间距）
6. 发型与发际线（头发长度、卷曲度、分线方向、发际线高低）
7. 肤色与肤质
8. 明显标记（痣、疤痕、眼镜款式、胡须、酒窝等）

评分标准（非常严格）：
- 10分：完全一致，像双胞胎
- 8-9分：明显是同一个人，只是角度/表情/光线不同
- 6-7分：有点像，但熟人可能需要多看几眼才能确认
- 5分及以下：不像，或者明显不是同一个人

输出格式必须如下：
评分：X/10
判断理由：...

如果评分低于8分，请在"判断理由"中具体列出差异最大的2-3个面部特征，并给出修改建议。"""

# ── Prompt framings ─────────────────────────────────────────────────────

def generation_prompt(base_prompt: str, num_user_photos: int = 1) -> str:
    """Template-first, user-second (production framing)."""
    user_indices = ", ".join(str(i) for i in range(2, 2 + num_user_photos))
    return f"""\
Image 1 is the style reference ONLY — use it for composition, lighting, background, and clothing style.
Images {user_indices} are all of the SAME PERSON (the user). This person is the ONLY subject whose face you must preserve.

Instruction:
{base_prompt}

Critical requirements:
- Generate a portrait of the person in images {user_indices}, NOT the person in image 1.
- Preserve the user's exact facial features, face shape, eyes, nose, mouth, eyebrows, skin tone, and overall identity.
- Apply only the style/composition/background/clothing from image 1.
- Natural skin texture, no beauty filter, no plastic skin.
- Photorealistic, professional portrait quality."""


def editing_prompt(base_prompt: str, num_user_photos: int = 1) -> str:
    """User-first, template-second (explicit edit framing)."""
    user_indices = ", ".join(str(i) for i in range(1, 1 + num_user_photos))
    style_index = 1 + num_user_photos
    return f"""\
Images {user_indices} are the user's photos. This is the person whose face, expression, and identity you must preserve EXACTLY.
Image {style_index} is the style reference — use it for composition, lighting, background, and clothing style.

Instruction:
Apply the style from image {style_index} to the person in images {user_indices}. Change clothing and background to match image {style_index}, but keep the face and identity identical.

Critical requirements:
- Edit the person in images {user_indices} to match the style/composition/background/clothing of image {style_index}.
- Keep the person's facial features, face shape, eyes, nose, mouth, eyebrows, skin tone, expression, and overall identity exactly the same as in images {user_indices}.
- Change only clothing, background, lighting, and overall aesthetic to match image {style_index}.
- Do NOT generate a different person.
- Natural skin texture, no beauty filter, no plastic skin.
- Photorealistic, professional portrait quality."""


# ── Transport helpers ───────────────────────────────────────────────────

def b64_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def content_blocks(text: str, image_paths: list[Path]) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": text}] if text else []
    for p in image_paths:
        blocks.append({"type": "image_url", "image_url": {"url": b64_data_url(p)}})
    return blocks


def post(
    model: str,
    messages: list[dict],
    modalities: list[str] | None = None,
    image_config: dict | None = None,
    timeout: int = 180,
) -> dict:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if modalities:
        payload["modalities"] = modalities
    if image_config:
        payload["image_config"] = image_config
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}/chat/completions"
    last_err: Exception | None = None
    for attempt in (1, 2):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:600]
            except Exception:
                pass
            if e.code in {429, 500, 502, 503, 504} and attempt == 1:
                print(f"    ⚠ transient HTTP {e.code}, retrying… {detail[:200]}")
                time.sleep(3)
                continue
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt == 1:
                print(f"    ⚠ transport error ({e}), retrying…")
                time.sleep(3)
                continue
            raise RuntimeError(f"OpenRouter transport error: {e}") from e
    raise RuntimeError(f"OpenRouter request failed: {last_err}")


def extract_image(resp: dict, save_dir: Path, stem: str) -> Path:
    msg = resp.get("choices", [{}])[0].get("message", {})
    images = msg.get("images") or []
    if not images:
        snippet = json.dumps(msg, ensure_ascii=False)[:500]
        raise RuntimeError(f"No image in response: {snippet}")
    url = images[0]["image_url"]["url"]
    raw = base64.b64decode(url.split(",", 1)[1])
    img = Image.open(BytesIO(raw))
    img.load()
    out = save_dir / f"{stem}_{uuid.uuid4().hex[:8]}.png"
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    img.save(out, format="PNG")
    return out


def extract_text(resp: dict) -> str:
    msg = resp.get("choices", [{}])[0].get("message", {})
    text = msg.get("content") or ""
    if isinstance(text, list):
        text = "\n".join(
            blk.get("text", "") for blk in text if isinstance(blk, dict)
        )
    return text.strip()


def parse_judge_response(text: str) -> tuple[int | None, str | None]:
    import re
    if not text:
        return None, None
    score = None
    patterns = [
        r"评分[：:]\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"评分[:：]?\s*(\d+(?:\.\d+)?)\s*[分点]",
        r"相似度[：:]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*分\s*[,，]\s*满分\s*10",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            score = int(float(m.group(1)) + 0.5)
            score = max(1, min(10, score))
            break
    feedback = None
    feedback_sections = []
    capturing = False
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(kw in line for kw in [
            "需要调整", "差异", "不像", "不一致", "不同",
            "调整以下", "以下面部特征", "具体来说",
        ]):
            capturing = True
        if capturing:
            feedback_sections.append(line)
    if feedback_sections:
        feedback = "\n".join(feedback_sections)
    elif score is not None and score < 8:
        feedback = text
    return score, feedback


# ── Core calls ──────────────────────────────────────────────────────────

def generate_image(
    model: str,
    prompt_text: str,
    image_paths: list[Path],
    image_config: dict | None = None,
    timeout: int = 180,
) -> dict:
    """Generate an image. Returns the raw API response."""
    messages = [
        {"role": "user", "content": content_blocks(prompt_text, image_paths)}
    ]
    return post(
        model=model,
        messages=messages,
        modalities=["image", "text"],
        image_config=image_config,
        timeout=timeout,
    )


def judge_image(output_path: Path, user_path: Path, judge_model: str = "google/gemini-2.5-flash") -> tuple[int | None, str | None, str]:
    """Run strict identity judge. Returns (score, feedback, raw_text)."""
    messages = [
        {
            "role": "user",
            "content": content_blocks(JUDGE_PROMPT, [output_path, user_path]),
        }
    ]
    resp = post(model=judge_model, messages=messages, modalities=["text"], timeout=120)
    text = extract_text(resp)
    score, feedback = parse_judge_response(text)
    return score, feedback, text


# ── Experiment data ─────────────────────────────────────────────────────

@dataclass
class TestCase:
    subject_id: str
    selfie_path: Path
    template_id: str
    template_path: Path
    base_prompt: str


@dataclass
class Condition:
    model: str
    framing: str  # "generation" or "editing"
    image_config: dict | None


@dataclass
class Result:
    subject_id: str
    template_id: str
    model: str
    framing: str
    output_path: str
    score: int | None
    feedback: str | None
    raw_judge: str
    latency_sec: float
    error: str | None


def load_prompts() -> dict:
    with open(ROOT / "prompts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(
    prompts: dict,
    subject_ids: list[str] | None = None,
    template_ids: list[str] | None = None,
) -> list[TestCase]:
    """Pick a few (subject, template) pairs. Adjust here for more coverage."""
    # Subjects chosen from existing uploads. Prefer real selfies / phone photos.
    all_subjects = [
        ("s_8ccf_iphone", ROOT / "data" / "uploads" / "s_8ccf8599" / "0858c821-0b19-49fe-85ed-84dd441afaa6.PNG"),
        ("s_b735", ROOT / "data" / "uploads" / "s_b7354aef" / "bf_m_01_20260608_120602.png"),
    ]
    subjects = [s for s in all_subjects if subject_ids is None or s[0] in subject_ids]
    # Templates chosen to span styles. All male to match the male subjects above.
    all_template_ids = ["gf_m_hanfu", "bf_m_04", "film_m_cyber"]
    tids = template_ids if template_ids is not None else all_template_ids
    cases = []
    styles = prompts.get("styles", {})
    for tid in tids:
        template_path = TEMPLATES_DIR / f"{tid}.png"
        if not template_path.exists():
            print(f"⚠ template not found: {template_path}")
            continue
        # Find the template entry to get its English gen_prompt.
        base_prompt = ""
        for cat in styles.values():
            for t in cat.get("templates", []):
                if t.get("id") == tid:
                    base_prompt = t.get("gen_prompt", t.get("prompt", ""))
                    break
            if base_prompt:
                break
        if not base_prompt:
            print(f"⚠ no prompt for template {tid}")
            continue
        for sid, spath in subjects:
            if not spath.exists():
                print(f"⚠ selfie not found: {spath}")
                continue
            cases.append(TestCase(
                subject_id=sid,
                selfie_path=spath,
                template_id=tid,
                template_path=template_path,
                base_prompt=base_prompt,
            ))
    return cases


ALL_MODELS = [
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-3-pro-image-preview",
    "google/gemini-2.5-flash-image",
    "openai/gpt-5-image-mini",
    "openai/gpt-5-image",
]


def build_conditions(models: list[str] | None = None, framings: list[str] | None = None) -> list[Condition]:
    """Models and framings to compare."""
    gemini_config = {"aspect_ratio": "3:4", "image_size": "1K"}
    # OpenAI image models do not use the same image_config keys; leave None.
    out: list[Condition] = []
    for m in (models or ALL_MODELS):
        for f in (framings or ["generation", "editing"]):
            config = None if m.startswith("openai/") else gemini_config
            out.append(Condition(m, f, config))
    return out


# ── Main runner ─────────────────────────────────────────────────────────

def run_experiment(
    models: list[str] | None = None,
    framings: list[str] | None = None,
    subject_ids: list[str] | None = None,
    template_ids: list[str] | None = None,
) -> list[Result]:
    prompts = load_prompts()
    cases = build_test_cases(prompts, subject_ids=subject_ids, template_ids=template_ids)
    conditions = build_conditions(models=models, framings=framings)
    total = len(cases) * len(conditions)
    print(f"Running {len(cases)} cases × {len(conditions)} conditions = {total} generations\n")

    results: list[Result] = []
    for ci, cond in enumerate(conditions, 1):
        print(f"[{ci}/{len(conditions)}] Model: {cond.model} | Framing: {cond.framing}")
        for ti, case in enumerate(cases, 1):
            print(f"  case {ti}/{len(cases)}: subject={case.subject_id} template={case.template_id}")
            # Determine image order and prompt.
            if cond.framing == "generation":
                image_paths = [case.template_path, case.selfie_path]
                prompt = generation_prompt(case.base_prompt, num_user_photos=1)
            else:
                image_paths = [case.selfie_path, case.template_path]
                prompt = editing_prompt(case.base_prompt, num_user_photos=1)

            stem = f"{case.subject_id}_{case.template_id}_{cond.model.replace('/', '_')}_{cond.framing}"
            error: str | None = None
            output_path: Path | None = None
            score: int | None = None
            feedback: str | None = None
            raw_judge = ""
            t0 = time.time()
            try:
                resp = generate_image(
                    model=cond.model,
                    prompt_text=prompt,
                    image_paths=image_paths,
                    image_config=cond.image_config,
                    timeout=240,
                )
                output_path = extract_image(resp, OUTPUT_DIR, stem)
                print(f"    ✓ generated: {output_path.name}")
                score, feedback, raw_judge = judge_image(output_path, case.selfie_path)
                print(f"    ✓ judge score: {score}/10")
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                print(f"    ❌ error: {error}")

            results.append(Result(
                subject_id=case.subject_id,
                template_id=case.template_id,
                model=cond.model,
                framing=cond.framing,
                output_path=str(output_path) if output_path else "",
                score=score,
                feedback=feedback,
                raw_judge=raw_judge,
                latency_sec=round(time.time() - t0, 1),
                error=error,
            ))
            # Small polite delay between calls.
            time.sleep(1)
        print()
    return results


def save_results(results: list[Result]) -> None:
    rows = [asdict(r) for r in results]
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    from collections import defaultdict
    by_model_framing: dict[tuple[str, str], list[int | None]] = defaultdict(list)
    for r in results:
        by_model_framing[(r.model, r.framing)].append(r.score)

    summary_rows = []
    for (model, framing), scores in sorted(by_model_framing.items()):
        valid = [s for s in scores if s is not None]
        errors = [s for s in scores if s is None]
        avg = round(sum(valid) / len(valid), 2) if valid else None
        pass_rate = round(sum(1 for s in valid if s >= 8) / len(valid), 2) if valid else 0
        summary_rows.append((model, framing, len(valid), len(errors), avg, pass_rate))

    print(f"{'model':<45} {'framing':<12} {'n':>3} {'err':>3} {'avg':>6} {'pass≥8':>7}")
    for model, framing, n, err, avg, pr in summary_rows:
        avg_s = f"{avg:.2f}" if avg is not None else "n/a"
        print(f"{model:<45} {framing:<12} {n:>3} {err:>3} {avg_s:>6} {pr:>7.0%}")

    # Best per (subject, template)
    print("\nBest condition per case:")
    by_case: dict[tuple[str, str], list[Result]] = defaultdict(list)
    for r in results:
        by_case[(r.subject_id, r.template_id)].append(r)
    for (sid, tid), rs in sorted(by_case.items()):
        valid = [r for r in rs if r.score is not None]
        if not valid:
            continue
        best = max(valid, key=lambda r: (r.score or 0))
        print(f"  {sid}/{tid}: {best.score}/10 via {best.model} {best.framing}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare identity-preserving portrait generation vs editing")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS + ["all"], default=["all"],
                        help="Models to test (default: all)")
    parser.add_argument("--framings", nargs="+", choices=["generation", "editing"], default=["generation", "editing"],
                        help="Prompt framings to test")
    parser.add_argument("--templates", nargs="+", default=None,
                        help="Template IDs to test (default: gf_m_hanfu bf_m_04 film_m_cyber)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs to test (default: s_8ccf_iphone s_b735)")
    parser.add_argument("--dry-run", action="store_true", help="Print matrix and exit without API calls")
    args = parser.parse_args()

    models = ALL_MODELS if "all" in args.models else args.models
    if args.dry_run:
        print("Dry run matrix:")
        for cond in build_conditions(models, args.framings):
            print(f"  {cond.model} / {cond.framing}")
        sys.exit(0)

    if not API_KEY:
        print("OPENROUTER_API_KEY not set. Aborting.")
        sys.exit(1)
    print(f"Output directory: {OUTPUT_DIR}\n")
    results = run_experiment(
        models=models,
        framings=args.framings,
        subject_ids=args.subjects,
        template_ids=args.templates,
    )
    save_results(results)
    print(f"\nSaved report: {REPORT_JSON}")
    print(f"Saved CSV:    {REPORT_CSV}")
    print_summary(results)
