#!/usr/bin/env python3
"""Judge existing experiment images and rebuild the report."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from compare_identity_preservation import (
    OUTPUT_DIR,
    judge_image,
    load_prompts,
    build_test_cases,
)


@dataclass
class RecoveredResult:
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


KNOWN_TEMPLATES = ("gf_m_hanfu", "bf_m_04", "film_m_cyber")
KNOWN_PROVIDERS = ("google_", "openai_", "black-forest-labs_", "sourceful_", "x-ai_")


def parse_filename(stem: str) -> dict | None:
    """Parse filenames like s_8ccf_iphone_bf_m_04_google_gemini-3.1-flash-image-preview_generation_fe20c2a4.

    Filename structure is: {subject}_{template}_{model_with_underscores}_{framing}_{uuid8}
    """
    if stem == "smoke_gen_e2de473f" or stem.endswith("_multi_edit"):
        return None
    # Strip uuid suffix (last 8 hex chars).
    if len(stem) < 9 or not stem[-8:].isalnum():
        return None
    prefix = stem[:-9].rstrip("_")
    # Find framing at the end.
    framing = None
    for f in ("generation", "editing"):
        if prefix.endswith(f"_{f}"):
            framing = f
            break
    if framing is None:
        return None
    rest = prefix[: -len(f"_{framing}")]
    # Find model provider in rest; model is everything from provider to end of rest.
    model_start = -1
    provider_found = None
    for provider in KNOWN_PROVIDERS:
        idx = rest.find(f"_{provider}")
        if idx != -1:
            model_start = idx + 1  # skip the leading underscore
            provider_found = provider
            break
    if model_start == -1:
        return None
    model_raw = rest[model_start:]
    model = model_raw.replace("_", "/")
    before_model = rest[: model_start - 1]
    # Find template in before_model.
    for tid in KNOWN_TEMPLATES:
        tid_idx = before_model.find(f"_{tid}")
        if tid_idx != -1:
            subject_id = before_model[:tid_idx]
            return {
                "subject_id": subject_id,
                "template_id": tid,
                "model": model,
                "framing": framing,
            }
    return None


def main() -> None:
    prompts = load_prompts()
    cases = {c.subject_id: c.selfie_path for c in build_test_cases(prompts)}

    results: list[RecoveredResult] = []
    pngs = sorted(OUTPUT_DIR.glob("*.png"))
    for p in pngs:
        meta = parse_filename(p.stem)
        if meta is None:
            continue
        selfie_path = cases.get(meta["subject_id"])
        if not selfie_path or not selfie_path.exists():
            print(f"⚠ no selfie for {meta['subject_id']}")
            continue
        print(f"Judging {p.name} …")
        t0 = time.time()
        try:
            score, feedback, raw = judge_image(p, selfie_path)
            error = None
            print(f"  score={score}/10")
        except Exception as e:
            score, feedback, raw, error = None, None, "", f"{type(e).__name__}: {e}"
            print(f"  error: {error}")
        results.append(RecoveredResult(
            subject_id=meta["subject_id"],
            template_id=meta["template_id"],
            model=meta["model"],
            framing=meta["framing"],
            output_path=str(p),
            score=score,
            feedback=feedback,
            raw_judge=raw,
            latency_sec=round(time.time() - t0, 1),
            error=error,
        ))
        time.sleep(0.5)

    rows = [asdict(r) for r in results]
    with open(OUTPUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if rows:
        with open(OUTPUT_DIR / "report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nRecovered {len(results)} results.")


if __name__ == "__main__":
    main()
