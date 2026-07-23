#!/usr/bin/env python3
"""Benchmark Hero identity stability across synthetic reference subsets.

This operator-only experiment deliberately requires an explicit synthetic-data
acknowledgement. It writes generated candidates and a compact QA report without
copying the input references into the artifact directory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from server.config import settings
from server.evaluation import EvaluationService
from server.gemini_worker import build_candidate_prompt
from server.generation.providers import OpenRouterProvider


HERO_DIRECTION = """\
Create an unretouched vertical 3:4 documentary editorial portrait of the adult
in the identity references. Use a natural chest-up composition photographed
from about two metres away, a subtle three-quarter turn, ordinary open-shade
daylight, matte white linen, and a physically readable neighborhood exterior.
The face should occupy roughly 28-38% of frame height. Preserve real apparent
age, facial asymmetry, moles, under-eye structure, pores, flyaway hairs, and
small skin-tone variation. This must look like a real camera photograph, not a
selfie, ID photo, beauty campaign, synthetic model, or retouched AI portrait.
"""

SHOT_SPEC = {
    "shot_id": "closeup",
    "shot_label": "Hero reference-pack benchmark",
    "hero_preview": True,
    "framing": "natural chest-up portrait with breathing room",
    "pose": "subtle three-quarter turn with relaxed shoulders",
    "environment": "readable neighborhood exterior in open shade",
    "lighting": "ordinary open-shade daylight",
    "lens": "50mm to 70mm documentary portrait lens",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--confirm-synthetic-fixture",
        action="store_true",
        help="Required acknowledgement that every input is repository-owned synthetic data.",
    )
    return parser.parse_args()


def compact_judgement(judgement: dict) -> dict:
    identity = judgement.get("identity_quality") or {}
    return {
        "scores": judgement.get("scores") or {},
        "hard_failures": judgement.get("hard_failures") or [],
        "recommended_action": judgement.get("recommended_action"),
        "identity": {
            "cosine_similarity": identity.get("cosine_similarity"),
            "reference_consistency": identity.get("reference_consistency"),
            "hard_failures": identity.get("hard_failures") or [],
            "measurements": identity.get("measurements") or {},
        },
        "local_quality": judgement.get("local_quality") or {},
        "quality_evaluation": judgement.get("quality_evaluation") or {},
        "notes": judgement.get("notes"),
    }


def main() -> int:
    args = parse_args()
    if not args.confirm_synthetic_fixture:
        raise SystemExit("Refusing to run without --confirm-synthetic-fixture")
    fixture_paths = sorted(
        path.resolve()
        for path in args.fixture_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(fixture_paths) != 4:
        raise SystemExit("This benchmark requires exactly four synthetic references")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    strategies = [
        ("front_only", [0]),
        ("front_smile", [0, 1]),
        ("front_angles", [0, 2, 3]),
        ("all_four", [0, 1, 2, 3]),
    ]
    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        output_dir=output_dir,
        model=settings.openrouter_hero_model,
        judge_model=settings.openrouter_judge_model,
        base_url=settings.openrouter_base_url,
        timeout=settings.gemini_wait_timeout,
        image_provider=settings.openrouter_hero_image_provider,
        image_size=settings.openrouter_image_size,
        estimated_image_cost=settings.openrouter_hero_estimated_image_cost,
        minimum_credit_balance=settings.openrouter_min_credit_balance,
        max_reference_images=settings.openrouter_max_reference_images,
    )
    evaluator = EvaluationService()
    report = {
        "fixture_kind": "repository-owned synthetic adult",
        "model": settings.openrouter_hero_model,
        "provider": settings.openrouter_hero_image_provider,
        "strategies": [],
    }

    try:
        for name, indexes in strategies:
            references = [fixture_paths[index] for index in indexes]
            started_at = time.time()
            prompt = build_candidate_prompt(
                HERO_DIRECTION,
                len(references),
                candidate_index=1,
                total_candidates=1,
            )
            output = provider.create_from_references(
                prompt=prompt,
                reference_paths=[str(path) for path in references],
                template_path=None,
                title=f"hero_refpack_{name}",
                editing_mode=True,
            )
            judgement = evaluator.judge_current_candidate(
                provider,
                output,
                [str(path) for path in fixture_paths],
                shot_spec=SHOT_SPEC,
                identity_attributes=None,
            )
            report["strategies"].append({
                "name": name,
                "reference_roles": [path.stem for path in references],
                "reference_count": len(references),
                "output_filename": Path(output).name,
                "latency_seconds": round(time.time() - started_at, 3),
                "provider_usage": dict(provider._image_client.last_usage or {}),
                "judgement": compact_judgement(judgement),
            })
            (output_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
                encoding="utf-8",
            )
    finally:
        evaluator.release_identity_app()
        provider.end_session()

    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
