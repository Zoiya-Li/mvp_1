#!/usr/bin/env python3
"""A/B paid-set realism and identity across Seedream and FLUX.

The experiment uses repository-owned synthetic references and three difficult
set compositions. It is intentionally separate from production routing: model
choice changes only after the resulting contact sheets and QA agree.
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
from server.shot_planner import build_style_shot_plan


BENCHMARK_SHOTS = {"environmental", "seated", "profile"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gender", choices=("male", "female"), default="male")
    parser.add_argument(
        "--confirm-synthetic-fixture",
        action="store_true",
        help="Required acknowledgement that all references are synthetic fixtures.",
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
        },
        "local_quality": judgement.get("local_quality") or {},
        "quality_evaluation": judgement.get("quality_evaluation") or {},
        "notes": judgement.get("notes"),
    }


def provider_variants(output_dir: Path) -> list[tuple[str, OpenRouterProvider]]:
    return [
        (
            "seedream_active",
            OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                output_dir=output_dir / "seedream_active",
                model=settings.gemini_model,
                judge_model=settings.openrouter_judge_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.gemini_wait_timeout,
                image_provider=settings.openrouter_image_provider,
                image_size=settings.openrouter_image_size,
                estimated_image_cost=settings.openrouter_estimated_image_cost,
                minimum_credit_balance=settings.openrouter_min_credit_balance,
                max_reference_images=settings.openrouter_max_reference_images,
            ),
        ),
        (
            "flux_identity",
            OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                output_dir=output_dir / "flux_identity",
                model=settings.openrouter_hero_model,
                judge_model=settings.openrouter_judge_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.gemini_wait_timeout,
                image_provider=settings.openrouter_hero_image_provider,
                image_size=settings.openrouter_image_size,
                estimated_image_cost=settings.openrouter_hero_estimated_image_cost,
                minimum_credit_balance=settings.openrouter_min_credit_balance,
                max_reference_images=settings.openrouter_max_reference_images,
            ),
        ),
    ]


def main() -> int:
    args = parse_args()
    if not args.confirm_synthetic_fixture:
        raise SystemExit("Refusing to run without --confirm-synthetic-fixture")
    references = sorted(
        path.resolve()
        for path in args.fixture_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if len(references) != 4:
        raise SystemExit("This benchmark requires exactly four synthetic references")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    prompts_path = Path(__file__).resolve().parents[1] / "prompts.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    style_data = prompts["styles"]["cinematic"]
    plan = [
        shot for shot in build_style_shot_plan(
            "cinematic", args.gender, style_data
        )
        if shot.shot_spec.get("shot_id") in BENCHMARK_SHOTS
    ]
    evaluator = EvaluationService()
    report = {
        "fixture_kind": "repository-owned synthetic adult",
        "reference_count": len(references),
        "gender": args.gender,
        "shots": [shot.shot_spec.get("shot_id") for shot in plan],
        "variants": [],
    }

    try:
        for variant_name, provider in provider_variants(output_dir):
            variant = {
                "name": variant_name,
                "model": provider.model,
                "provider": provider.image_provider,
                "results": [],
            }
            report["variants"].append(variant)
            for shot in plan:
                shot_id = str(shot.shot_spec["shot_id"])
                started_at = time.time()
                prompt = build_candidate_prompt(
                    shot.prompt,
                    len(references),
                    candidate_index=1,
                    total_candidates=1,
                )
                output = provider.create_from_references(
                    prompt=prompt,
                    reference_paths=[str(path) for path in references],
                    template_path=None,
                    title=f"{variant_name}_{shot_id}",
                    editing_mode=True,
                )
                judgement = evaluator.judge_current_candidate(
                    provider,
                    output,
                    [str(path) for path in references],
                    shot_spec=shot.shot_spec,
                    identity_attributes=None,
                )
                variant["results"].append({
                    "shot_id": shot_id,
                    "output_filename": Path(output).name,
                    "latency_seconds": round(time.time() - started_at, 3),
                    "provider_usage": dict(provider._image_client.last_usage or {}),
                    "judgement": compact_judgement(judgement),
                })
                (output_dir / "report.json").write_text(
                    json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            provider.end_session()
    finally:
        evaluator.release_identity_app()

    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
