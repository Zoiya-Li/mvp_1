#!/usr/bin/env python3
"""Compare fallback image models across every audited synthetic identity."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from experiments.benchmark_hero_reference_pack import (  # noqa: E402
    HERO_DIRECTION,
    SHOT_SPEC,
    compact_judgement,
)
from experiments.benchmark_pipeline_matrix import discover_fixtures  # noqa: E402
from server.config import settings  # noqa: E402
from server.evaluation import EvaluationService  # noqa: E402
from server.gemini_worker import build_candidate_prompt  # noqa: E402
from server.generation.providers import OpenRouterProvider  # noqa: E402


HERO_THRESHOLDS = {
    "profile": "closeup",
    "identity_pass_threshold": 8.0,
    "identity_repair_threshold": 7.0,
    "quality_accept_threshold": 9.0,
    "realism_accept_threshold": 9.0,
    "commercial_accept_threshold": 9.0,
}


def parse_candidate(value: str) -> dict:
    """Parse NAME,MODEL,PROVIDER_TAG,ESTIMATED_COST."""
    fields = [field.strip() for field in value.split(",")]
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            "candidate must be NAME,MODEL,PROVIDER_TAG,ESTIMATED_COST"
        )
    try:
        cost = float(fields[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate cost must be numeric") from exc
    if not all(fields[:3]) or cost <= 0:
        raise argparse.ArgumentTypeError("candidate fields must be non-empty")
    return {
        "name": fields[0],
        "model": fields[1],
        "provider_tag": fields[2],
        "estimated_cost": cost,
    }


def summarize_variant(results: list[dict]) -> dict:
    cosines = [
        item["judgement"]["identity"].get("cosine_similarity")
        for item in results
    ]
    cosines = [float(value) for value in cosines if value is not None]
    return {
        "case_count": len(results),
        "pass_count": sum(bool(item["gate_status"]["hard_gates_pass"]) for item in results),
        "pass_rate": round(
            sum(bool(item["gate_status"]["hard_gates_pass"]) for item in results)
            / len(results),
            4,
        ) if results else 0.0,
        "mean_identity_cosine": round(statistics.mean(cosines), 4) if cosines else None,
        "min_identity_cosine": round(min(cosines), 4) if cosines else None,
        "estimated_cost": round(sum(item["estimated_cost"] for item in results), 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--confirm-synthetic-fixtures", action="store_true")
    parser.add_argument("--minimum-identities", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_synthetic_fixtures:
        raise SystemExit("Refusing to run without --confirm-synthetic-fixtures")
    fixtures = discover_fixtures(args.fixture_root.resolve())
    if len(fixtures) < args.minimum_identities:
        raise SystemExit(
            f"Requires at least {args.minimum_identities} audited identities"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    evaluator = EvaluationService()
    report = {
        "fixture_kind": "repository-owned synthetic adults",
        "identity_count": len(fixtures),
        "thresholds": HERO_THRESHOLDS,
        "variants": [],
    }
    try:
        for descriptor in args.candidate:
            variant_dir = output_dir / descriptor["name"]
            provider = OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                output_dir=variant_dir,
                model=descriptor["model"],
                judge_model=settings.openrouter_judge_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.gemini_wait_timeout,
                image_provider=descriptor["provider_tag"],
                image_size=settings.openrouter_image_size,
                estimated_image_cost=descriptor["estimated_cost"],
                minimum_credit_balance=settings.openrouter_min_credit_balance,
                max_reference_images=settings.openrouter_max_reference_images,
            )
            variant = {**descriptor, "results": []}
            report["variants"].append(variant)
            try:
                for fixture in fixtures:
                    references = sorted(
                        path.resolve()
                        for path in Path(fixture["fixture_dir"]).iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    )
                    prompt = build_candidate_prompt(
                        HERO_DIRECTION,
                        len(references),
                        candidate_index=1,
                        total_candidates=1,
                    )
                    started_at = time.time()
                    output = provider.create_from_references(
                        prompt=prompt,
                        reference_paths=[str(path) for path in references],
                        template_path=None,
                        title=f"{descriptor['name']}_{fixture['identity_id']}",
                        editing_mode=True,
                    )
                    judgement = evaluator.judge_current_candidate(
                        provider,
                        output,
                        [str(path) for path in references],
                        shot_spec=SHOT_SPEC,
                        identity_attributes=None,
                    )
                    gate = evaluator._candidate_gate_status(
                        judgement,
                        HERO_THRESHOLDS,
                    )
                    result = {
                        "identity_id": fixture["identity_id"],
                        "output_filename": Path(output).name,
                        "latency_seconds": round(time.time() - started_at, 3),
                        "estimated_cost": descriptor["estimated_cost"],
                        "provider_usage": dict(provider._image_client.last_usage or {}),
                        "judgement": compact_judgement(judgement),
                        "gate_status": gate,
                    }
                    variant["results"].append(result)
                    variant["summary"] = summarize_variant(variant["results"])
                    (output_dir / "report.json").write_text(
                        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
            finally:
                provider.end_session()
    finally:
        evaluator.release_identity_app()

    eligible = [
        variant for variant in report["variants"]
        if variant["summary"]["pass_rate"] >= 2 / 3
    ]
    report["winner"] = (
        max(
            eligible,
            key=lambda item: (
                item["summary"]["pass_rate"],
                item["summary"]["min_identity_cosine"] or 0,
                item["summary"]["mean_identity_cosine"] or 0,
            ),
        )["name"]
        if eligible else None
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0 if report["winner"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
