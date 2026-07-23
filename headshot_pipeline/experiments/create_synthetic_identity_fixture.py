#!/usr/bin/env python3
"""Create an internal-QA-only multi-view identity fixture from a synthetic seed."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from server.config import settings
from server.generation.providers import OpenRouterProvider


VIEW_SPECS = (
    (
        "01-front-neutral.jpg",
        "front-facing at eye level, neutral relaxed expression, looking into the camera",
    ),
    (
        "02-front-smile.jpg",
        "front-facing at eye level, small natural smile, looking into the camera",
    ),
    (
        "03-three-quarter-left.jpg",
        "head turned about 35 degrees toward the person's left, neutral expression, both eyes readable",
    ),
    (
        "04-three-quarter-right.jpg",
        "head turned about 35 degrees toward the person's right, neutral expression, both eyes readable",
    ),
)


def build_view_prompt(view: str) -> str:
    return f"""Create a plain smartphone identity-reference photograph of the exact
same fictional adult shown in the reference images.

Requested view: {view}.

Use a complete head-and-shoulders crop against an ordinary light-gray indoor
wall, soft natural window light, a plain charcoal crew-neck shirt, and normal
smartphone perspective from about 1.5 metres away. Keep one unobstructed face
near the centre with breathing room around the hair.

Identity is the hard constraint. Preserve the same facial proportions, eye
size and spacing, nose, mouth, jaw width, ears, hairline, hairstyle, skin tone,
apparent age, asymmetry, moles, and other stable markers across every view.
Render an unretouched reference capture with pores, fine lines, under-eye
structure, flyaway hairs, and ordinary local contrast.

Do not beautify, reshape, rejuvenate, add makeup, enlarge the eyes, narrow the
jaw, smooth the skin, change ethnicity, or copy the seed image's wardrobe and
background. No studio glamour, fake bokeh, text, border, watermark, hands, or
additional people."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--identity-id", required=True)
    parser.add_argument("--gender", choices=("female", "male"), required=True)
    parser.add_argument("--confirm-synthetic-seed", action="store_true")
    parser.add_argument("--confirm-api-spend", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_synthetic_seed:
        raise SystemExit("Refusing to run without --confirm-synthetic-seed")
    if not args.confirm_api_spend:
        raise SystemExit("Refusing to run without --confirm-api-spend")
    if not settings.openrouter_api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing")
    seed = args.seed_image.resolve()
    if not seed.is_file():
        raise SystemExit(f"Synthetic seed does not exist: {seed}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = output_dir / ".generation"
    raw_dir.mkdir()

    provider = OpenRouterProvider(
        api_key=settings.openrouter_api_key,
        output_dir=raw_dir,
        model=settings.gemini_model,
        judge_model=settings.openrouter_judge_model,
        base_url=settings.openrouter_base_url,
        timeout=settings.gemini_wait_timeout,
        image_provider=settings.openrouter_image_provider,
        image_size=settings.openrouter_image_size,
        estimated_image_cost=settings.openrouter_estimated_image_cost,
        minimum_credit_balance=settings.openrouter_min_credit_balance,
        max_reference_images=settings.openrouter_max_reference_images,
    )
    generation_records = []
    primary_front: Path | None = None
    try:
        for index, (filename, view) in enumerate(VIEW_SPECS, start=1):
            references = [str(seed)]
            if primary_front is not None:
                references.append(str(primary_front))
            started_at = time.time()
            generated = Path(provider.create_from_references(
                prompt=build_view_prompt(view),
                reference_paths=references,
                template_path=None,
                title=f"{args.identity_id}_{index}",
                editing_mode=True,
            ))
            destination = output_dir / filename
            shutil.copy2(generated, destination)
            if primary_front is None:
                primary_front = destination
            generation_records.append({
                "filename": filename,
                "view": view,
                "reference_roles": (
                    ["synthetic_seed"]
                    if index == 1 else ["synthetic_seed", "generated_front_anchor"]
                ),
                "latency_seconds": round(time.time() - started_at, 3),
                "provider_usage": dict(provider._image_client.last_usage or {}),
            })
    finally:
        provider.end_session()

    metadata = {
        "allowed_uses": ["internal_qa"],
        "contains_real_person": False,
        "created_at": date.today().isoformat(),
        "fixture_kind": "repository-owned synthetic adult",
        "generator": f"{settings.gemini_model} via OpenRouter",
        "gender": args.gender,
        "identity_id": args.identity_id,
        "source_kind": "repository-owned synthetic template plus generated identity views",
        "synthetic": True,
    }
    (output_dir / "fixture.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "generation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": settings.gemini_model,
                "provider": settings.openrouter_image_provider,
                "estimated_max_cost": round(
                    len(VIEW_SPECS) * settings.openrouter_estimated_image_cost,
                    4,
                ),
                "records": generation_records,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(raw_dir)
    print(json.dumps({
        "status": "generated_pending_audit",
        "identity_id": args.identity_id,
        "output_dir": str(output_dir),
        "image_count": len(VIEW_SPECS),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
