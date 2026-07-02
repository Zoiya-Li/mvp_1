#!/usr/bin/env python3
"""Generate template reference images via Gemini.

Reads prompts.json, finds templates without existing images,
generates them one-by-one via Chrome CDP + Gemini, removes watermarks,
and saves to templates/ directory.

Usage:
    python generate_templates.py              # Generate all missing
    python generate_templates.py --force      # Regenerate all
    python generate_templates.py --id bf_m_tech  # Generate specific template
"""

import json
import shutil
import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate template images via Gemini")
    parser.add_argument("--force", action="store_true", help="Regenerate all templates")
    parser.add_argument("--id", dest="template_id", help="Generate specific template only")
    parser.add_argument("--style", help="Generate only templates for this style category")
    args = parser.parse_args()

    prompts_path = HERE / "prompts.json"
    templates_dir = HERE / "templates"
    templates_dir.mkdir(exist_ok=True)

    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect templates to generate
    todos = []
    for style_key, style_data in data["styles"].items():
        if args.style and style_key != args.style:
            continue
        for tmpl in style_data["templates"]:
            if args.template_id and tmpl["id"] != args.template_id:
                continue

            image_path = templates_dir / f"{tmpl['id']}.png"
            exists = image_path.exists()

            if exists and not args.force:
                print(f"  SKIP {tmpl['id']}: already exists ({image_path.stat().st_size // 1024}KB)")
                continue

            todos.append({
                "id": tmpl["id"],
                "label": tmpl["label"],
                "style": style_key,
                "gen_prompt": tmpl["gen_prompt"],
                "image_path": image_path,
                "exists": exists,
            })

    if not todos:
        print("No templates to generate. All images exist.")
        return

    print(f"\n{'='*60}")
    print(f"  {len(todos)} templates to generate")
    print(f"{'='*60}\n")

    for i, t in enumerate(todos):
        print(f"\n--- [{i+1}/{len(todos)}] {t['id']}: {t['label']} ({t['style']}) ---")

    # Connect to Chrome
    print("\nConnecting to Chrome...")
    from persistent_client import PersistentGeminiClient
    from watermark_remover import remove_gemini_watermark

    client = PersistentGeminiClient(port=9222)
    client.connect()
    print("Connected.\n")

    success = 0
    failed = 0

    for i, t in enumerate(todos):
        print(f"\n[{i+1}/{len(todos)}] Generating: {t['id']} — {t['label']}")
        try:
            filepath = client.generate(
                prompt=t["gen_prompt"],
                title=f"tmpl_{t['id']}",
                photo_path=None,  # No reference photo — template is the style itself
            )

            if filepath and Path(filepath).exists():
                # Remove watermark
                from PIL import Image as PILImage
                img = PILImage.open(filepath)
                cleaned = remove_gemini_watermark(img)

                # Save to templates directory
                cleaned.save(str(t["image_path"]), format="PNG")
                print(f"  ✓ Saved: {t['image_path']} ({t['image_path'].stat().st_size // 1024}KB)")
                success += 1
            else:
                print(f"  ✗ FAILED: no image returned")
                failed += 1

            # Brief pause between generations
            if i < len(todos) - 1:
                time.sleep(3)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
            # Try to continue with next
            try:
                client.ensure_gemini_page()
                client._new_chat()
                time.sleep(2)
            except Exception:
                print("  ⚠ Could not recover, stopping.")
                break

    print(f"\n{'='*60}")
    print(f"  Done: {success} generated, {failed} failed")
    print(f"{'='*60}")

    # Copy to landing page
    landing_images = HERE.parent / "headshot-landing" / "public" / "images"
    if landing_images.exists():
        for t in todos:
            if t["image_path"].exists():
                dest = landing_images / t["image_path"].name
                shutil.copy2(t["image_path"], dest)
        print(f"  Copied new templates to {landing_images}")

    client.disconnect()


if __name__ == "__main__":
    main()
