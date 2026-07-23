"""
Batch Headshot Generation Pipeline
====================================
基于 gemini_automation 库的批量职业形象照生成管线。

用法:
  # 生成全部 20 张样张（无用户照片，纯 prompt 生成）
  python batch_headshot.py --all

  # 只生成某个风格
  python batch_headshot.py --style business_formal
  python batch_headshot.py --style academic
  python batch_headshot.py --style id_photo
  python batch_headshot.py --style light_workplace

  # 指定性别
  python batch_headshot.py --style business_formal --gender female

  # 上传用户照片 + 批量风格
  python batch_headshot.py --style business_formal --photo user_photo.jpg

  # 自定义 prompt 列表 JSON
  python batch_headshot.py --custom my_prompts.json
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent dir so we can import gemini_automation
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gemini-image-gen-automation"))

from gemini_automation import GeminiClient


def load_prompts(prompts_file="prompts.json"):
    """Load prompt templates from JSON file."""
    prompts_path = Path(__file__).resolve().parent / prompts_file
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_prompts(data, style_filter=None, gender_filter=None):
    """Flatten the nested prompt structure into a list of (id, prompt, metadata)."""
    results = []
    for style_key, style_data in data["styles"].items():
        if style_filter and style_key != style_filter:
            continue
        for p in style_data["prompts"]:
            if gender_filter and p.get("gender") != gender_filter:
                continue
            results.append({
                "id": p["id"],
                "prompt": p["prompt"],
                "style": style_key,
                "style_label": style_data["label"],
                "gender": p.get("gender", "unknown"),
                "use_cases": style_data.get("use_cases", []),
            })
    return results


def build_headshot_prompt(base_prompt, user_photo_mode=False):
    """Build the final prompt sent to Gemini.

    If user_photo_mode, prepend instructions to use the uploaded photo as reference.
    """
    if user_photo_mode:
        return (
            f"Based on the uploaded photo of a real person, generate a professional headshot. "
            f"Keep the person's facial features, face shape, and likeness as close to the original as possible. "
            f"Style direction: {base_prompt} "
            f"IMPORTANT: The generated photo must look like the same person as in the uploaded reference photo. "
            f"Do NOT change ethnicity, age range, or major facial features. "
            f"Make it photorealistic with natural skin texture, no over-smoothing or beauty filter effects."
        )
    return base_prompt


def run_batch(prompts_list, output_base, user_photo=None, headless=False, delay_between=5):
    """Run batch generation for a list of prompt entries."""
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create subdirectories per style
    for entry in prompts_list:
        (output_base / entry["style"]).mkdir(parents=True, exist_ok=True)

    # Results log
    results = []
    succeeded = 0
    failed = 0

    client = GeminiClient(
        headless=headless,
        output_dir=str(output_base / "temp"),
        wait_timeout=60,
    )

    try:
        client.initialize()

        for i, entry in enumerate(prompts_list):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(prompts_list)}] {entry['id']} | {entry['style_label']} | {entry['gender']}")
            print(f"{'='*60}")

            prompt = build_headshot_prompt(
                entry["prompt"],
                user_photo_mode=(user_photo is not None)
            )

            style_dir = output_base / entry["style"]
            title = entry["id"]

            try:
                image_path = client.generate_image(
                    prompt=prompt,
                    title=title,
                    logo_path=user_photo,
                )

                # Move from temp to style directory
                src = Path(image_path)
                if src.exists():
                    # Determine extension from generated file
                    ext = src.suffix
                    dst = style_dir / f"{title}{ext}"
                    # Handle duplicate
                    if dst.exists():
                        dst = style_dir / f"{title}_{datetime.now().strftime('%H%M%S')}{ext}"
                    src.rename(dst)
                    print(f"✓ Saved: {dst.relative_to(output_base)}")
                    results.append({
                        "id": entry["id"],
                        "status": "success",
                        "path": str(dst),
                        "style": entry["style"],
                        "style_label": entry["style_label"],
                        "gender": entry["gender"],
                    })
                    succeeded += 1
                else:
                    print(f"✗ File not found after generation: {src}")
                    results.append({"id": entry["id"], "status": "file_missing"})
                    failed += 1

            except Exception as e:
                print(f"✗ Failed: {e}")
                results.append({"id": entry["id"], "status": "error", "error": str(e)})
                failed += 1

            # Delay between generations
            if i < len(prompts_list) - 1:
                print(f"  Waiting {delay_between}s before next generation...")
                time.sleep(delay_between)

            # Start new chat for next prompt
            if i < len(prompts_list) - 1:
                try:
                    client.generator.click_new_chat()
                    time.sleep(2)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Saving partial results...")
    finally:
        client.close()

    # Save results log
    log_path = output_base / f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "total": len(prompts_list),
        "succeeded": succeeded,
        "failed": failed,
        "user_photo": user_photo,
        "results": results,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Batch generation complete!")
    print(f"  ✓ Succeeded: {succeeded}/{len(prompts_list)}")
    print(f"  ✗ Failed:    {failed}/{len(prompts_list)}")
    print(f"  Log saved:   {log_path}")
    print(f"  Output dir:  {output_base}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch AI Professional Headshot Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Generate all 20 sample headshots")
    parser.add_argument("--style", type=str, choices=[
        "business_formal", "academic", "id_photo", "light_workplace"
    ], help="Generate only a specific style")
    parser.add_argument("--gender", type=str, choices=["male", "female"], help="Filter by gender")
    parser.add_argument("--photo", type=str, help="Path to user photo for reference-based generation")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--delay", type=int, default=5, help="Delay between generations in seconds")
    parser.add_argument("--custom", type=str, help="Path to custom prompts JSON file")

    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).resolve().parent / "output"

    # Load prompts
    if args.custom:
        data = json.load(open(args.custom, "r", encoding="utf-8"))
    else:
        data = load_prompts()

    # Filter prompts
    if not args.all and not args.style:
        parser.error("Specify --all or --style <style_name>")

    prompts = flatten_prompts(data, style_filter=args.style, gender_filter=args.gender)

    if not prompts:
        print("No prompts matched the given filters.")
        return

    print(f"📋 Loaded {len(prompts)} prompt(s):")
    for p in prompts:
        print(f"  • {p['id']}: {p['style_label']} / {p['gender']}")
    print()

    # Confirm before starting
    print(f"Output: {output_dir.resolve()}")
    print(f"User photo: {args.photo or 'None (text-only prompts)'}")
    print(f"Headless: {args.headless}")
    print()

    # Run batch
    run_batch(
        prompts_list=prompts,
        output_base=output_dir,
        user_photo=args.photo,
        headless=args.headless,
        delay_between=args.delay,
    )


if __name__ == "__main__":
    main()
