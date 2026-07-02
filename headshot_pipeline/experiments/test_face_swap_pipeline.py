"""End-to-end test: generate + face-swap for the failing female case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make server/ importable from experiments/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.config import settings
from server.gemini_worker import GeminiWorker

# Optional: force face swap on/off for comparison
settings.face_swap_enabled = True

PROMPTS = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))


def find_template(style_key: str, template_id: str) -> dict:
    for t in PROMPTS["styles"][style_key]["templates"]:
        if t["id"] == template_id:
            return t
    raise KeyError(template_id)


def main():
    subject_dir = ROOT / "data" / "uploads" / "s_7fbddea5"
    photos = sorted(p for p in subject_dir.iterdir() if p.is_file())
    print(f"Using {len(photos)} photos: {[p.name for p in photos]}")

    template = find_template("business", "bf_f_01")
    template_path = ROOT / template["template_image"]
    base_prompt = template.get("gen_prompt", template["prompt"])

    print(f"Template: {template['id']} — {template['label']}")
    print(f"Face swap enabled: {settings.face_swap_enabled}")
    print(f"Model path: {settings.face_swap_model_path}")

    worker = GeminiWorker()
    worker.connect()

    output_dir = ROOT / "experiments" / "output" / "face_swap_pipeline_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Point worker output dir to our test dir
    worker.client.output_dir = output_dir

    filepath, meta = worker.execute_generate_with_resemblance_loop(
        session_id="test_face_swap",
        prompt=base_prompt,
        photo_paths=[str(p) for p in photos],
        title="test_fs",
        template_path=str(template_path),
        progress_callback=lambda i, m, phase, detail: print(f"  [{phase}] {detail}"),
    )

    print("\nResult:", filepath)
    print("Metadata:", json.dumps(meta, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
