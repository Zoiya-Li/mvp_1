# legacy/ — Deprecated Chrome/Gemini-web era artifacts

These files are **not imported by the production server** (`server/`). They are kept
here for reference and git-history continuity, not as active dependencies.

## What's here

| File(s) | What it was |
|---|---|
| `batch_headshot.py` | Legacy batch headshot CLI (pre-FastAPI). Zero external references. |
| `generate_templates.py` | One-off script that produced `../templates/*.png` style images. Its output already lives in `templates/`; the script is not re-run by the pipeline. |
| `wm_alpha_*.npy`, `wm_binary_mask_v2.npy`, `wm_full_mask.npy`, `wm_mask_60_572.npy`, `wm_mask_80_837.npy`, `wm_mask_final.{npy,png}`, `wm_mask_template.npy`, `wm_mask_universal.{npy,png}`, `wm_mask_v3.npy` | Watermark-mask **experiments** from the Chrome/Gemini-web era. None of them are loaded by any code. |

## Why they are dead

The pipeline used to drive `gemini.google.com` through a logged-in Chrome session,
which watermarked its outputs — hence the watermark-removal experiments. After the
pivot to the **SiliconFlow / OpenRouter image API** (default backend `siliconflow`),
generation returns clean images and watermark removal is never invoked. The current
delivery pipeline applies an AI-provenance label (`server/delivery_label.py`), which
is unrelated to third-party watermark removal.

## What is NOT here (still live, kept at the repo root)

The Chrome backend is still a *supported* config option, so its coupled unit stays at
the `headshot_pipeline/` root:

- `persistent_client.py` — imported by `server/generation/providers.py` (ChromeProvider)
- `watermark_remover.py` — imported by `persistent_client.py`
- `wm_template.npy`, `wm_shape_mask.npy` — loaded by `watermark_remover.py`

Fully retiring the Chrome backend (moving this trio + removing ChromeProvider) is a
separate, larger change — see the project plan.

## Restoring something

These files were moved with `git mv`, so history is preserved. To restore any of them
to the root, e.g. to re-run `generate_templates.py`:

```bash
git mv legacy/generate_templates.py .
```

Note: `generate_templates.py` does `from persistent_client import ...` and
`from watermark_remover import ...` (flat root-level imports), so it only runs from the
repo root, not from inside `legacy/`.
