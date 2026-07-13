"""OpenRouter Gemini client — API-driven image generation/editing.

Replaces the headless-Chrome-CDP ``persistent_client.py`` for production. The
Gemini web-UI driver (Selenium over a logged-in Chrome) was the single biggest
source of "needs constant debugging": login expiry, DOM-selector drift, VNC
login, profile locks. Driving ``gemini-3.1-flash-image`` ("Nano Banana
2") via the OpenRouter REST API removes that entire layer.

Design — STATELESS multi-turn:
  Every call is a self-contained POST to /chat/completions. We do NOT rely on
  the provider keeping a server-side conversation alive; instead each request
  re-packs the reference photos + the current generated image into the user
  message. Verified end-to-end (generate / judge / revise all return correctly).
  This trades a little extra input-token cost (reference image resent per turn)
  for robustness: no session state to lose, no multi-turn assistant-image
  round-trip to break.

Interface is deliberately a drop-in for ``PersistentGeminiClient`` so
``gemini_worker.py``'s resemblance loop needs no logic changes:
  connect() / disconnect() / ensure_gemini_page() / _new_chat()   (compat no-ops)
  start_conversation(prompt, photo_paths, photo_path, title, template_path) -> filepath
  converse(prompt, title, turn_number)                            -> filepath
  converse_text(prompt, timeout)                                  -> str   (judge)
  end_conversation()

Response shape (verified against OpenRouter):
  - generated image lives in choices[0].message.images[i].image_url.url
    as "data:image/jpeg;base64,...". message.content is null on image turns.
  - judge (text) replies live in choices[0].message.content (plain string),
    with NO image and NO reasoning blob.
  - a ~1MB encrypted ``reasoning_details`` blob accompanies image turns; we
    ignore it entirely (never read it). It cannot be excluded via the
    ``reasoning`` request param for this model.

Zero new dependencies: stdlib urllib only (httpx already in requirements for
other code; this client does not need it). Image decode uses Pillow (present).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

# Lazy import: PIL is in requirements, but importing at module load would hard
# couple this module to it. Keep it lazy so a missing PIL fails loudly at use,
# not at import — and so unit tests can monkeypatch.
def _pil():
    from PIL import Image
    return Image


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3.1-flash-image"

# Retries for transient transport errors (429 / 5xx). One retry with backoff —
# the worker's judge loop already adds a second layer of fast-retry, so we keep
# this minimal to avoid doubling long timeouts.
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_RETRY_BACKOFF_S = 3.0


class OpenRouterError(RuntimeError):
    """Raised on a non-recoverable API error (bad key, model, request shape)."""


def _b64_data_url(path: str | Path) -> str:
    """Read an image file and return a ``data:<mime>;base64,...`` URL."""
    p = Path(path)
    ext = p.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OpenRouterGeminiClient:
    """API-driven Gemini image client, drop-in for PersistentGeminiClient."""

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path = "output",
        model: str = DEFAULT_MODEL,
        timeout: int = 180,
        base_url: str = DEFAULT_BASE_URL,
        wait_timeout: int | None = None,  # compat: PersistentGeminiClient kw
        port: int | None = None,          # compat: ignored (no CDP)
    ):
        if not api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is empty — set it in .env. The API client "
                "cannot run without a key (unlike Chrome, there is no logged-in "
                "session to fall back on)."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout if timeout else (wait_timeout or 180))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Per-conversation state (stateless at the API level, but we remember
        #    which reference photos + last generated image to re-pack per call).
        self._ref_photos: list[str] = []      # user selfie paths (本人)
        self._template_path: str | None = None  # style template (风格参考)
        self._last_image_path: str | None = None  # most recent generated image
        self._in_conversation = False

    # ── compat no-ops (keep method names so gemini_worker calls still resolve) ──
    def connect(self):
        """Validate the key is present. The API has no socket to open."""
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY not set")
        print(f"✓ OpenRouter API client ready (model={self.model})")

    def check_model_access(self, timeout: int = 30) -> dict:
        """Make a low-cost text probe against the configured image model.

        A key-only check is insufficient: OpenRouter may authenticate the key
        while the selected image model is unavailable in the caller's region.
        The result is cached by the process-level worker, not called per health
        request.
        """
        response = self._post(
            [{"role": "user", "content": "Reply with exactly OK."}],
            timeout=timeout,
            modalities=["text"],
        )
        text = self._extract_text(response)
        if not text:
            raise OpenRouterError(
                f"Model access probe returned no text for {self.model}"
            )
        return {
            "pass": True,
            "provider": "openrouter",
            "model": self.model,
        }

    def disconnect(self):
        """No-op. The API has no persistent connection to close."""
        pass

    def ensure_gemini_page(self):
        """No-op compat. There is no browser page in the API path."""
        pass

    def _new_chat(self):
        """Reset per-conversation state (compat for gemini_worker session switch)."""
        self._ref_photos = []
        self._template_path = None
        self._last_image_path = None
        self._in_conversation = False

    # ── transport ──────────────────────────────────────────────────────────
    def _post(
        self,
        messages: list,
        timeout: int | None = None,
        modalities: list[str] | None = None,
        image_config: dict | None = None,
    ) -> dict:
        """POST a chat completion; return the parsed JSON. One transient retry.

        ``modalities`` and ``image_config`` are required by OpenRouter for image
        generation models; we default to text+image for generation calls and
        text-only for judge calls. See:
        https://openrouter.ai/docs/guides/overview/multimodal/image-generation
        """
        payload: dict = {"model": self.model, "messages": messages}
        if modalities:
            payload["modalities"] = modalities
        if image_config:
            payload["image_config"] = image_config
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        to = int(timeout if timeout else self.timeout)
        last_err: Exception | None = None
        for attempt in (1, 2):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=to) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                last_err = e
                detail = ""
                try:
                    detail = e.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                if e.code in _TRANSIENT_STATUS and attempt == 1:
                    print(f"  ⚠  OpenRouter {e.code} (transient), retrying in "
                          f"{_RETRY_BACKOFF_S}s … {detail}")
                    time.sleep(_RETRY_BACKOFF_S)
                    continue
                raise OpenRouterError(
                    f"OpenRouter API error {e.code}: {detail}"
                ) from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt == 1:
                    print(f"  ⚠  OpenRouter transport error ({e}), retrying in "
                          f"{_RETRY_BACKOFF_S}s …")
                    time.sleep(_RETRY_BACKOFF_S)
                    continue
                raise OpenRouterError(f"OpenRouter transport error: {e}") from e
        # Should not reach here; final attempt already raised above.
        raise OpenRouterError(f"OpenRouter request failed: {last_err}")

    @staticmethod
    def _content_blocks(text: str, image_paths: list[str]) -> list[dict]:
        """Build an OpenAI-style multimodal user content list: text first, then images."""
        blocks: list[dict] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for p in image_paths:
            if p and Path(p).exists():
                blocks.append({"type": "image_url", "image_url": {"url": _b64_data_url(p)}})
        return blocks

    def _extract_image(self, resp: dict, title: str | None) -> str:
        """Pull the first generated image from the response, save as PNG, return path."""
        msg = resp.get("choices", [{}])[0].get("message", {})
        images = msg.get("images") or []
        if not images:
            # Surface what we DID get so a model/prompt regression is debuggable.
            snippet = json.dumps(msg, ensure_ascii=False)[:400]
            raise OpenRouterError(f"No image in OpenRouter response. message={snippet}")
        url = images[0]["image_url"]["url"]
        if "," not in url:
            raise OpenRouterError(f"Unexpected image url shape: {url[:80]}")
        raw = base64.b64decode(url.split(",", 1)[1])

        # API returns JPEG. Re-encode as PNG to match the Chrome path's on-disk
        # format (job_queue stores {image_id}.png and serves it), so no downstream
        # change is needed. Pillow normalizes any odd JPEG container too.
        Image = _pil()
        img = Image.open(BytesIO(raw))
        img.load()
        stem = title if title else "gemini"
        fname = f"{stem}_{uuid.uuid4().hex[:8]}.png"
        filepath = self.output_dir / fname
        # RGBA→RGB for clean PNG JPEG-origin save; headshots are RGB anyway.
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(filepath, format="PNG")
        print(f"  📐 Generated image saved: {img.size[0]}×{img.size[1]} → {filepath.name}")
        return str(filepath)

    @staticmethod
    def _extract_text(resp: dict) -> str:
        """Pull the assistant text from a judge turn (message.content)."""
        msg = resp.get("choices", [{}])[0].get("message", {})
        text = msg.get("content")
        if isinstance(text, list):
            # Some providers nest text blocks; join any text parts.
            text = "\n".join(
                blk.get("text", "") for blk in text if isinstance(blk, dict)
            )
        return (text or "").strip()

    # ── public, drop-in interface ──────────────────────────────────────────
    def start_conversation(
        self,
        prompt: str,
        photo_paths: list[str] | None = None,
        photo_path: str | None = None,
        title: str | None = None,
        template_path: str | None = None,
        editing_mode: bool = False,
    ) -> str:
        """Initial generation from reference photos (+ optional style template).

        By default the image order is [template, ...user_photos] (generation
        framing). When ``editing_mode=True`` the order flips to
        [...user_photos, template] so the model treats the user's selfie as the
        image to edit and the template as the style reference.
        """
        # Backward compat: accept a single photo_path.
        if photo_paths is None and photo_path is not None:
            photo_paths = [photo_path]
        elif photo_paths is None:
            photo_paths = []

        self._ref_photos = [p for p in photo_paths if p and Path(p).exists()]
        self._template_path = (
            template_path if template_path and Path(template_path).exists() else None
        )
        self._in_conversation = True

        if editing_mode:
            ordered = self._ref_photos + ([self._template_path] if self._template_path else [])
        else:
            ordered = ([self._template_path] if self._template_path else []) + self._ref_photos
        print(f"  📷 Reference set: {len(ordered)} image(s) "
              f"(template={'yes' if self._template_path else 'no'}, "
              f"user_photos={len(self._ref_photos)}, editing={editing_mode})")

        messages = [{"role": "user", "content": self._content_blocks(prompt, ordered)}]
        resp = self._post(
            messages,
            modalities=["image", "text"],
            image_config=self._image_config(),
        )
        filepath = self._extract_image(resp, title=title or "turn_1")
        self._last_image_path = filepath
        return filepath

    def converse(self, prompt: str, title: str | None = None,
                 turn_number: int | None = None) -> str:
        """Revise the current generated image (stateless re-pack).

        Re-packs [last_generated_image, ...user_photos] so the model sees the
        image to edit AND the person it must stay faithful to. The style template
        is intentionally omitted here: a revision tunes the existing image
        (which already carries the style), it does not re-seed the style.
        """
        if not self._in_conversation:
            raise OpenRouterError("No active conversation. Call start_conversation() first.")
        turn = turn_number or 2
        print(f"\n🔄 Turn {turn}: {prompt[:80]}…")

        ordered = ([self._last_image_path] if self._last_image_path else []) + self._ref_photos
        messages = [{"role": "user", "content": self._content_blocks(prompt, ordered)}]
        resp = self._post(
            messages,
            modalities=["image", "text"],
            image_config=self._image_config(),
        )
        filepath = self._extract_image(resp, title=title or f"turn_{turn}")
        self._last_image_path = filepath
        print(f"✓ Turn {turn} saved: {filepath}")
        return filepath

    def converse_text(self, prompt: str, timeout: int | None = None) -> str:
        """Judge turn: ask for a TEXT verdict (no image). Returns the reply text.

        Packs [last_generated_image, ref_photos[0]] — i.e. the generated headshot
        and ONE anchor reference photo — so the JUDGE_PROMPT's "第一张图片(你生成的)
        / 第二张图片(本人参考)" maps to exactly two images. Multiple reference
        photos would make "第二张" ambiguous; the first uploaded photo is the
        canonical identity anchor.
        """
        if not self._in_conversation:
            raise OpenRouterError("No active conversation. Call start_conversation() first.")

        anchor = self._ref_photos[:1]  # first user photo = identity anchor
        ordered = ([self._last_image_path] if self._last_image_path else []) + anchor
        messages = [{"role": "user", "content": self._content_blocks(prompt, ordered)}]
        resp = self._post(messages, timeout=timeout, modalities=["text"])
        text = self._extract_text(resp)
        if not text:
            raise OpenRouterError(
                "Judge turn returned no text. The model may have emitted an image "
                "instead of a score; check the prompt/model."
            )
        return text

    def end_conversation(self):
        """End the current conversation (clears re-pack state)."""
        self._in_conversation = False
        self._last_image_path = None
        print("✓ Conversation ended.")

    def _image_config(self) -> dict:
        """Return the image generation config from settings if available."""
        # Avoid a hard import dependency on server.config so this module stays
        # usable in standalone scripts; fall back to sensible headshot defaults.
        try:
            from .config import settings
            return {
                "aspect_ratio": settings.gemini_image_aspect_ratio,
                "image_size": settings.gemini_image_size,
            }
        except Exception:
            return {"aspect_ratio": "3:4", "image_size": "1K"}
