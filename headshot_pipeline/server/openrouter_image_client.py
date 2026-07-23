"""OpenRouter dedicated Image API client.

Generation and editing use ``POST /api/v1/images``. Text/image QA remains a
separate concern so an image generator never grades its own output and image
models are never probed with unsupported text-only requests.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from .openrouter_client import OpenRouterError


_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
# FLUX.2 Pro limits input plus output to 9MP. Five 0.95MP references
# (four identity views plus an optional style image) leave enough room for the
# configured 1728x2304 (3.98MP) portrait output.
_MAX_REFERENCE_PIXELS = 950_000


class OpenRouterImageError(OpenRouterError):
    """Raised when the dedicated image API cannot complete a request."""


class OpenRouterImageClient:
    """Small stdlib client for reference-based image generation."""

    def __init__(
        self,
        *,
        api_key: str,
        output_dir: str | Path,
        model: str,
        base_url: str,
        timeout: int,
        image_size: str,
        provider_tag: str,
        estimated_image_cost: float,
        minimum_credit_balance: float,
        max_reference_images: int,
    ) -> None:
        if not api_key:
            raise OpenRouterImageError("OPENROUTER_API_KEY is empty")
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.image_size = image_size
        self.provider_tag = provider_tag.strip()
        self.estimated_image_cost = float(estimated_image_cost)
        self.minimum_credit_balance = float(minimum_credit_balance)
        self.max_reference_images = max(1, int(max_reference_images))
        self.last_usage: dict = {}

    @staticmethod
    def _data_uri(path: str | Path) -> str:
        source = Path(path)
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                pixel_count = image.width * image.height
                if pixel_count > _MAX_REFERENCE_PIXELS:
                    scale = (_MAX_REFERENCE_PIXELS / pixel_count) ** 0.5
                    image = image.resize(
                        (
                            max(1, round(image.width * scale)),
                            max(1, round(image.height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                output = BytesIO()
                image.save(
                    output,
                    "JPEG",
                    quality=92,
                    optimize=True,
                    progressive=False,
                )
        except Exception as exc:
            raise OpenRouterImageError(
                f"Reference image could not be normalized: {source.name}: {exc}"
            ) from exc
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _request_json(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        timeout: int | None = None,
        retries: int = 1,
    ) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        method = "POST" if payload is not None else "GET"
        for attempt in range(retries + 1):
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method=method,
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout or self.timeout
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1200]
                if exc.code not in _TRANSIENT_STATUS or attempt == retries:
                    raise OpenRouterImageError(
                        f"OpenRouter {endpoint} failed with HTTP {exc.code}: {detail}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == retries:
                    raise OpenRouterImageError(
                        f"OpenRouter {endpoint} request failed: {exc}"
                    ) from exc
            time.sleep(0.75 * (2**attempt))
        raise OpenRouterImageError(f"OpenRouter {endpoint} request failed")

    def credit_status(self) -> dict:
        response = self._request_json("credits", retries=1)
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            raise OpenRouterImageError("OpenRouter credits response is invalid")
        try:
            purchased = float(data["total_credits"])
            used = float(data["total_usage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenRouterImageError("OpenRouter credits response is incomplete") from exc
        return {
            "total_credits": purchased,
            "total_usage": used,
            "remaining": round(purchased - used, 6),
        }

    def require_credit_balance(self, minimum: float | None = None) -> dict:
        status = self.credit_status()
        required = self.minimum_credit_balance if minimum is None else float(minimum)
        if status["remaining"] < required:
            raise OpenRouterImageError(
                "OpenRouter insufficient credit balance: "
                f"${status['remaining']:.4f} remaining, ${required:.4f} required"
            )
        return status

    def check_readiness(self) -> dict:
        model = self._request_json(
            f"images/models/{self.model}/endpoints", retries=1
        )
        endpoints = model.get("endpoints") if isinstance(model, dict) else None
        if not isinstance(endpoints, list) or not endpoints:
            raise OpenRouterImageError(
                f"Configured OpenRouter image model is unavailable: {self.model}"
            )
        if not any(
            "input_references" in (item.get("supported_parameters") or {})
            for item in endpoints
            if isinstance(item, dict)
        ):
            raise OpenRouterImageError(
                f"Configured image model does not support references: {self.model}"
            )
        credits = self.require_credit_balance()
        return {
            "pass": True,
            "provider": "openrouter",
            "model": self.model,
            "remaining_credits": credits["remaining"],
            "minimum_credit_balance": self.minimum_credit_balance,
        }

    def check_text_model(self, model: str) -> dict:
        response = self._request_json(f"models/{model}/endpoints", retries=1)
        data = response.get("data") if isinstance(response, dict) else None
        endpoints = data.get("endpoints") if isinstance(data, dict) else None
        if not isinstance(endpoints, list) or not endpoints:
            raise OpenRouterImageError(
                f"Configured OpenRouter judge model is unavailable: {model}"
            )
        return {"pass": True, "model": model}

    def _save_image(self, response: dict, title: str | None) -> str:
        data = response.get("data") if isinstance(response, dict) else None
        item = data[0] if isinstance(data, list) and data else None
        encoded = item.get("b64_json") if isinstance(item, dict) else None
        if not isinstance(encoded, str) or not encoded:
            raise OpenRouterImageError("OpenRouter image API returned no image bytes")
        try:
            raw = base64.b64decode(encoded)
            image = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise OpenRouterImageError(f"OpenRouter returned an invalid image: {exc}") from exc

        safe_title = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in (title or "portrait")
        ).strip("_")[:60] or "portrait"
        destination = self.output_dir / f"{safe_title}_{uuid.uuid4().hex[:10]}.png"
        image.save(destination, "PNG")
        self.last_usage = response.get("usage") or {}
        return str(destination)

    def generate(
        self,
        *,
        prompt: str,
        reference_paths: list[str],
        title: str | None,
    ) -> str:
        # Check the cost of the next image, not the project-level reserve. The
        # reserve is enforced by readiness before a project is admitted.
        self.require_credit_balance(self.estimated_image_cost)
        existing = [
            str(Path(path)) for path in reference_paths
            if path and Path(path).is_file()
        ][: self.max_reference_images]
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "size": self.image_size,
            "n": 1,
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {"url": self._data_uri(path)},
                }
                for path in existing
            ],
            "provider": {"allow_fallbacks": False},
        }
        if self.provider_tag:
            payload["provider"]["only"] = [self.provider_tag]
        response = self._request_json("images", payload, retries=1)
        return self._save_image(response, title)
