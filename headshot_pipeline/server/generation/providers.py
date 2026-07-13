"""Image provider interface and concrete implementations.

Business code should call ImageProvider.create_from_references / local_edit /
judge / upscale, not the underlying client directly.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class ImageProvider(ABC):
    """Abstract image-generation provider.

    All methods return filesystem paths (str) to saved images, or text (str) for
    judge turns.  The provider manages its own per-call state; callers should
    not assume multi-turn conversation state survives between calls.
    """

    @abstractmethod
    def create_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
        editing_mode: bool = True,
    ) -> str:
        """Generate a new image from reference photos + optional style template.

        Returns path to the saved generated image.
        """

    @abstractmethod
    def local_edit(
        self,
        current_image_path: str,
        reference_paths: list[str],
        edit_prompt: str,
        title: str | None,
    ) -> str:
        """Edit an existing generated image (local artifact fix, identity tweak).

        Returns path to the saved edited image.
        """

    @abstractmethod
    def judge(
        self,
        current_image_path: str,
        reference_paths: list[str],
        judge_prompt: str,
        timeout: int | None = None,
    ) -> str:
        """Ask the model for a text verdict on the current image.

        Returns the raw text response (e.g. JSON or free-form score).
        """

    @abstractmethod
    def upscale(self, image_path: str) -> str:
        """Upscale an image.  Default implementation returns the same path."""

    @abstractmethod
    def end_session(self) -> None:
        """Clean up any provider-side session state."""

    @abstractmethod
    def check_readiness(self) -> dict:
        """Verify the configured provider/model can accept requests."""


class OpenRouterProvider(ImageProvider):
    """Provider backed by OpenRouter REST API (google/gemini-3.1-flash-image)."""

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path = "output",
        model: str = "google/gemini-3.1-flash-image",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 180,
    ):
        from ..openrouter_client import OpenRouterGeminiClient

        self._client = OpenRouterGeminiClient(
            api_key=api_key,
            output_dir=str(output_dir),
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    # -- ImageProvider interface --

    def create_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
        editing_mode: bool = True,
    ) -> str:
        self._client._new_chat()
        self._client._ref_photos = [p for p in reference_paths if p and Path(p).exists()]
        self._client._template_path = (
            template_path if template_path and Path(template_path).exists() else None
        )
        self._client._in_conversation = True
        return self._client.start_conversation(
            prompt=prompt,
            photo_paths=reference_paths,
            title=title,
            template_path=template_path,
            editing_mode=editing_mode,
        )

    def local_edit(
        self,
        current_image_path: str,
        reference_paths: list[str],
        edit_prompt: str,
        title: str | None,
    ) -> str:
        # Ensure the client knows about the current image and references.
        self._client._last_image_path = current_image_path
        self._client._ref_photos = [p for p in reference_paths if p and Path(p).exists()]
        return self._client.converse(
            prompt=edit_prompt,
            title=title,
            turn_number=2,
        )

    def judge(
        self,
        current_image_path: str,
        reference_paths: list[str],
        judge_prompt: str,
        timeout: int | None = None,
    ) -> str:
        self._client._last_image_path = current_image_path
        self._client._ref_photos = [p for p in reference_paths if p and Path(p).exists()]
        self._client._in_conversation = True
        return self._client.converse_text(
            prompt=judge_prompt,
            timeout=timeout,
        )

    def upscale(self, image_path: str) -> str:
        # TODO: wire to RealESRGAN or dedicated upscale model.
        return image_path

    def end_session(self) -> None:
        self._client.end_conversation()

    def check_readiness(self) -> dict:
        return self._client.check_model_access()


class SiliconFlowError(RuntimeError):
    """Raised when SiliconFlow rejects or cannot complete a request."""


class SiliconFlowProvider(ImageProvider):
    """SiliconFlow image-edit provider with a separate VLM quality judge."""

    MAX_EDIT_IMAGES = 3

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path = "output",
        image_model: str = "Qwen/Qwen-Image-Edit-2509",
        text_to_image_model: str = "Qwen/Qwen-Image",
        judge_model: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: int = 180,
    ) -> None:
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_model = image_model
        self.text_to_image_model = text_to_image_model
        self.judge_model = judge_model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _existing(paths: list[str]) -> list[str]:
        return [path for path in paths if path and Path(path).is_file()]

    @staticmethod
    def _data_uri(path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _request_json(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        timeout: int | None = None,
        retries: int = 2,
    ) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                    raise SiliconFlowError(
                        f"SiliconFlow {endpoint} failed with HTTP {exc.code}: {detail}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == retries:
                    raise SiliconFlowError(
                        f"SiliconFlow {endpoint} request failed: {exc}"
                    ) from exc
            time.sleep(0.75 * (2**attempt))
        raise SiliconFlowError(f"SiliconFlow {endpoint} request failed")

    def _save_generated_image(self, response: dict, title: str | None) -> str:
        images = response.get("images") or response.get("data") or []
        if not images or not isinstance(images[0], dict):
            raise SiliconFlowError("SiliconFlow returned no generated image")
        item = images[0]
        raw: bytes
        if item.get("url"):
            try:
                with urllib.request.urlopen(item["url"], timeout=self.timeout) as image_response:
                    raw = image_response.read()
            except (urllib.error.URLError, TimeoutError) as exc:
                raise SiliconFlowError(f"Could not download generated image: {exc}") from exc
        elif item.get("b64_json"):
            raw = base64.b64decode(item["b64_json"])
        else:
            raise SiliconFlowError("SiliconFlow image response has neither url nor b64_json")

        safe_title = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in (title or "portrait")
        ).strip("_")[:60] or "portrait"
        destination = self.output_dir / f"{safe_title}_{uuid.uuid4().hex[:10]}.png"
        try:
            from io import BytesIO

            with Image.open(BytesIO(raw)) as image:
                image.convert("RGB").save(destination, "PNG")
        except Exception as exc:
            raise SiliconFlowError(f"Generated image is invalid: {exc}") from exc
        return str(destination)

    def _generate(
        self,
        prompt: str,
        input_paths: list[str],
        title: str | None,
    ) -> str:
        if input_paths:
            payload: dict = {"model": self.image_model, "prompt": prompt}
            for index, path in enumerate(input_paths[: self.MAX_EDIT_IMAGES], start=1):
                key = "image" if index == 1 else f"image{index}"
                payload[key] = self._data_uri(path)
        else:
            payload = {
                "model": self.text_to_image_model,
                "prompt": prompt,
                "image_size": "1140x1472",
            }
        response = self._request_json("images/generations", payload)
        return self._save_generated_image(response, title)

    def create_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
        editing_mode: bool = True,
    ) -> str:
        references = self._existing(reference_paths)
        template = template_path if template_path and Path(template_path).is_file() else None
        if template:
            selected = references[:2] + [template]
            role_prompt = (
                "Images 1 and 2 are identity references of the same person. "
                "Preserve that person's facial identity. Image 3 is a composition/style "
                "template only; do not copy its person's identity."
            )
        else:
            selected = references[: self.MAX_EDIT_IMAGES]
            role_prompt = (
                "All supplied images are identity references of the same person. "
                "Preserve their facial identity consistently."
            )
        return self._generate(f"{role_prompt}\n\n{prompt}", selected, title)

    def local_edit(
        self,
        current_image_path: str,
        reference_paths: list[str],
        edit_prompt: str,
        title: str | None,
    ) -> str:
        if not Path(current_image_path).is_file():
            raise SiliconFlowError(f"Current image does not exist: {current_image_path}")
        selected = [current_image_path] + self._existing(reference_paths)[:2]
        prompt = (
            "Image 1 is the portrait to edit. Images 2 and 3, when supplied, are identity "
            "references. Apply only the requested correction while preserving identity, "
            "pose, framing, clothing, and background unless explicitly requested.\n\n"
            f"{edit_prompt}"
        )
        return self._generate(prompt, selected, title)

    def judge(
        self,
        current_image_path: str,
        reference_paths: list[str],
        judge_prompt: str,
        timeout: int | None = None,
    ) -> str:
        if not Path(current_image_path).is_file():
            raise SiliconFlowError(f"Current image does not exist: {current_image_path}")
        content: list[dict] = [{"type": "text", "text": judge_prompt}]
        for path in [current_image_path] + self._existing(reference_paths)[:3]:
            content.append({"type": "image_url", "image_url": {"url": self._data_uri(path)}})
        response = self._request_json(
            "chat/completions",
            {
                "model": self.judge_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 800,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        choices = response.get("choices") or []
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        if not isinstance(text, str) or not text.strip():
            raise SiliconFlowError("SiliconFlow judge returned no text")
        return text

    def upscale(self, image_path: str) -> str:
        return image_path

    def end_session(self) -> None:
        return None

    def check_readiness(self) -> dict:
        response = self._request_json("models", retries=1)
        available = {
            item.get("id") for item in response.get("data", []) if isinstance(item, dict)
        }
        missing = [
            model
            for model in (self.image_model, self.judge_model)
            if model not in available
        ]
        if missing:
            raise SiliconFlowError(
                "Configured SiliconFlow model is unavailable: " + ", ".join(missing)
            )
        return {
            "pass": True,
            "provider": "siliconflow",
            "model": self.image_model,
            "judge_model": self.judge_model,
        }


class ChromeProvider(ImageProvider):
    """Legacy provider backed by a logged-in Chrome via CDP."""

    def __init__(
        self,
        port: int = 9222,
        output_dir: str | Path = "output",
        wait_timeout: int = 120,
    ):
        from persistent_client import PersistentGeminiClient

        self._client = PersistentGeminiClient(
            port=port,
            output_dir=str(output_dir),
            wait_timeout=wait_timeout,
        )

    def create_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
        editing_mode: bool = True,
    ) -> str:
        self._client._new_chat()
        self._client._ref_photos = [p for p in reference_paths if p and Path(p).exists()]
        self._client._template_path = (
            template_path if template_path and Path(template_path).exists() else None
        )
        self._client._in_conversation = True
        return self._client.start_conversation(
            prompt=prompt,
            photo_paths=reference_paths,
            title=title,
            template_path=template_path,
            editing_mode=editing_mode,
        )

    def local_edit(
        self,
        current_image_path: str,
        reference_paths: list[str],
        edit_prompt: str,
        title: str | None,
    ) -> str:
        self._client._last_image_path = current_image_path
        self._client._ref_photos = [p for p in reference_paths if p and Path(p).exists()]
        return self._client.converse(
            prompt=edit_prompt,
            title=title,
            turn_number=2,
        )

    def judge(
        self,
        current_image_path: str,
        reference_paths: list[str],
        judge_prompt: str,
        timeout: int | None = None,
    ) -> str:
        self._client._last_image_path = current_image_path
        self._client._ref_photos = [p for p in reference_paths if p and Path(p).exists()]
        self._client._in_conversation = True
        return self._client.converse_text(
            prompt=judge_prompt,
            timeout=timeout,
        )

    def upscale(self, image_path: str) -> str:
        return image_path

    def end_session(self) -> None:
        self._client.end_conversation()

    def check_readiness(self) -> dict:
        self._client.ensure_gemini_page()
        return {
            "pass": True,
            "provider": "chrome",
            "model": "gemini-web-ui",
        }
