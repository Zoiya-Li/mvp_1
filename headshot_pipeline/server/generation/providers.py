"""Image provider interface and concrete implementations.

Business code should call ImageProvider.create_from_references / local_edit /
judge / upscale, not the underlying client directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


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


class OpenRouterProvider(ImageProvider):
    """Provider backed by OpenRouter REST API (google/gemini-3.1-flash-image-preview)."""

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path = "output",
        model: str = "google/gemini-3.1-flash-image-preview",
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
