"""Image Gateway — routes semantic image operations to concrete providers.

Business code (GeminiWorker, job_queue, etc.) should call ImageGateway methods
instead of directly instantiating OpenRouterGeminiClient or PersistentGeminiClient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import settings
from ..image_gateway import ImageOperation, provider_for_operation
from .providers import ChromeProvider, ImageProvider, OpenRouterProvider
from .smart_router import SmartModelRouter

ImageOperation = Literal[
    "CREATE_FROM_REFERENCES",
    "LOCAL_EDIT",
    "IDENTITY_REPAIR",
    "UPSCALE",
    "FINAL_RENDER",
]


class ImageGateway:
    """Router for image-generation operations.

    Holds provider instances and routes each operation to the right backend.
    """

    def __init__(self) -> None:
        self._openrouter: OpenRouterProvider | None = None
        self._chrome: ChromeProvider | None = None
        self._smart_router = SmartModelRouter()
        self._init_providers()

    def _init_providers(self) -> None:
        if settings.gemini_backend == "openrouter":
            if not settings.openrouter_api_key:
                from ..openrouter_client import OpenRouterError
                raise OpenRouterError(
                    "OPENROUTER_API_KEY is not set and gemini_backend is openrouter."
                )
            self._openrouter = OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                output_dir=str(settings.output_dir),
                model=settings.gemini_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.gemini_wait_timeout,
            )
        elif settings.gemini_backend == "chrome":
            self._chrome = ChromeProvider(
                port=settings.chrome_cdp_port,
                output_dir=str(settings.output_dir),
                wait_timeout=settings.chrome_wait_timeout,
            )
        else:
            raise RuntimeError(f"Unknown gemini_backend: {settings.gemini_backend}")

    # -- smart routing --

    def route_by_task(
        self,
        task_type: str,
        shot_spec: dict | None = None,
        user_tier: str = "standard",
        budget_remaining: float | None = None,
    ) -> dict:
        """Return smart routing decision for a task.

        This is the entry point for task-aware model selection.
        Business code should call this before operations to log routing decisions.
        """
        decision = self._smart_router.route_for_task(
            task_type=task_type,  # type: ignore[arg-type]
            shot_spec=shot_spec,
            user_tier=user_tier,
            budget_remaining=budget_remaining,
        )
        return {
            "provider": decision.provider,
            "model": decision.model,
            "reason": decision.reason,
            "estimated_cost": decision.estimated_cost,
            "estimated_latency_ms": decision.estimated_latency_ms,
            "confidence": decision.confidence,
        }

    # -- provider selection --

    def _provider_for(self, operation: ImageOperation) -> ImageProvider:
        """Return the active provider for a given operation."""
        cap = provider_for_operation(operation)
        if cap.provider == "chrome" and self._chrome is not None:
            return self._chrome
        if self._openrouter is not None:
            return self._openrouter
        if self._chrome is not None:
            return self._chrome
        raise RuntimeError("No image provider is configured.")

    # -- public semantic API --

    def create_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
        editing_mode: bool = True,
    ) -> str:
        """Generate a new image from reference photos + optional style template."""
        provider = self._provider_for("CREATE_FROM_REFERENCES")
        return provider.create_from_references(
            prompt=prompt,
            reference_paths=reference_paths,
            template_path=template_path,
            title=title,
            editing_mode=editing_mode,
        )

    def local_edit(
        self,
        current_image_path: str,
        reference_paths: list[str],
        edit_prompt: str,
        title: str | None,
    ) -> str:
        """Edit an existing generated image (local artifact fix, identity tweak)."""
        provider = self._provider_for("LOCAL_EDIT")
        return provider.local_edit(
            current_image_path=current_image_path,
            reference_paths=reference_paths,
            edit_prompt=edit_prompt,
            title=title,
        )

    def judge(
        self,
        current_image_path: str,
        reference_paths: list[str],
        judge_prompt: str,
        timeout: int | None = None,
    ) -> str:
        """Ask the model for a text verdict on the current image."""
        provider = self._provider_for("CREATE_FROM_REFERENCES")
        return provider.judge(
            current_image_path=current_image_path,
            reference_paths=reference_paths,
            judge_prompt=judge_prompt,
            timeout=timeout,
        )

    def upscale(self, image_path: str) -> str:
        """Upscale an image.  Currently a no-op; future versions will use a dedicated model."""
        provider = self._provider_for("UPSCALE")
        return provider.upscale(image_path)

    def end_session(self) -> None:
        """Clean up all provider-side session state."""
        for p in (self._openrouter, self._chrome):
            if p is not None:
                try:
                    p.end_session()
                except Exception:
                    pass
