"""Image Gateway — routes semantic image operations to concrete providers.

Business code (GeminiWorker, job_queue, etc.) should call ImageGateway methods
instead of directly instantiating OpenRouterGeminiClient or PersistentGeminiClient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..config import settings
from ..image_gateway import ImageOperation, provider_for_operation
from .providers import (
    ChromeProvider,
    ImageProvider,
    OpenRouterProvider,
    SiliconFlowError,
    SiliconFlowProvider,
)
from .smart_router import SmartModelRouter

ImageOperation = Literal[
    "COMPOSITION_SCAFFOLD",
    "CREATE_FROM_REFERENCES",
    "IDENTITY_BLEND",
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
        self._openrouter_hero: OpenRouterProvider | None = None
        self._openrouter_recovery: OpenRouterProvider | None = None
        self._siliconflow: SiliconFlowProvider | None = None
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
                judge_model=settings.openrouter_judge_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.gemini_wait_timeout,
                image_provider=settings.openrouter_image_provider,
                image_size=settings.openrouter_image_size,
                estimated_image_cost=settings.openrouter_estimated_image_cost,
                minimum_credit_balance=settings.openrouter_min_credit_balance,
                max_reference_images=settings.openrouter_max_reference_images,
            )
            self._openrouter_hero = OpenRouterProvider(
                api_key=settings.openrouter_api_key,
                output_dir=str(settings.output_dir),
                model=settings.openrouter_hero_model,
                judge_model=settings.openrouter_judge_model,
                base_url=settings.openrouter_base_url,
                timeout=settings.gemini_wait_timeout,
                image_provider=settings.openrouter_hero_image_provider,
                image_size=settings.openrouter_image_size,
                estimated_image_cost=settings.openrouter_hero_estimated_image_cost,
                minimum_credit_balance=settings.openrouter_min_credit_balance,
                max_reference_images=settings.openrouter_max_reference_images,
            )
            if (
                settings.openrouter_recovery_model
                and settings.openrouter_recovery_image_provider
            ):
                self._openrouter_recovery = OpenRouterProvider(
                    api_key=settings.openrouter_api_key,
                    output_dir=str(settings.output_dir),
                    model=settings.openrouter_recovery_model,
                    judge_model=settings.openrouter_judge_model,
                    base_url=settings.openrouter_base_url,
                    timeout=settings.gemini_wait_timeout,
                    image_provider=settings.openrouter_recovery_image_provider,
                    image_size=settings.openrouter_image_size,
                    estimated_image_cost=(
                        settings.openrouter_recovery_estimated_image_cost
                    ),
                    minimum_credit_balance=settings.openrouter_min_credit_balance,
                    max_reference_images=settings.openrouter_max_reference_images,
                )
        elif settings.gemini_backend == "siliconflow":
            if not settings.siliconflow_api_key:
                raise SiliconFlowError(
                    "SILICONFLOW_API_KEY is not set and gemini_backend is siliconflow."
                )
            self._siliconflow = SiliconFlowProvider(
                api_key=settings.siliconflow_api_key,
                output_dir=str(settings.output_dir),
                image_model=settings.siliconflow_image_model,
                text_to_image_model=settings.siliconflow_text_to_image_model,
                judge_model=settings.siliconflow_judge_model,
                base_url=settings.siliconflow_base_url,
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
        if not hasattr(self, "_smart_router"):
            self._smart_router = SmartModelRouter()
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
        chrome = getattr(self, "_chrome", None)
        siliconflow = getattr(self, "_siliconflow", None)
        openrouter = getattr(self, "_openrouter", None)
        if cap.provider == "chrome" and chrome is not None:
            return chrome
        if cap.provider == "siliconflow" and siliconflow is not None:
            return siliconflow
        if openrouter is not None:
            return openrouter
        if siliconflow is not None:
            return siliconflow
        if chrome is not None:
            return chrome
        raise RuntimeError("No image provider is configured.")

    # -- public semantic API --

    def create_composition_scaffold(
        self,
        prompt: str,
        title: str | None,
    ) -> str:
        """Create a reference-free composition that will receive identity later."""
        provider = self._provider_for("COMPOSITION_SCAFFOLD")
        return provider.create_from_references(
            prompt=prompt,
            reference_paths=[],
            template_path=None,
            title=title,
            editing_mode=False,
        )

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

    def hero_route(self) -> dict:
        """Return the concrete identity-first route used for Hero generation."""
        provider = getattr(self, "_openrouter_hero", None)
        if provider is None:
            fallback = self._provider_for("CREATE_FROM_REFERENCES")
            return {
                "provider": settings.gemini_backend,
                "model": getattr(fallback, "model", settings.gemini_model),
                "reason": "active_backend_hero_fallback",
                "estimated_cost": settings.openrouter_estimated_image_cost,
                "estimated_latency_ms": 55_000,
                "confidence": 0.5,
            }
        return {
            "provider": "openrouter",
            "model": provider.model,
            "provider_tag": provider.image_provider,
            "reason": "identity_first_hero_benchmark_winner",
            "estimated_cost": provider.estimated_image_cost,
            "estimated_latency_ms": 55_000,
            "confidence": 0.9,
        }

    def quality_route(self) -> dict:
        """Return the photoreal route used for identity-critical set images."""
        route = self.hero_route()
        if getattr(self, "_openrouter_hero", None) is not None:
            route = {
                **route,
                "reason": "photoreal_identity_quality_route",
            }
        return route

    def has_recovery_route(self) -> bool:
        """Whether a benchmark-approved alternate generation route is active."""
        return getattr(self, "_openrouter_recovery", None) is not None

    def recovery_route(self) -> dict | None:
        provider = getattr(self, "_openrouter_recovery", None)
        if provider is None:
            return None
        return {
            "provider": "openrouter",
            "model": provider.model,
            "provider_tag": provider.image_provider,
            "reason": "episode_failure_escalation",
            "estimated_cost": provider.estimated_image_cost,
            "estimated_latency_ms": 55_000,
            "confidence": 0.75,
        }

    def create_recovery_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
    ) -> str:
        """Generate through the separately benchmarked recovery capability."""
        provider = getattr(self, "_openrouter_recovery", None)
        if provider is None:
            raise RuntimeError("No benchmark-approved recovery route is configured")
        return provider.create_from_references(
            prompt=prompt,
            reference_paths=reference_paths,
            template_path=template_path,
            title=title,
            editing_mode=True,
        )

    def create_quality_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        template_path: str | None,
        title: str | None,
    ) -> str:
        """Generate a set image with the same photoreal model as Hero."""
        provider = getattr(self, "_openrouter_hero", None)
        if provider is None:
            return self.create_from_references(
                prompt=prompt,
                reference_paths=reference_paths,
                template_path=template_path,
                title=title,
                editing_mode=True,
            )
        return provider.create_from_references(
            prompt=prompt,
            reference_paths=reference_paths,
            template_path=template_path,
            title=title,
            editing_mode=True,
        )

    def create_hero_from_references(
        self,
        prompt: str,
        reference_paths: list[str],
        title: str | None,
    ) -> str:
        """Generate Hero directly from identity references with its tuned model."""
        return self.create_quality_from_references(
            prompt=prompt,
            reference_paths=reference_paths,
            template_path=None,
            title=title,
        )

    def identity_blend(
        self,
        current_image_path: str,
        reference_paths: list[str],
        blend_prompt: str,
        title: str | None,
    ) -> str:
        """Blend a real identity into an approved scaffold without reframing it."""
        provider = self._provider_for("IDENTITY_BLEND")
        return provider.local_edit(
            current_image_path=current_image_path,
            reference_paths=reference_paths,
            edit_prompt=blend_prompt,
            title=title,
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
        for p in (
            getattr(self, "_openrouter", None),
            getattr(self, "_openrouter_hero", None),
            getattr(self, "_openrouter_recovery", None),
            getattr(self, "_siliconflow", None),
            getattr(self, "_chrome", None),
        ):
            if p is not None:
                try:
                    p.end_session()
                except Exception:
                    pass

    def check_readiness(self) -> dict:
        """Probe the active generation provider once during worker startup."""
        provider = self._provider_for("CREATE_FROM_REFERENCES")
        status = provider.check_readiness()
        hero = getattr(self, "_openrouter_hero", None)
        if hero is not None:
            hero_status = hero._image_client.check_readiness()
            status = {
                **status,
                "hero_generation_ready": bool(hero_status.get("pass")),
                "hero_model": hero_status.get("model"),
            }
        recovery = getattr(self, "_openrouter_recovery", None)
        if recovery is not None:
            recovery_status = recovery._image_client.check_readiness()
            status = {
                **status,
                "recovery_generation_ready": bool(recovery_status.get("pass")),
                "recovery_model": recovery_status.get("model"),
            }
        return status
