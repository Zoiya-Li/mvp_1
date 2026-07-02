"""Smart Model Router — task-aware provider selection for ImageGateway.

Extends the basic ImageGateway with intelligent routing that selects the best
provider/model for each task based on:
- Task type (hero face, full body, edit, upscale)
- Provider capabilities (identity stability, composition strength, editing support)
- Cost/quality/latency trade-offs
- User tier (free vs paid)

This is the bridge from "single model for everything" to "right model for each job".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import settings
from ..image_gateway import ImageOperation, ProviderCapabilities


TaskType = Literal[
    "hero_face",       # Close-up portrait, highest identity priority
    "half_body",       # Medium shot, balanced identity + composition
    "full_body",       # Full body, composition priority
    "environmental",   # Wide shot with background
    "local_edit",      # Inpainting / artifact removal
    "identity_repair", # Face-specific restoration
    "upscale",         # HD enhancement
    "final_render",    # Delivery packaging
]


@dataclass(frozen=True)
class TaskProfile:
    """Characteristics of a generation task that influence model selection."""
    task_type: TaskType
    identity_priority: float  # 0.0-1.0, how much identity matters
    composition_priority: float  # 0.0-1.0, how much pose/background matters
    editing_required: bool
    latency_sensitive: bool  # True for hero preview (user waiting)
    cost_sensitive: bool  # True for free tier


@dataclass
class RoutingDecision:
    """Result of a smart routing decision."""
    provider: str
    model: str
    reason: str
    estimated_cost: float
    estimated_latency_ms: int
    confidence: float  # 0.0-1.0


class SmartModelRouter:
    """Task-aware model router for ImageGateway.

    Usage:
        router = SmartModelRouter()
        decision = router.route_for_task(
            task_type="hero_face",
            shot_spec={"shot_type": "closeup"},
            user_tier="standard",
        )
        # decision.provider = "openrouter"
        # decision.model = "google/gemini-3.1-flash-image-preview"
        # decision.reason = "hero_face requires high identity fidelity"
    """

    # Task profiles — define what each task needs
    TASK_PROFILES: dict[TaskType, TaskProfile] = {
        "hero_face": TaskProfile(
            task_type="hero_face",
            identity_priority=0.95,
            composition_priority=0.40,
            editing_required=False,
            latency_sensitive=True,
            cost_sensitive=False,
        ),
        "half_body": TaskProfile(
            task_type="half_body",
            identity_priority=0.85,
            composition_priority=0.60,
            editing_required=False,
            latency_sensitive=False,
            cost_sensitive=False,
        ),
        "full_body": TaskProfile(
            task_type="full_body",
            identity_priority=0.70,
            composition_priority=0.85,
            editing_required=False,
            latency_sensitive=False,
            cost_sensitive=False,
        ),
        "environmental": TaskProfile(
            task_type="environmental",
            identity_priority=0.60,
            composition_priority=0.90,
            editing_required=False,
            latency_sensitive=False,
            cost_sensitive=False,
        ),
        "local_edit": TaskProfile(
            task_type="local_edit",
            identity_priority=0.80,
            composition_priority=0.30,
            editing_required=True,
            latency_sensitive=False,
            cost_sensitive=False,
        ),
        "identity_repair": TaskProfile(
            task_type="identity_repair",
            identity_priority=1.0,
            composition_priority=0.20,
            editing_required=True,
            latency_sensitive=False,
            cost_sensitive=False,
        ),
        "upscale": TaskProfile(
            task_type="upscale",
            identity_priority=0.90,
            composition_priority=0.50,
            editing_required=False,
            latency_sensitive=False,
            cost_sensitive=False,
        ),
        "final_render": TaskProfile(
            task_type="final_render",
            identity_priority=0.50,
            composition_priority=0.50,
            editing_required=False,
            latency_sensitive=False,
            cost_sensitive=True,
        ),
    }

    # Provider scoring weights
    IDENTITY_WEIGHT = 0.40
    COMPOSITION_WEIGHT = 0.25
    COST_WEIGHT = 0.20
    LATENCY_WEIGHT = 0.15

    def __init__(self) -> None:
        self._providers: list[ProviderCapabilities] = []
        self._init_providers()

    def _init_providers(self) -> None:
        """Register available providers based on settings."""
        backend = getattr(settings, "gemini_backend", "openrouter")
        if backend == "openrouter":
            from ..image_gateway import OPENROUTER_GEMINI_CAPABILITIES
            self._providers.append(OPENROUTER_GEMINI_CAPABILITIES)
        elif backend == "chrome":
            from ..image_gateway import CHROME_GEMINI_CAPABILITIES
            self._providers.append(CHROME_GEMINI_CAPABILITIES)
        # Local providers are always available
        from ..image_gateway import (
            LOCAL_IDENTITY_REPAIR_CAPABILITIES,
            LOCAL_UPSCALE_CAPABILITIES,
            LOCAL_FINAL_RENDER_CAPABILITIES,
        )
        self._providers.append(LOCAL_IDENTITY_REPAIR_CAPABILITIES)
        self._providers.append(LOCAL_UPSCALE_CAPABILITIES)
        self._providers.append(LOCAL_FINAL_RENDER_CAPABILITIES)

    def route_for_task(
        self,
        task_type: TaskType,
        shot_spec: dict | None = None,
        user_tier: str = "standard",
        budget_remaining: float | None = None,
    ) -> RoutingDecision:
        """Select the best provider for a given task.

        Args:
            task_type: The kind of task being performed
            shot_spec: Optional shot specification from shot planner
            user_tier: User's subscription tier (free, standard, pro)
            budget_remaining: Remaining API budget for this session
        """
        profile = self.TASK_PROFILES.get(task_type)
        if profile is None:
            profile = self._infer_profile_from_shot_spec(shot_spec)

        # Score each provider
        best_score = -1.0
        best_provider = None
        best_reason = ""

        for provider in self._providers:
            score, reason = self._score_provider(provider, profile, user_tier, budget_remaining)
            if score > best_score:
                best_score = score
                best_provider = provider
                best_reason = reason

        if best_provider is None:
            # Fallback to default
            from ..image_gateway import provider_for_operation
            cap = provider_for_operation("CREATE_FROM_REFERENCES")
            return RoutingDecision(
                provider=cap.provider,
                model=cap.model,
                reason="fallback_no_provider_scored",
                estimated_cost=cap.estimated_cost,
                estimated_latency_ms=cap.average_latency_ms,
                confidence=0.0,
            )

        return RoutingDecision(
            provider=best_provider.provider,
            model=best_provider.model,
            reason=best_reason,
            estimated_cost=best_provider.estimated_cost,
            estimated_latency_ms=best_provider.average_latency_ms,
            confidence=min(1.0, best_score),
        )

    def _score_provider(
        self,
        provider: ProviderCapabilities,
        profile: TaskProfile,
        user_tier: str,
        budget_remaining: float | None,
    ) -> tuple[float, str]:
        """Score a provider for a task profile. Returns (score, reason)."""
        score = 0.0
        reasons = []

        # Identity capability
        if provider.supports_high_fidelity and profile.identity_priority > 0.7:
            score += self.IDENTITY_WEIGHT * profile.identity_priority
            reasons.append("high_fidelity_identity")
        elif profile.identity_priority > 0.7:
            score += self.IDENTITY_WEIGHT * profile.identity_priority * 0.5
            reasons.append("moderate_identity")
        else:
            score += self.IDENTITY_WEIGHT * 0.5
            reasons.append("standard_identity")

        # Composition capability (portrait ratio support)
        if provider.supports_portrait_ratio and profile.composition_priority > 0.5:
            score += self.COMPOSITION_WEIGHT * profile.composition_priority
            reasons.append("portrait_ratio")
        else:
            score += self.COMPOSITION_WEIGHT * 0.3

        # Cost efficiency
        cost_score = self._cost_score(provider, profile, user_tier, budget_remaining)
        score += self.COST_WEIGHT * cost_score
        if cost_score > 0.8:
            reasons.append("cost_efficient")
        elif cost_score < 0.3:
            reasons.append("cost_expensive")

        # Latency
        latency_score = self._latency_score(provider, profile)
        score += self.LATENCY_WEIGHT * latency_score
        if latency_score > 0.8:
            reasons.append("fast")

        # Editing capability
        if profile.editing_required and provider.supports_mask_edit:
            score += 0.15
            reasons.append("supports_mask_edit")
        elif profile.editing_required and not provider.supports_mask_edit:
            score -= 0.1
            reasons.append("no_mask_edit")

        # Multi-reference support
        if provider.supports_multiple_references and profile.identity_priority > 0.7:
            score += 0.1
            reasons.append("multi_reference")

        return score, ";".join(reasons) if reasons else "default"

    def _cost_score(
        self,
        provider: ProviderCapabilities,
        profile: TaskProfile,
        user_tier: str,
        budget_remaining: float | None,
    ) -> float:
        """Return 0.0-1.0 cost efficiency score."""
        cost = provider.estimated_cost
        if cost <= 0:
            return 1.0  # Free is always efficient

        # Paid tiers are less cost-sensitive
        tier_multiplier = {
            "free": 1.0,
            "starter": 0.8,
            "standard": 0.6,
            "pro": 0.4,
        }.get(user_tier, 0.6)

        # Budget pressure
        budget_pressure = 0.0
        if budget_remaining is not None and budget_remaining > 0:
            budget_pressure = min(1.0, 1.0 / budget_remaining)

        # Normalize cost (assume $0.12 is "standard", $0.0 is free, $0.50 is expensive)
        normalized_cost = min(1.0, cost / 0.50)
        efficiency = 1.0 - normalized_cost

        return max(0.0, min(1.0, efficiency * tier_multiplier - budget_pressure * 0.2))

    def _latency_score(self, provider: ProviderCapabilities, profile: TaskProfile) -> float:
        """Return 0.0-1.0 latency score."""
        latency_ms = provider.average_latency_ms
        if latency_ms <= 5_000:
            base = 1.0
        elif latency_ms <= 15_000:
            base = 0.8
        elif latency_ms <= 30_000:
            base = 0.5
        else:
            base = 0.2

        # Latency-sensitive tasks get penalized for slow providers
        if profile.latency_sensitive and latency_ms > 15_000:
            base *= 0.5

        return base

    def _infer_profile_from_shot_spec(self, shot_spec: dict | None) -> TaskProfile:
        """Infer task profile from shot specification when task_type is not explicit."""
        if shot_spec is None:
            return self.TASK_PROFILES["half_body"]

        shot_type = shot_spec.get("shot_type", "half_body")
        return self.TASK_PROFILES.get(shot_type, self.TASK_PROFILES["half_body"])

    def get_provider_for_operation(self, operation: ImageOperation) -> ProviderCapabilities:
        """Backward-compatible provider selection."""
        from ..image_gateway import provider_for_operation
        return provider_for_operation(operation)
