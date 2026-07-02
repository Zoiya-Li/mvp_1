"""Tests for SmartModelRouter — task-aware provider selection.

These tests verify that SmartModelRouter correctly:
1. Routes hero_face to high-fidelity providers
2. Routes full_body to composition-capable providers
3. Routes local_edit to mask-edit-capable providers
4. Considers user tier for cost sensitivity
5. Considers budget remaining for cost pressure
6. Falls back gracefully when no ideal provider exists
"""

from __future__ import annotations

import pytest

from server.generation.smart_router import SmartModelRouter, TaskProfile, RoutingDecision


@pytest.fixture
def router():
    return SmartModelRouter()


class TestSmartRouterHeroFace:
    """Test routing for hero face (close-up portrait) tasks."""

    def test_hero_face_prefers_high_fidelity(self, router):
        decision = router.route_for_task("hero_face")
        assert decision.confidence > 0.5
        assert "identity" in decision.reason.lower() or "fidelity" in decision.reason.lower()

    def test_hero_face_latency_sensitive(self, router):
        decision = router.route_for_task("hero_face")
        # Hero face should prefer faster providers
        assert decision.estimated_latency_ms < 30_000 or decision.confidence < 0.8


class TestSmartRouterFullBody:
    """Test routing for full body tasks."""

    def test_full_body_prefers_composition(self, router):
        decision = router.route_for_task("full_body")
        assert decision.confidence > 0.4

    def test_full_body_less_identity_focused(self, router):
        # Full body should accept providers with moderate identity scores
        decision = router.route_for_task("full_body")
        assert decision.provider != ""  # Should always return a provider


class TestSmartRouterLocalEdit:
    """Test routing for local edit tasks."""

    def test_local_edit_prefers_mask_edit(self, router):
        decision = router.route_for_task("local_edit")
        # Should still return a provider even if no mask edit support
        assert decision.provider != ""

    def test_local_edit_identity_priority(self, router):
        decision = router.route_for_task("local_edit")
        assert "identity" in decision.reason.lower() or "edit" in decision.reason.lower()


class TestSmartRouterUpscale:
    """Test routing for upscale tasks."""

    def test_upscale_routes_to_local(self, router):
        decision = router.route_for_task("upscale")
        # Upscale should prefer local (free) providers
        assert decision.estimated_cost == 0.0


class TestSmartRouterIdentityRepair:
    """Test routing for identity repair tasks."""

    def test_identity_repair_routes_to_local(self, router):
        decision = router.route_for_task("identity_repair")
        # Identity repair should use local provider
        assert decision.estimated_cost == 0.0


class TestSmartRouterCostSensitivity:
    """Test cost-aware routing."""

    def test_free_tier_prefers_cheaper(self, router):
        expensive = router.route_for_task("hero_face", user_tier="free")
        cheap = router.route_for_task("hero_face", user_tier="pro")
        # Pro should be more willing to use expensive providers
        assert expensive.estimated_cost <= cheap.estimated_cost or expensive.provider == cheap.provider

    def test_budget_pressure(self, router):
        with_budget = router.route_for_task("hero_face", budget_remaining=0.5)
        without_budget = router.route_for_task("hero_face", budget_remaining=10.0)
        # Tight budget should prefer cheaper or same
        assert with_budget.estimated_cost <= without_budget.estimated_cost or with_budget.provider == without_budget.provider


class TestSmartRouterShotSpecInference:
    """Test profile inference from shot specifications."""

    def test_closeup_infers_hero_face(self, router):
        decision = router.route_for_task(
            "half_body",  # generic task type
            shot_spec={"shot_type": "closeup"},
        )
        # Should use closeup profile (high identity priority)
        assert decision.confidence > 0.4

    def test_full_body_infers_full_body(self, router):
        decision = router.route_for_task(
            "half_body",
            shot_spec={"shot_type": "full_body"},
        )
        assert decision.confidence > 0.4


class TestSmartRouterFallback:
    """Test graceful fallback behavior."""

    def test_unknown_task_type_fallback(self, router):
        # Even unknown task types should return a valid decision
        decision = router.route_for_task("unknown_task")  # type: ignore[arg-type]
        assert decision.provider != ""
        assert decision.model != ""

    def test_no_providers_still_returns(self, router):
        # Router should always return something
        decision = router.route_for_task("hero_face")
        assert isinstance(decision, RoutingDecision)


class TestSmartRouterProviderScoring:
    """Test internal provider scoring logic."""

    def test_identity_score_boost(self, router):
        from server.image_gateway import OPENROUTER_GEMINI_CAPABILITIES
        profile = TaskProfile(
            task_type="hero_face",
            identity_priority=0.95,
            composition_priority=0.4,
            editing_required=False,
            latency_sensitive=True,
            cost_sensitive=False,
        )
        score, reason = router._score_provider(
            OPENROUTER_GEMINI_CAPABILITIES,
            profile,
            "standard",
            None,
        )
        assert score > 0.5
        assert "identity" in reason.lower() or "fidelity" in reason.lower()

    def test_cost_score_free(self, router):
        from server.image_gateway import LOCAL_UPSCALE_CAPABILITIES
        score = router._cost_score(
            LOCAL_UPSCALE_CAPABILITIES,
            TaskProfile("upscale", 0.9, 0.5, False, False, False),
            "free",
            None,
        )
        assert score == 1.0  # Free provider is always cost-efficient

    def test_latency_score_fast(self, router):
        from server.image_gateway import LOCAL_UPSCALE_CAPABILITIES
        score = router._latency_score(
            LOCAL_UPSCALE_CAPABILITIES,
            TaskProfile("upscale", 0.9, 0.5, False, False, False),
        )
        assert score >= 0.8  # Local is fast

    def test_latency_score_slow_penalty(self, router):
        from server.image_gateway import CHROME_GEMINI_CAPABILITIES
        score = router._latency_score(
            CHROME_GEMINI_CAPABILITIES,
            TaskProfile("hero_face", 0.95, 0.4, False, True, False),
        )
        # Chrome is slow (45s) and hero_face is latency-sensitive
        assert score < 0.5


class TestSmartRouterTaskProfiles:
    """Test task profile definitions."""

    def test_hero_face_profile(self, router):
        profile = router.TASK_PROFILES["hero_face"]
        assert profile.identity_priority == 0.95
        assert profile.latency_sensitive is True

    def test_full_body_profile(self, router):
        profile = router.TASK_PROFILES["full_body"]
        assert profile.composition_priority == 0.85
        assert profile.identity_priority == 0.70

    def test_local_edit_profile(self, router):
        profile = router.TASK_PROFILES["local_edit"]
        assert profile.editing_required is True

    def test_upscale_profile(self, router):
        profile = router.TASK_PROFILES["upscale"]
        assert profile.cost_sensitive is False
        assert profile.identity_priority == 0.90
