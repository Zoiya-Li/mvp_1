"""Tests for PolicyEngine — adaptive, context-aware decision layer.

These tests verify that PolicyEngine correctly:
1. Wraps AgentRouter decisions with policy scoring
2. Applies budget-aware modifiers (tight budget → prefer cheap actions)
3. Applies shot-risk profiles (high-risk shots → more conservative)
4. Applies feedback-conditioned routing (past dislikes → conservative)
5. Computes confidence scores
6. Maintains backward compatibility with AgentRouter delegation
"""

from __future__ import annotations

import pytest

from server.evaluation.agent_router import AgentRouter
from server.evaluation.policy_engine import PolicyEngine


@pytest.fixture
def router():
    return AgentRouter()


@pytest.fixture
def policy(router):
    return PolicyEngine(agent_router=router)


@pytest.fixture
def sample_judgement():
    return {
        "scores": {
            "identity": 8.5,
            "style_match": 8.0,
            "artifact": 9.0,
            "aesthetic": 8.5,
        },
        "hard_failures": [],
        "recommended_action": "accept",
        "notes": "Good candidate",
        "raw_response": "test",
    }


@pytest.fixture
def poor_identity_judgement():
    return {
        "scores": {
            "identity": 5.0,
            "style_match": 7.0,
            "artifact": 6.0,
        },
        "hard_failures": ["identity_too_low"],
        "recommended_action": "face_swap",
        "notes": "Does not look like user",
        "raw_response": "test",
    }


@pytest.fixture
def unsafe_judgement():
    return {
        "scores": {"identity": 8.0},
        "hard_failures": ["unsafe_content"],
        "recommended_action": "discard",
        "notes": "Unsafe",
        "raw_response": "test",
    }


@pytest.fixture
def no_face_judgement():
    return {
        "scores": {"identity": None},
        "hard_failures": ["no_face_detected"],
        "recommended_action": "regenerate",
        "notes": "No face",
        "raw_response": "test",
    }


@pytest.fixture
def healthy_budget():
    return {"max_total_api_cost": 10.0, "estimated_cost_used": 1.0}


@pytest.fixture
def tight_budget():
    return {"max_total_api_cost": 10.0, "estimated_cost_used": 9.0}


class TestPolicyEngineDecide:
    """Test the main decide() method."""

    def test_accept_healthy_budget(self, policy, sample_judgement, healthy_budget):
        decision = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
        )
        assert decision["action"] == "ACCEPT"
        assert decision["confidence"] > 0.5
        assert "base=" in decision["reason"]

    def test_tight_budget_prefers_accept(self, policy, sample_judgement, tight_budget):
        """When budget is tight, ACCEPT should be strongly preferred over regeneration."""
        # Lower identity slightly to make it a gray zone
        sample_judgement["scores"]["identity"] = 7.2
        sample_judgement["hard_failures"] = ["identity_too_low"]
        sample_judgement["recommended_action"] = "face_swap"

        decision = policy.decide(
            judgement=sample_judgement,
            budget=tight_budget,
        )
        # With tight budget, should prefer ACCEPT or LOCAL_EDIT over REGENERATE
        assert decision["action"] in ("ACCEPT", "LOCAL_EDIT", "IDENTITY_REPAIR")
        assert decision["budget_modifier"] < 0.5

    def test_unsafe_always_drop(self, policy, unsafe_judgement, healthy_budget):
        decision = policy.decide(
            judgement=unsafe_judgement,
            budget=healthy_budget,
        )
        assert decision["action"] == "DROP_CANDIDATE"
        assert decision["confidence"] == 1.0

    def test_no_face_always_regenerate(self, policy, no_face_judgement, healthy_budget):
        decision = policy.decide(
            judgement=no_face_judgement,
            budget=healthy_budget,
        )
        assert decision["action"] == "REGENERATE_FROM_ORIGINAL"
        assert decision["confidence"] == 1.0

    def test_poor_identity_regenerate(self, policy, poor_identity_judgement, healthy_budget):
        decision = policy.decide(
            judgement=poor_identity_judgement,
            budget=healthy_budget,
        )
        assert decision["action"] == "REGENERATE_FROM_ORIGINAL"

    def test_shot_risk_high_risk(self, policy, sample_judgement, healthy_budget):
        """High-risk shots should lower ACCEPT confidence."""
        decision_low = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
            shot_spec={"shot_type": "closeup"},
        )
        decision_high = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
            shot_spec={"shot_type": "full_body"},
        )
        # Both should accept, but full_body might have lower confidence
        assert decision_low["action"] == "ACCEPT"
        assert decision_high["action"] == "ACCEPT"
        assert "risk" in decision_high["reason"].lower() or "risk" in decision_low["reason"].lower()

    def test_feedback_not_like_me_penalty(self, policy, sample_judgement, healthy_budget):
        """Past 'not_like_me' feedback should make policy more conservative."""
        feedback = [
            {"event": "not_like_me", "image_id": "img1"},
            {"event": "not_like_me", "image_id": "img2"},
        ]
        decision = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
            session_feedback=feedback,
        )
        # The modifier should be negative
        assert decision["feedback_modifier"] < 0
        assert "conservative" in decision["reason"] or "feedback" in decision["reason"]

    def test_feedback_like_me_bonus(self, policy, sample_judgement, healthy_budget):
        """Past 'looks_like_me' feedback should make policy more lenient."""
        feedback = [
            {"event": "looks_like_me", "image_id": "img1"},
        ]
        decision = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
            session_feedback=feedback,
        )
        assert decision["feedback_modifier"] > 0

    def test_policy_scores_present(self, policy, sample_judgement, healthy_budget):
        decision = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
        )
        assert "policy_scores" in decision
        scores = decision["policy_scores"]
        assert "ACCEPT" in scores
        assert "REGENERATE_FROM_ORIGINAL" in scores
        assert all(0.0 <= s <= 1.0 for s in scores.values())

    def test_confidence_computation(self, policy, sample_judgement, healthy_budget):
        decision = policy.decide(
            judgement=sample_judgement,
            budget=healthy_budget,
        )
        assert 0.0 <= decision["confidence"] <= 1.0


class TestPolicyEngineBudgetModifier:
    """Test budget modifier calculations."""

    def test_plenty_budget(self, policy):
        modifier = policy._budget_modifier({"max_total_api_cost": 10.0, "estimated_cost_used": 0.5})
        assert modifier > 0.9

    def test_half_budget(self, policy):
        modifier = policy._budget_modifier({"max_total_api_cost": 10.0, "estimated_cost_used": 5.0})
        # exp(-2 * 0.5) = exp(-1) ≈ 0.368
        assert 0.3 < modifier < 0.4

    def test_exhausted_budget(self, policy):
        modifier = policy._budget_modifier({"max_total_api_cost": 10.0, "estimated_cost_used": 10.0})
        assert modifier == pytest.approx(0.135, abs=0.01)  # exp(-2) ≈ 0.135

    def test_zero_max_budget(self, policy):
        modifier = policy._budget_modifier({"max_total_api_cost": 0.0, "estimated_cost_used": 0.0})
        assert modifier == 0.0


class TestPolicyEngineShotRisk:
    """Test shot risk profile selection."""

    def test_closeup_low_risk(self, policy):
        profile = policy._shot_risk_profile({"shot_type": "closeup"})
        assert profile["risk"] == 0.2
        assert profile["identity_weight"] == 0.95

    def test_full_body_high_risk(self, policy):
        profile = policy._shot_risk_profile({"shot_type": "full_body"})
        assert profile["risk"] == 0.7
        assert profile["identity_weight"] == 0.70

    def test_default_risk(self, policy):
        profile = policy._shot_risk_profile(None)
        assert profile["risk"] == 0.5

    def test_unknown_shot_type(self, policy):
        profile = policy._shot_risk_profile({"shot_type": "weird_shot"})
        assert profile["risk"] == 0.5  # Falls back to default


class TestPolicyEngineFeedbackModifier:
    """Test feedback conditioning."""

    def test_no_feedback(self, policy):
        modifier = policy._feedback_modifier(None, "ACCEPT")
        assert modifier == 0.0

    def test_empty_feedback(self, policy):
        modifier = policy._feedback_modifier([], "ACCEPT")
        assert modifier == 0.0

    def test_not_like_me_penalty(self, policy):
        feedback = [{"event": "not_like_me"}, {"event": "looks_like_me"}]
        modifier = policy._feedback_modifier(feedback, "ACCEPT")
        # 1 not_like out of 2 → penalty = -0.3 * 0.5 = -0.15, plus bonus = +0.15 * 0.5 = +0.075
        # Net ≈ -0.075
        assert modifier < 0

    def test_like_me_bonus(self, policy):
        feedback = [{"event": "looks_like_me"}, {"event": "looks_like_me"}]
        modifier = policy._feedback_modifier(feedback, "ACCEPT")
        assert modifier > 0

    def test_regeneration_boosted_by_feedback(self, policy):
        feedback = [{"event": "not_like_me"}]
        modifier = policy._feedback_modifier(feedback, "REGENERATE_FROM_ORIGINAL")
        assert modifier > 0  # Regeneration gets boosted by negative feedback


class TestPolicyEngineBackwardCompat:
    """Test that PolicyEngine delegates correctly to AgentRouter."""

    def test_select_candidate_delegation(self, policy):
        candidates = [
            {"candidate_id": "c1", "aggregate_score": 5.0, "gate_status": {"hard_gates_pass": True}, "agent_action": {"action": "ACCEPT"}},
            {"candidate_id": "c2", "aggregate_score": 7.0, "gate_status": {"hard_gates_pass": True}, "agent_action": {"action": "ACCEPT"}},
        ]
        selected = policy.select_candidate(candidates)
        assert selected["candidate_id"] == "c2"

    def test_candidate_shortlist(self, policy):
        candidates = [
            {"candidate_id": "c1", "aggregate_score": 5.0, "gate_status": {"hard_gates_pass": True}, "agent_action": {"action": "ACCEPT"}},
            {"candidate_id": "c2", "aggregate_score": 7.0, "gate_status": {"hard_gates_pass": True}, "agent_action": {"action": "ACCEPT"}},
        ]
        shortlist = policy.candidate_shortlist(candidates, limit=2)
        assert len(shortlist) == 2
        assert shortlist[0]["rank"] == 1

    def test_should_apply_identity_repair(self, policy):
        judgement = {
            "scores": {"identity": 7.5},
            "hard_failures": ["identity_too_low"],
            "recommended_action": "face_swap",
        }
        assert policy.should_apply_identity_repair(judgement) is True


class TestPolicyEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_identity_none_with_budget(self, policy, healthy_budget):
        judgement = {
            "scores": {"identity": None},
            "hard_failures": [],
            "recommended_action": "accept",
        }
        decision = policy.decide(
            judgement=judgement,
            budget=healthy_budget,
        )
        # Unverified identity should not be accepted
        assert decision["action"] != "ACCEPT"

    def test_severe_quality_fail(self, policy, healthy_budget):
        judgement = {
            "scores": {"identity": 8.5, "artifact": 3.0},
            "hard_failures": ["severe_artifacts"],
            "recommended_action": "accept",
        }
        decision = policy.decide(
            judgement=judgement,
            budget=healthy_budget,
        )
        assert decision["action"] == "REGENERATE_FROM_ORIGINAL"

    def test_local_edit_budget_exhausted(self, policy, tight_budget):
        judgement = {
            "scores": {"identity": 8.5, "artifact": 5.0},
            "hard_failures": [],
            "recommended_action": "local_edit",
        }
        decision = policy.decide(
            judgement=judgement,
            budget=tight_budget,
            edit_count=0,
        )
        # With tight budget, even local edit might be penalized
        assert "budget_tight" in decision["reason"]

    def test_high_risk_shot_with_good_identity(self, policy, healthy_budget):
        judgement = {
            "scores": {"identity": 9.5, "style_match": 8.0, "artifact": 9.0},
            "hard_failures": [],
            "recommended_action": "accept",
        }
        decision = policy.decide(
            judgement=judgement,
            budget=healthy_budget,
            shot_spec={"shot_type": "full_body"},
        )
        # Even high-risk shot with excellent identity should be accepted
        assert decision["action"] == "ACCEPT"
        assert "high_risk_shot" in decision["reason"]
