"""Policy Engine — adaptive, context-aware decision layer over AgentRouter.

The PolicyEngine wraps the low-level rule-based AgentRouter with:
1. Budget-aware decision scoring (cost used vs max)
2. Shot-risk-aware thresholds (closeup vs full_body have different risk profiles)
3. Feedback-conditioned routing (past "not like me" for this style → more conservative)
4. Action probability scoring (each action gets a score, highest wins)

This is the bridge from static rules to adaptive policy-driven decisions.
"""

from __future__ import annotations

import math
from typing import Any

from .agent_router import AgentRouter, IDENTITY_REPAIR_THRESHOLD, IDENTITY_PASS_THRESHOLD, QUALITY_ACCEPT_THRESHOLD
from .failure_taxonomy import classify_failure


class PolicyEngine:
    """Adaptive policy-driven decision engine for candidate actions.

    Wraps AgentRouter with context-aware scoring that considers:
    - Budget remaining (cost used vs max)
    - Feedback history for this session/style
    - Shot risk level (closeup=low, full_body=high)
    - Action probability scoring (probabilistic decision, not deterministic)

    Usage:
        policy = PolicyEngine(agent_router, learning_layer)
        decision = policy.decide(
            judgement=judgement,
            budget=budget_dict,
            shot_spec=shot_spec,
            session_feedback=feedback_stats,
            edit_count=0,
            identity_repairs=0,
            identity_thresholds=thresholds,
        )
    """

    # Shot risk profiles — lower risk = higher confidence in single-shot success
    SHOT_RISK_PROFILES = {
        "closeup": {"risk": 0.2, "identity_weight": 0.95, "regeneration_penalty": 0.8},
        "half_body": {"risk": 0.4, "identity_weight": 0.85, "regeneration_penalty": 0.6},
        "full_body": {"risk": 0.7, "identity_weight": 0.70, "regeneration_penalty": 0.4},
        "environmental": {"risk": 0.8, "identity_weight": 0.60, "regeneration_penalty": 0.3},
        "default": {"risk": 0.5, "identity_weight": 0.80, "regeneration_penalty": 0.5},
    }

    # Action base scores — higher = more preferred when context is neutral
    ACTION_BASE_SCORES = {
        "ACCEPT": 1.0,
        "LOCAL_EDIT": 0.7,
        "IDENTITY_REPAIR": 0.5,
        "REGENERATE_FROM_ORIGINAL": 0.3,
        "REGENERATE_WITH_POSE_REFERENCE": 0.25,
        "DROP_CANDIDATE": 0.0,
        "REQUEST_BETTER_REFERENCE": 0.1,
    }

    # Feedback conditioning weights
    FEEDBACK_NOT_LIKE_ME_PENALTY = 0.3  # Reduce ACCEPT score if user previously disliked
    FEEDBACK_LIKE_ME_BONUS = 0.15  # Increase ACCEPT score if user previously liked
    FEEDBACK_HISTORY_WINDOW = 5  # Only consider last N feedback events

    def __init__(
        self,
        agent_router: AgentRouter | None = None,
        learning_layer=None,
    ) -> None:
        self._router = agent_router or AgentRouter()
        self._learning = learning_layer

    # ── Public API ────────────────────────────────────────────

    def decide(
        self,
        judgement: dict,
        budget: dict,
        shot_spec: dict | None = None,
        session_feedback: list[dict] | None = None,
        edit_count: int = 0,
        identity_repairs: int = 0,
        identity_thresholds: dict | None = None,
    ) -> dict:
        """Return a policy-driven decision with scoring context.

        The decision includes:
        - action: the chosen action string
        - reason: human-readable explanation
        - policy_scores: all action scores for transparency/debugging
        - confidence: 0-1 score for how confident the policy is
        """
        # 1. Get base rule-based decision from AgentRouter
        base_decision = self._router.decide_candidate_action(
            judgement,
            edit_count=edit_count,
            identity_repairs=identity_repairs,
            identity_thresholds=identity_thresholds,
        )
        base_action = base_decision["action"]
        from .evaluator import EvaluationService
        resolved_thresholds = self._router._resolve_thresholds(identity_thresholds)
        current_gate = EvaluationService._candidate_gate_status(
            judgement,
            resolved_thresholds,
        )
        diagnosis = classify_failure(
            judgement,
            gate_failures=current_gate.get("hard_gate_failures") or [],
        )

        # 1.5 Hard constraints override everything
        hard_override = self._hard_constraint_override(judgement, identity_thresholds)
        if hard_override is not None:
            return {
                "action": list(hard_override.keys())[0],
                "reason": "hard_constraint_override",
                "policy_scores": hard_override,
                "confidence": 1.0,
                "base_action": base_action,
                "budget_modifier": 1.0,
                "risk_profile": 0.0,
                "feedback_modifier": 0.0,
                **diagnosis,
            }

        # 2. Compute context modifiers
        budget_modifier = self._budget_modifier(budget)
        risk_profile = self._shot_risk_profile(shot_spec)
        # 3. Score all possible actions
        policy_scores = self._score_all_actions(
            judgement=judgement,
            base_action=base_action,
            budget_modifier=budget_modifier,
            risk_profile=risk_profile,
            session_feedback=session_feedback,
            edit_count=edit_count,
            identity_repairs=identity_repairs,
            identity_thresholds=identity_thresholds,
            shot_profile=self._shot_profile_name(shot_spec),
        )

        # 4. Select highest-scoring feasible action. Policy scoring may rank
        # valid actions, but it must never turn a failed hard gate into ACCEPT.
        feasible_actions = [
            action for action, item in policy_scores.items()
            if item.get("eligible")
        ]
        best_action = max(
            feasible_actions,
            key=lambda action: policy_scores[action]["score"],
        ) if feasible_actions else base_action
        best_score = policy_scores[best_action]["score"]
        feedback_modifier = self._feedback_modifier(session_feedback, best_action)

        # 5. Build enriched decision
        confidence = self._compute_confidence(
            best_score,
            {action: policy_scores[action] for action in feasible_actions},
        )
        reason = self._build_reason(
            base_action=base_action,
            final_action=best_action,
            budget_modifier=budget_modifier,
            risk_profile=risk_profile,
            feedback_modifier=feedback_modifier,
        )

        return {
            "action": best_action,
            "reason": reason,
            "policy_scores": {a: round(s["score"], 3) for a, s in policy_scores.items()},
            "policy_evidence": {
                action: item["learning_adjustment"]
                for action, item in policy_scores.items()
                if item.get("learning_adjustment", {}).get("active")
            },
            "eligible_actions": feasible_actions,
            "confidence": round(confidence, 3),
            "base_action": base_action,
            "budget_modifier": round(budget_modifier, 3),
            "risk_profile": risk_profile.get("risk", 0.5),
            "feedback_modifier": round(feedback_modifier, 3),
            **diagnosis,
        }

    def select_candidate(self, candidates: list[dict]) -> dict | None:
        """Delegate to AgentRouter for backward compatibility."""
        return self._router.select_candidate(candidates)

    def candidate_shortlist(self, candidates: list[dict], limit: int = 2) -> list[dict]:
        """Delegate to AgentRouter for backward compatibility."""
        return self._router.candidate_shortlist(candidates, limit=limit)

    def should_apply_identity_repair(
        self,
        judgement: dict,
        identity_thresholds: dict | None = None,
        shot_profile: str = "default",
    ) -> bool:
        """Delegate to AgentRouter for backward compatibility."""
        return self._router.should_apply_identity_repair(judgement, identity_thresholds)

    # ── Internal scoring ──────────────────────────────────────

    def _hard_constraint_override(self, judgement: dict, identity_thresholds: dict | None) -> dict | None:
        """Return hard override if any non-negotiable constraint is violated."""
        scores_data = judgement.get("scores", {}) or {}
        identity = scores_data.get("identity")
        failures = set(judgement.get("hard_failures") or [])
        diagnosis = classify_failure(judgement)

        # Safety is always a hard gate
        if "unsafe_content" in failures:
            return {"DROP_CANDIDATE": {"score": 1.0}}

        # No face detected → must regenerate
        if any(f in failures for f in ("no_face", "identity_no_generated_face", "no_face_detected")):
            return {"REGENERATE_FROM_ORIGINAL": {"score": 1.0}}

        if "identity_geometry_drift" in failures:
            return {"REGENERATE_WITH_POSE_REFERENCE": {"score": 1.0}}

        # A low-identity but otherwise excellent frame is precisely what the
        # deterministic identity writer is for. Regenerate only when quality,
        # composition, safety, or face detection makes that repair unsafe.
        thresholds = self._router._resolve_thresholds(identity_thresholds)
        identity_repair_threshold = float(
            thresholds.get("identity_repair_threshold", IDENTITY_REPAIR_THRESHOLD)
        )
        if (
            identity is not None
            and identity < identity_repair_threshold
            and not self._router.identity_repair_is_worthwhile(
                judgement, identity_thresholds
            )
        ):
            return {"REGENERATE_FROM_ORIGINAL": {"score": 1.0}}

        # Severe failures are routed by diagnosis instead of collapsing every
        # problem into the same blind regeneration.
        severe = {
            "face_distorted",
            "bad_artifacts",
            "unreadable_image",
            "bad_resolution",
            "too_blurry",
            "multiple_faces",
            "severe_artifacts",
            "wrong_composition",
            "anti_selfie_composition",
            "synthetic_appearance",
            "skin_over_smoothed",
        }
        if failures & severe:
            action = diagnosis["recovery_action"]
            if action == "IDENTITY_REPAIR":
                action = "REGENERATE_FROM_ORIGINAL"
            return {action: {"score": 1.0}}

        return None

    def _resolve_thresholds(self, identity_thresholds: dict | None) -> dict:
        """Delegate to router's threshold resolution."""
        return self._router._resolve_thresholds(identity_thresholds)

    def _score_all_actions(
        self,
        judgement: dict,
        base_action: str,
        budget_modifier: float,
        risk_profile: dict,
        session_feedback: list[dict] | None,
        edit_count: int,
        identity_repairs: int,
        identity_thresholds: dict | None = None,
        shot_profile: str = "default",
    ) -> dict[str, dict]:
        """Score every possible action under current context."""
        scores = {}
        risk = risk_profile.get("risk", 0.5)
        identity_weight = risk_profile.get("identity_weight", 0.8)
        regen_penalty = risk_profile.get("regeneration_penalty", 0.5)

        scores_data = judgement.get("scores", {}) or {}
        identity = scores_data.get("identity")
        style_match = scores_data.get("style_match")
        artifact = scores_data.get("artifact")
        failures = set(judgement.get("hard_failures") or [])
        thresholds = self._router._resolve_thresholds(identity_thresholds)
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )
        identity_repair_threshold = float(
            thresholds.get("identity_repair_threshold", IDENTITY_REPAIR_THRESHOLD)
        )
        from .evaluator import EvaluationService
        gate = EvaluationService._candidate_gate_status(judgement, thresholds)
        diagnosis = classify_failure(
            judgement,
            gate_failures=gate.get("hard_gate_failures") or [],
        )

        # Hard constraints already handled in decide() via _hard_constraint_override
        for action in AgentRouter.ALLOWED_ACTIONS:
            eligible = self._action_is_eligible(
                action=action,
                base_action=base_action,
                gate=gate,
                edit_count=edit_count,
                identity_repairs=identity_repairs,
            )
            base = self.ACTION_BASE_SCORES.get(action, 0.0)
            score = base

            # Budget pressure: penalize expensive actions when budget is tight
            if action in ("REGENERATE_FROM_ORIGINAL", "REGENERATE_WITH_POSE_REFERENCE"):
                score *= budget_modifier * regen_penalty
            elif action == "IDENTITY_REPAIR":
                score *= budget_modifier * 0.9
            elif action == "LOCAL_EDIT":
                score *= budget_modifier * 0.95
            elif action == "ACCEPT":
                # ACCEPT gets boosted when budget is tight (cheap)
                score *= (1.0 + (1.0 - budget_modifier) * 0.3)

            # Edit/repair exhaustion penalties
            if action == "LOCAL_EDIT" and edit_count >= 2:
                score *= 0.1
            if action == "IDENTITY_REPAIR" and identity_repairs >= 1:
                score *= 0.1

            # Local artifact detection: boost LOCAL_EDIT when artifact score is low
            if (
                action == "LOCAL_EDIT"
                and artifact is not None
                and artifact < QUALITY_ACCEPT_THRESHOLD
                and edit_count < 2
            ):
                score *= 5.0  # Very strong boost — must exceed ACCEPT

            # Risk profile adjustments
            if action == "ACCEPT":
                # High-risk shots need higher identity confidence to accept
                if identity is not None:
                    identity_boost = (identity / 10.0) * identity_weight
                    score *= (0.5 + identity_boost)
                    # Strong penalty if identity is below pass threshold
                    if identity < identity_pass_threshold:
                        score *= 0.1  # Gray zone identity → don't accept, repair instead
                else:
                    score *= 0.3  # Unverified identity → very risky to accept
                # Penalize ACCEPT if artifact is below threshold (should edit instead)
                if artifact is not None and artifact < QUALITY_ACCEPT_THRESHOLD:
                    score *= 0.2

            if action == "IDENTITY_REPAIR":
                # Favor one local identity write whenever the frame is
                # repairable and identity has not passed yet.
                if (
                    identity is not None
                    and identity < identity_pass_threshold
                    and self._router.identity_repair_is_worthwhile(
                        judgement, identity_thresholds
                    )
                ):
                    score *= 3.0

            if action == "REGENERATE_FROM_ORIGINAL":
                # High-risk shots benefit more from regeneration
                score *= (1.0 + risk * 0.5)

            if action == "REGENERATE_WITH_POSE_REFERENCE":
                if "identity_geometry_drift" in failures:
                    score *= 5.0
                score *= (1.0 + risk * 0.5)

            # Feedback conditioning is action-specific: dislikes reduce ACCEPT
            # while increasing identity repair/regeneration preference.
            action_feedback_modifier = self._feedback_modifier(
                session_feedback,
                action,
            )
            score *= (1.0 + action_feedback_modifier)

            # Base action bias: slight preference for the rule-based recommendation
            if action == base_action:
                score *= 1.1

            learning_adjustment = self._strategy_adjustment(
                failure_class=diagnosis["failure_class"],
                action=action,
                shot_profile=shot_profile,
            )
            score *= float(learning_adjustment.get("multiplier") or 1.0)

            scores[action] = {
                "score": max(0.0, min(1.0, score)),
                "eligible": eligible,
                "learning_adjustment": learning_adjustment,
            }

        return scores

    def _strategy_adjustment(
        self,
        *,
        failure_class: str,
        action: str,
        shot_profile: str,
    ) -> dict:
        method = getattr(self._learning, "strategy_adjustment", None)
        if not callable(method):
            return {"active": False, "multiplier": 1.0}
        try:
            return method(
                failure_class=failure_class,
                action=action,
                shot_profile=shot_profile,
            )
        except Exception:
            # Learning evidence must never make generation unavailable. A
            # malformed or locked analytics store degrades to the static policy.
            return {
                "active": False,
                "multiplier": 1.0,
                "reason": "strategy_evidence_unavailable",
            }

    @staticmethod
    def _action_is_eligible(
        *,
        action: str,
        base_action: str,
        gate: dict,
        edit_count: int,
        identity_repairs: int,
    ) -> bool:
        """Keep policy ranking inside the bounded state-machine action set."""
        if action == "ACCEPT":
            return bool(gate.get("hard_gates_pass"))
        if action == "LOCAL_EDIT":
            return base_action == "LOCAL_EDIT" and edit_count < 2
        if action == "IDENTITY_REPAIR":
            return base_action == "IDENTITY_REPAIR" and identity_repairs < 1
        if action == "REGENERATE_FROM_ORIGINAL":
            return base_action in {
                "REGENERATE_FROM_ORIGINAL",
                "DROP_CANDIDATE",
            }
        if action == "REGENERATE_WITH_POSE_REFERENCE":
            return base_action == "REGENERATE_WITH_POSE_REFERENCE"
        if action == "REQUEST_BETTER_REFERENCE":
            return base_action == "REQUEST_BETTER_REFERENCE"
        if action == "DROP_CANDIDATE":
            return True
        return False

    def _budget_modifier(self, budget: dict) -> float:
        """Return 0.0-1.0 modifier based on budget remaining.

        1.0 = plenty of budget, 0.0 = exhausted.
        """
        max_cost = budget.get("max_total_api_cost", 1.0)
        used = budget.get("estimated_cost_used", 0.0)
        if max_cost <= 0:
            return 0.0
        ratio = used / max_cost
        # Exponential decay: tight budget strongly penalizes expensive actions
        return max(0.0, math.exp(-2.0 * ratio))

    def _shot_risk_profile(self, shot_spec: dict | None) -> dict:
        """Return risk profile for the given shot type."""
        if shot_spec is None:
            return self.SHOT_RISK_PROFILES["default"]
        shot_type = shot_spec.get("shot_type", "default")
        return self.SHOT_RISK_PROFILES.get(shot_type, self.SHOT_RISK_PROFILES["default"])

    @staticmethod
    def _shot_profile_name(shot_spec: dict | None) -> str:
        if not shot_spec:
            return "default"
        return str(
            shot_spec.get("shot_type")
            or shot_spec.get("shot_id")
            or "default"
        )

    def _feedback_modifier(self, session_feedback: list[dict] | None, proposed_action: str) -> float:
        """Return -1.0 to +1.0 modifier based on recent user feedback.

        If user previously said "not like me" for similar images,
        be more conservative (lower ACCEPT scores, boost regeneration).
        """
        if not session_feedback:
            return 0.0

        # Only consider recent feedback
        recent = session_feedback[-self.FEEDBACK_HISTORY_WINDOW:]
        not_like_count = sum(1 for f in recent if f.get("event") == "not_like_me")
        like_count = sum(1 for f in recent if f.get("event") == "looks_like_me")

        modifier = 0.0
        if proposed_action == "ACCEPT":
            if not_like_count > 0:
                modifier -= self.FEEDBACK_NOT_LIKE_ME_PENALTY * (not_like_count / len(recent))
            if like_count > 0:
                modifier += self.FEEDBACK_LIKE_ME_BONUS * (like_count / len(recent))
        elif proposed_action in ("REGENERATE_FROM_ORIGINAL", "IDENTITY_REPAIR"):
            if not_like_count > 0:
                modifier += self.FEEDBACK_NOT_LIKE_ME_PENALTY * 0.5 * (not_like_count / len(recent))

        return max(-1.0, min(1.0, modifier))

    def _compute_confidence(self, best_score: float, all_scores: dict) -> float:
        """Compute confidence as the margin between best and second-best."""
        if len(all_scores) <= 1:
            return 1.0
        sorted_scores = sorted([s["score"] for s in all_scores.values()], reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        # Normalize: margin > 0.3 → high confidence, margin < 0.05 → low
        return min(1.0, margin * 3.0 + 0.1)

    def _build_reason(
        self,
        base_action: str,
        final_action: str,
        budget_modifier: float,
        risk_profile: dict,
        feedback_modifier: float,
    ) -> str:
        """Build human-readable reason for the policy decision."""
        parts = [f"base={base_action}"]
        if budget_modifier < 0.5:
            parts.append(f"budget_tight({budget_modifier:.2f})")
        risk = risk_profile.get("risk", 0.5)
        if risk > 0.5:
            parts.append(f"high_risk_shot({risk:.2f})")
        elif risk < 0.3:
            parts.append(f"low_risk_shot({risk:.2f})")
        if abs(feedback_modifier) > 0.1:
            direction = "conservative" if feedback_modifier < 0 else "lenient"
            parts.append(f"feedback_{direction}({feedback_modifier:+.2f})")
        if final_action != base_action:
            parts.append(f"policy_overrode→{final_action}")
        else:
            parts.append(f"policy_confirmed")
        return "; ".join(parts)


def select_best_variant(variants: list[dict] | None) -> dict | None:
    """Pick the most real repair-stage variant that passes the hard gates.

    The delivery candidate is chosen across ALL recorded pipeline stages
    (scaffold, face swap, identity blend, local edit, identity repair,
    sharpening) instead of defaulting to the last repair output:

    1. Pool = variants whose ``gate_status.hard_gates_pass`` is True and that
       carry a valid judgement (stages whose judge failed are skipped).
    2. Sort by VLM realism score first (a missing score counts as -1), then
       by ``aggregate_score``. Full ties keep the LATER stage, so an
       equal-scoring pipeline keeps its most processed output.
    3. Return None when the pool is empty. The caller then keeps the
       last-stage output and lets the final delivery gate reject it honestly
       instead of fabricating a pass.
    """
    pool = []
    for position, variant in enumerate(variants or []):
        judgement = variant.get("judgement")
        if not isinstance(judgement, dict):
            continue
        gate = variant.get("gate_status") or {}
        if not gate.get("hard_gates_pass"):
            continue
        realism = (judgement.get("scores") or {}).get("realism")
        realism_key = (
            float(realism) if isinstance(realism, (int, float)) else -1.0
        )
        aggregate = variant.get("aggregate_score")
        aggregate_key = (
            float(aggregate) if isinstance(aggregate, (int, float)) else -1.0
        )
        pool.append((realism_key, aggregate_key, position, variant))
    if not pool:
        return None
    pool.sort(key=lambda item: (item[0], item[1], item[2]))
    return pool[-1][3]
