"""Episode-level recovery planning for the portrait Agent.

Candidate routing answers "what is wrong now?".  This planner answers the
second, equally important question: "what have we already tried for this kind
of failure?"  It keeps a failed episode from spending its bounded budget on
the same action and route repeatedly.
"""

from __future__ import annotations

from typing import Any


RECOVERY_PLAN_VERSION = "episode_recovery_v1"


# Each step changes at least one meaningful capability: repair mechanism,
# reference geometry, or generation route.  Alternate-route steps are skipped
# when no benchmark-approved recovery model is configured.
RECOVERY_LADDERS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "identity_similarity": (
        ("IDENTITY_REPAIR", "primary", "identity_writeback"),
        ("REGENERATE_WITH_POSE_REFERENCE", "primary", "pose_anchored_identity_reset"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_identity_reset"),
        ("REQUEST_BETTER_REFERENCE", "none", "reference_evidence_exhausted"),
    ),
    "identity_geometry": (
        ("REGENERATE_WITH_POSE_REFERENCE", "primary", "pose_matched_regeneration"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_geometry_reset"),
        ("REQUEST_BETTER_REFERENCE", "none", "pose_evidence_exhausted"),
    ),
    "synthetic_texture": (
        ("REGENERATE_FROM_ORIGINAL", "primary", "photoreal_regeneration"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_texture_reset"),
        ("DROP_CANDIDATE", "none", "texture_routes_exhausted"),
    ),
    "composition": (
        ("REGENERATE_FROM_ORIGINAL", "primary", "composition_regeneration"),
        ("REGENERATE_WITH_POSE_REFERENCE", "primary", "pose_anchored_composition"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_composition_reset"),
        ("DROP_CANDIDATE", "none", "composition_routes_exhausted"),
    ),
    "face_detection": (
        ("REGENERATE_FROM_ORIGINAL", "primary", "rebuild_face_and_composition"),
        ("REGENERATE_WITH_POSE_REFERENCE", "primary", "pose_anchored_face_rebuild"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_face_rebuild"),
        ("REQUEST_BETTER_REFERENCE", "none", "face_evidence_exhausted"),
    ),
    "image_integrity": (
        ("REGENERATE_FROM_ORIGINAL", "primary", "clean_source_regeneration"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_clean_regeneration"),
        ("DROP_CANDIDATE", "none", "integrity_routes_exhausted"),
    ),
    "local_artifact": (
        ("LOCAL_EDIT", "primary", "localized_artifact_repair"),
        ("REGENERATE_FROM_ORIGINAL", "primary", "clean_source_regeneration"),
        ("REGENERATE_FROM_ORIGINAL", "alternate", "alternate_model_clean_regeneration"),
        ("DROP_CANDIDATE", "none", "artifact_routes_exhausted"),
    ),
}


def _strategy_key(action: str, route_mode: str, strategy: str) -> str:
    return f"{action}:{route_mode}:{strategy}"


class EpisodeRecoveryPlanner:
    """Choose the next novel recovery step for one bounded pipeline episode."""

    def plan(
        self,
        decision: dict[str, Any],
        action_history: list[dict[str, Any]] | None,
        *,
        alternate_route_available: bool = False,
    ) -> dict[str, Any]:
        failure_class = str(decision.get("failure_class") or "unknown_quality")
        history = [
            item
            for item in (action_history or [])
            if item.get("executed") is True
            and item.get("failure_class") == failure_class
        ]
        attempted = {
            _strategy_key(
                str(item.get("action") or ""),
                str(item.get("route_mode") or "primary"),
                str(item.get("recovery_strategy") or item.get("strategy_variant") or ""),
            )
            for item in history
        }
        generation_already_reset = any(
            str(item.get("action") or "").startswith("REGENERATE_")
            for item in history
        )

        ladder = RECOVERY_LADDERS.get(failure_class)
        if not ladder:
            return self._decorate(
                decision,
                step=0,
                failure_streak=len(history),
                route_mode=str(decision.get("route_mode") or "primary"),
                strategy=str(decision.get("recovery_strategy") or "quality_regeneration"),
                attempted=attempted,
                reason="no_specialized_ladder",
            )

        base_action = str(decision.get("action") or "")
        if not history:
            for step, (action, route_mode, strategy) in enumerate(ladder):
                if action == base_action and self._route_is_available(
                    route_mode, alternate_route_available
                ):
                    return self._decorate(
                        decision,
                        step=step,
                        failure_streak=0,
                        route_mode=route_mode,
                        strategy=strategy,
                        attempted=attempted,
                        reason="base_action_first_attempt",
                    )
            return self._decorate(
                decision,
                step=0,
                failure_streak=0,
                route_mode=str(decision.get("route_mode") or "primary"),
                strategy=str(decision.get("recovery_strategy") or "policy_override"),
                attempted=attempted,
                reason="base_action_outside_specialized_ladder",
            )

        for step, (action, route_mode, strategy) in enumerate(ladder):
            if generation_already_reset and action in {"LOCAL_EDIT", "IDENTITY_REPAIR"}:
                # Do not move backward to a weaker local mutation after policy
                # already concluded that the whole generation needed a reset.
                continue
            if not self._route_is_available(route_mode, alternate_route_available):
                continue
            key = _strategy_key(action, route_mode, strategy)
            if key in attempted:
                continue
            return self._decorate(
                {**decision, "action": action},
                step=step,
                failure_streak=len(history),
                route_mode=route_mode,
                strategy=strategy,
                attempted=attempted,
                reason="novel_strategy_after_repeated_failure",
            )

        # The finite ladder is deliberately terminal.  Spending another call on
        # an already-failed strategy creates latency and cost, not intelligence.
        return self._decorate(
            {**decision, "action": "DROP_CANDIDATE"},
            step=len(ladder),
            failure_streak=len(history),
            route_mode="none",
            strategy="recovery_ladder_exhausted",
            attempted=attempted,
            reason="all_available_strategies_exhausted",
        )

    @staticmethod
    def _route_is_available(route_mode: str, alternate_route_available: bool) -> bool:
        return route_mode != "alternate" or alternate_route_available

    @staticmethod
    def _decorate(
        decision: dict[str, Any],
        *,
        step: int,
        failure_streak: int,
        route_mode: str,
        strategy: str,
        attempted: set[str],
        reason: str,
    ) -> dict[str, Any]:
        return {
            **decision,
            "route_mode": route_mode,
            "recovery_strategy": strategy,
            "recovery_plan": {
                "version": RECOVERY_PLAN_VERSION,
                "step": step,
                "failure_streak": failure_streak,
                "attempted_strategies": sorted(attempted),
                "selection_reason": reason,
            },
        }
