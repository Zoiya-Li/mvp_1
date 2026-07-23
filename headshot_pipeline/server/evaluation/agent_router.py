"""Agent Router — pure decision logic for candidate selection and action routing.

Extracted from GeminiWorker to keep the agent state-machine testable and
reusable without I/O or model loading.
"""

from __future__ import annotations

# ── Constants (mirrored from gemini_worker to avoid circular import) ──

QUALITY_ACCEPT_THRESHOLD = 8
IDENTITY_PASS_THRESHOLD = 8
IDENTITY_REPAIR_THRESHOLD = 7
MAX_PIPELINE_IDENTITY_REPAIRS = 1
MAX_PIPELINE_LOCAL_EDITS = 2
PIPELINE_ALLOWED_ACTIONS = [
    "ACCEPT",
    "LOCAL_EDIT",
    "IDENTITY_REPAIR",
    "REGENERATE_FROM_ORIGINAL",
    "REGENERATE_WITH_POSE_REFERENCE",
    "DROP_CANDIDATE",
    "REQUEST_BETTER_REFERENCE",
]


def _default_identity_threshold_profile() -> dict:
    """Fallback closeup profile when no dependency is injected."""
    return {
        "identity_pass_threshold": 8.0,
        "identity_repair_threshold": 7.0,
        "profile": "closeup",
    }


class AgentRouter:
    """Pure-logic router for candidate actions and selection.

    Accepts either a callable ``identity_threshold_profile`` or a static
    thresholds dict.  All methods are stateless and perform no I/O.
    """

    ACTION_ACCEPT = "ACCEPT"
    ACTION_LOCAL_EDIT = "LOCAL_EDIT"
    ACTION_IDENTITY_REPAIR = "IDENTITY_REPAIR"
    ACTION_REGENERATE_FROM_ORIGINAL = "REGENERATE_FROM_ORIGINAL"
    ACTION_REGENERATE_WITH_POSE_REFERENCE = "REGENERATE_WITH_POSE_REFERENCE"
    ACTION_DROP_CANDIDATE = "DROP_CANDIDATE"
    ACTION_REQUEST_BETTER_REFERENCE = "REQUEST_BETTER_REFERENCE"

    ALLOWED_ACTIONS = PIPELINE_ALLOWED_ACTIONS

    _IDENTITY_REPAIR_BLOCKERS = {
        "unsafe_content",
        "no_face",
        "identity_no_generated_face",
        "no_face_detected",
        "face_distorted",
        "bad_artifacts",
        "severe_artifacts",
        "unreadable_image",
        "bad_resolution",
        "too_blurry",
        "multiple_faces",
        "wrong_composition",
    }

    def __init__(
        self,
        identity_threshold_profile=None,
    ) -> None:
        self._identity_threshold_profile = (
            identity_threshold_profile or _default_identity_threshold_profile
        )

    def _resolve_thresholds(self, identity_thresholds: dict | None) -> dict:
        if identity_thresholds is not None:
            return identity_thresholds
        if callable(self._identity_threshold_profile):
            return self._identity_threshold_profile()
        return dict(self._identity_threshold_profile)

    def decide_candidate_action(
        self,
        judgement: dict,
        edit_count: int = 0,
        identity_repairs: int = 0,
        identity_thresholds: dict | None = None,
    ) -> dict:
        """Bounded state-machine action for one evaluated candidate."""
        from .evaluator import EvaluationService
        from .failure_taxonomy import classify_failure

        scores = judgement.get("scores", {}) or {}
        failures = set(judgement.get("hard_failures") or [])
        action_hint = judgement.get("recommended_action")
        identity = scores.get("identity")
        style_match = scores.get("style_match")
        artifact = scores.get("artifact")
        thresholds = self._resolve_thresholds(identity_thresholds)
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )
        identity_repair_threshold = float(
            thresholds.get("identity_repair_threshold", IDENTITY_REPAIR_THRESHOLD)
        )
        gate = EvaluationService._candidate_gate_status(judgement, thresholds)
        diagnosis = classify_failure(
            judgement,
            gate_failures=gate.get("hard_gate_failures") or [],
        )

        def decision(action: str, reason: str) -> dict:
            return {"action": action, "reason": reason}

        if action_hint == "discard":
            return decision(
                self.ACTION_DROP_CANDIDATE,
                "judge_or_local_gate_marked_discard",
            )
        if not gate["safety_pass"]:
            return decision(self.ACTION_DROP_CANDIDATE, "unsafe_content")
        if not gate["face_detected"]:
            return decision(
                self.ACTION_REGENERATE_FROM_ORIGINAL,
                "no_usable_face_detected",
            )
        if diagnosis["failure_class"] == "identity_geometry":
            return decision(
                self.ACTION_REGENERATE_WITH_POSE_REFERENCE,
                "identity_geometry_requires_pose_matched_reference",
            )
        if identity is None:
            if identity_repairs < MAX_PIPELINE_IDENTITY_REPAIRS:
                return decision(self.ACTION_IDENTITY_REPAIR, "identity_unverified")
            return decision(self.ACTION_DROP_CANDIDATE, "identity_unverified")
        if identity < identity_repair_threshold:
            if (
                identity_repairs < MAX_PIPELINE_IDENTITY_REPAIRS
                and self.identity_repair_is_worthwhile(
                    judgement, identity_thresholds
                )
            ):
                return decision(
                    self.ACTION_IDENTITY_REPAIR,
                    "low_identity_with_usable_composition",
                )
            return decision(
                self.ACTION_REGENERATE_FROM_ORIGINAL,
                "identity_below_repair_threshold",
            )
        if identity < identity_pass_threshold:
            good_composition = style_match is None or style_match >= QUALITY_ACCEPT_THRESHOLD
            if identity_repairs < MAX_PIPELINE_IDENTITY_REPAIRS and good_composition:
                return decision(
                    self.ACTION_IDENTITY_REPAIR,
                    "identity_gray_zone_with_usable_composition",
                )
            return decision(
                self.ACTION_REGENERATE_FROM_ORIGINAL,
                "identity_gray_zone_not_worth_repair",
            )
        if gate["severe_quality_fail"]:
            return decision(diagnosis["recovery_action"], "global_quality_failure")
        if (
            artifact is not None
            and artifact < QUALITY_ACCEPT_THRESHOLD
            and edit_count < MAX_PIPELINE_LOCAL_EDITS
        ):
            return decision(self.ACTION_LOCAL_EDIT, "local_artifact")
        if gate["hard_gates_pass"]:
            return decision(self.ACTION_ACCEPT, "all_hard_gates_pass")
        return decision(diagnosis["recovery_action"], "quality_below_delivery_gate")

    def should_apply_identity_repair(
        self,
        judgement: dict,
        identity_thresholds: dict | None = None,
    ) -> bool:
        """Return True only for identity-gray-zone candidates worth repairing."""
        scores = judgement.get("scores", {})
        identity = scores.get("identity")
        failures = set(judgement.get("hard_failures") or [])
        action = judgement.get("recommended_action")
        thresholds = self._resolve_thresholds(identity_thresholds)
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )
        identity_repair_threshold = float(
            thresholds.get("identity_repair_threshold", IDENTITY_REPAIR_THRESHOLD)
        )

        if identity is None:
            # If the judge failed to score identity, prefer a repair attempt over
            # silently accepting an unverified face.
            return True
        if identity < identity_repair_threshold:
            return self.identity_repair_is_worthwhile(
                judgement, identity_thresholds
            )
        if identity >= identity_pass_threshold:
            return False
        return (
            action == "face_swap"
            or "identity_too_low" in failures
            or identity < identity_pass_threshold
        )

    def identity_repair_is_worthwhile(
        self,
        judgement: dict,
        identity_thresholds: dict | None = None,
    ) -> bool:
        """Allow deterministic identity repair only on an otherwise good frame."""
        scores = judgement.get("scores", {}) or {}
        failures = set(judgement.get("hard_failures") or [])
        identity = scores.get("identity")
        thresholds = self._resolve_thresholds(identity_thresholds)
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )
        if identity is not None and identity >= identity_pass_threshold:
            return False
        if failures & self._IDENTITY_REPAIR_BLOCKERS:
            return False
        for key in ("face_quality", "style_match", "artifact", "commercial_readiness"):
            score = scores.get(key)
            if score is not None and float(score) < QUALITY_ACCEPT_THRESHOLD:
                return False
        return (
            judgement.get("recommended_action") == "face_swap"
            or "identity_too_low" in failures
            or identity is None
        )

    def select_candidate(self, candidates: list[dict]) -> dict | None:
        """Select best candidate from pool."""
        if not candidates:
            return None
        deliverable = [
            c for c in candidates
            if c.get("gate_status", {}).get("hard_gates_pass")
        ]
        if deliverable:
            return max(deliverable, key=self._candidate_rank_key)

        locally_editable = [
            c for c in candidates
            if c.get("agent_action", {}).get("action") == self.ACTION_LOCAL_EDIT
        ]
        if locally_editable:
            return max(locally_editable, key=self._candidate_rank_key)

        repairable = [
            c for c in candidates
            if c.get("agent_action", {}).get("action") == self.ACTION_IDENTITY_REPAIR
        ]
        if repairable:
            return max(repairable, key=self._candidate_rank_key)

        regeneratable = [
            c for c in candidates
            if c.get("agent_action", {}).get("action") in {
                self.ACTION_REGENERATE_FROM_ORIGINAL,
                self.ACTION_REGENERATE_WITH_POSE_REFERENCE,
            }
        ]
        if regeneratable:
            return max(regeneratable, key=self._candidate_rank_key)

        return max(candidates, key=self._candidate_rank_key)

    @staticmethod
    def _candidate_rank_key(candidate: dict) -> tuple[float, float]:
        aggregate = float(candidate.get("aggregate_score") or 0.0)
        identity_quality = (candidate.get("judgement") or {}).get(
            "identity_quality"
        ) or {}
        cosine = float(identity_quality.get("cosine_similarity") or 0.0)
        if candidate.get("selection_profile") == "hero_identity":
            return cosine, aggregate
        return aggregate, cosine

    def candidate_shortlist(self, candidates: list[dict], limit: int = 2) -> list[dict]:
        """Public candidate-funnel summary: top retained candidates, no paths."""
        ranked = sorted(
            candidates,
            key=lambda c: (
                bool(c.get("gate_status", {}).get("hard_gates_pass")),
                *self._candidate_rank_key(c),
            ),
            reverse=True,
        )
        shortlist = []
        for rank, candidate in enumerate(ranked[:limit], start=1):
            gate = candidate.get("gate_status") or {}
            action = candidate.get("agent_action") or {}
            shortlist.append({
                "rank": rank,
                "candidate_id": candidate.get("candidate_id"),
                "candidate_index": candidate.get("index"),
                "filename": candidate.get("filename"),
                "aggregate_score": candidate.get("aggregate_score"),
                "hard_gates_pass": gate.get("hard_gates_pass"),
                "hard_gate_failures": gate.get("hard_gate_failures", []),
                "recommended_action": action.get("action"),
                "action_reason": action.get("reason"),
                "selected": bool(candidate.get("selected")),
            })
        return shortlist
