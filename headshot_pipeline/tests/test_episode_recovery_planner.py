from server.evaluation.recovery_planner import EpisodeRecoveryPlanner


def _decision(failure_class: str, action: str) -> dict:
    return {
        "failure_class": failure_class,
        "action": action,
        "recovery_strategy": "initial",
    }


def _executed(failure_class: str, action: str, strategy: str, route="primary") -> dict:
    return {
        "failure_class": failure_class,
        "action": action,
        "recovery_strategy": strategy,
        "route_mode": route,
        "executed": True,
    }


def test_first_attempt_preserves_matching_base_action():
    planned = EpisodeRecoveryPlanner().plan(
        _decision("synthetic_texture", "REGENERATE_FROM_ORIGINAL"),
        [],
    )

    assert planned["action"] == "REGENERATE_FROM_ORIGINAL"
    assert planned["route_mode"] == "primary"
    assert planned["recovery_strategy"] == "photoreal_regeneration"
    assert planned["recovery_plan"]["failure_streak"] == 0


def test_identity_failure_changes_mechanism_after_failed_writeback():
    history = [
        _executed(
            "identity_similarity",
            "IDENTITY_REPAIR",
            "identity_writeback",
        )
    ]

    planned = EpisodeRecoveryPlanner().plan(
        _decision("identity_similarity", "IDENTITY_REPAIR"),
        history,
    )

    assert planned["action"] == "REGENERATE_WITH_POSE_REFERENCE"
    assert planned["recovery_strategy"] == "pose_anchored_identity_reset"


def test_repeated_texture_failure_switches_to_approved_alternate_route():
    history = [
        _executed(
            "synthetic_texture",
            "REGENERATE_FROM_ORIGINAL",
            "photoreal_regeneration",
        )
    ]

    planned = EpisodeRecoveryPlanner().plan(
        _decision("synthetic_texture", "REGENERATE_FROM_ORIGINAL"),
        history,
        alternate_route_available=True,
    )

    assert planned["action"] == "REGENERATE_FROM_ORIGINAL"
    assert planned["route_mode"] == "alternate"
    assert planned["recovery_strategy"] == "alternate_model_texture_reset"


def test_unavailable_alternate_is_not_pretended_to_be_a_new_strategy():
    history = [
        _executed(
            "synthetic_texture",
            "REGENERATE_FROM_ORIGINAL",
            "photoreal_regeneration",
        )
    ]

    planned = EpisodeRecoveryPlanner().plan(
        _decision("synthetic_texture", "REGENERATE_FROM_ORIGINAL"),
        history,
        alternate_route_available=False,
    )

    assert planned["action"] == "DROP_CANDIDATE"
    assert planned["route_mode"] == "none"


def test_unexecuted_recommendations_do_not_consume_ladder_steps():
    history = [{
        **_executed(
            "composition",
            "REGENERATE_FROM_ORIGINAL",
            "composition_regeneration",
        ),
        "executed": False,
    }]

    planned = EpisodeRecoveryPlanner().plan(
        _decision("composition", "REGENERATE_FROM_ORIGINAL"),
        history,
    )

    assert planned["action"] == "REGENERATE_FROM_ORIGINAL"
    assert planned["recovery_plan"]["failure_streak"] == 0


def test_first_policy_override_is_preserved_even_when_not_in_ladder():
    planned = EpisodeRecoveryPlanner().plan(
        _decision("identity_similarity", "REGENERATE_FROM_ORIGINAL"),
        [],
    )

    assert planned["action"] == "REGENERATE_FROM_ORIGINAL"
    assert planned["recovery_plan"]["selection_reason"] == (
        "base_action_outside_specialized_ladder"
    )
