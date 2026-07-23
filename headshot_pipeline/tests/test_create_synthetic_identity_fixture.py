from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/create_synthetic_identity_fixture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "create_synthetic_identity_fixture", SCRIPT
)
fixture_creator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fixture_creator)


def test_fixture_views_cover_front_expression_and_both_angles():
    filenames = [item[0] for item in fixture_creator.VIEW_SPECS]
    views = " ".join(item[1] for item in fixture_creator.VIEW_SPECS)

    assert len(filenames) == len(set(filenames)) == 4
    assert "neutral" in views
    assert "smile" in views
    assert "left" in views
    assert "right" in views


def test_fixture_prompt_is_identity_locked_and_unretouched():
    prompt = fixture_creator.build_view_prompt("front-facing neutral")

    assert "Identity is the hard constraint" in prompt
    assert "unretouched" in prompt
    assert "Do not beautify" in prompt
    assert "additional people" in prompt
