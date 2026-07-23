from __future__ import annotations

import pytest

from server.inspiration_analyzer import (
    analyze_with_provider,
    inspiration_generation_prompt,
    parse_json_object,
)


class FakeProvider:
    def __init__(self, payload: str):
        self.payload = payload
        self.call = None

    def judge(self, **kwargs):
        self.call = kwargs
        return self.payload


def test_parses_fenced_json_and_forces_identity_exclusion():
    provider = FakeProvider(
        """```json
        {
          "scene": "cafe",
          "wardrobe": "cream knit",
          "lighting": "window light",
          "composition": "half body",
          "pose": "seated",
          "palette": ["cream", "green"],
          "mood": "quiet",
          "camera": "50mm",
          "complexity": "low",
          "safety": {"pass": true, "reasons": []},
          "forbidden_transfer": []
        }
        ```"""
    )

    result = analyze_with_provider(provider, "/tmp/inspiration.png")

    assert "source_person_identity" in result["forbidden_transfer"]
    assert "watermarks" in result["forbidden_transfer"]
    assert provider.call["reference_paths"] == []


def test_rejects_unsafe_inspiration():
    provider = FakeProvider(
        '{"scene":"x","wardrobe":"x","lighting":"x",'
        '"composition":"x","pose":"x","mood":"x",'
        '"safety":{"pass":false,"reasons":["unsafe_content"]}}'
    )
    with pytest.raises(ValueError, match="unsafe_content"):
        analyze_with_provider(provider, "/tmp/inspiration.png")


def test_invalid_analysis_is_not_silently_accepted():
    with pytest.raises((ValueError, TypeError)):
        parse_json_object("not json")


def test_full_set_prompt_keeps_identity_boundary_without_hero_framing():
    prompt = inspiration_generation_prompt({
        "scene": "rainy street",
        "wardrobe": "dark coat",
        "lighting": "neon rim light",
        "composition": "editorial",
        "pose": "three-quarter turn",
        "mood": "cinematic",
    }, hero_only=False)

    assert "Never copy the source person's face" in prompt
    assert "Follow the supplied ShotSpec" in prompt
    assert "close-up hero portrait" not in prompt
