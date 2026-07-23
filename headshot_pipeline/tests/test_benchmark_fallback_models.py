from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/benchmark_fallback_models.py"
SPEC = importlib.util.spec_from_file_location("benchmark_fallback_models", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_parse_candidate_and_summary():
    candidate = benchmark.parse_candidate("gpt2,openai/gpt-image-2,openai,0.20")
    assert candidate == {
        "name": "gpt2",
        "model": "openai/gpt-image-2",
        "provider_tag": "openai",
        "estimated_cost": 0.2,
    }
    summary = benchmark.summarize_variant([
        {
            "estimated_cost": 0.2,
            "gate_status": {"hard_gates_pass": True},
            "judgement": {"identity": {"cosine_similarity": 0.82}},
        },
        {
            "estimated_cost": 0.2,
            "gate_status": {"hard_gates_pass": False},
            "judgement": {"identity": {"cosine_similarity": 0.76}},
        },
    ])
    assert summary["pass_rate"] == 0.5
    assert summary["mean_identity_cosine"] == 0.79
    assert summary["estimated_cost"] == 0.4
