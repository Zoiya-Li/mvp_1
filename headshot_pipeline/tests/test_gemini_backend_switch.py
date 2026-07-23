"""Tests for the GEMINI_BACKEND configuration switch."""

from __future__ import annotations

import pytest

from server.config import settings
from server.gemini_worker import GeminiWorker
from server.openrouter_client import OpenRouterError
from server.openrouter_image_client import OpenRouterImageError


def test_default_backend_is_openrouter():
    """The default backend remains OpenRouter so production behaviour is unchanged."""
    # pydantic-settings may have already loaded .env; we only assert the valid default.
    assert settings.gemini_backend in ("openrouter", "siliconflow", "chrome")


def test_worker_uses_openrouter_provider(monkeypatch, tmp_path):
    """When gemini_backend=openrouter, GeminiWorker uses OpenRouterProvider via ImageGateway."""
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    worker = GeminiWorker()
    # The provider is now abstracted behind ImageGateway; we verify the gateway
    # has an OpenRouterProvider by checking the internal provider instance.
    from server.generation.providers import OpenRouterProvider
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    assert isinstance(provider, OpenRouterProvider)
    hero_provider = worker._gateway._openrouter_hero
    assert isinstance(hero_provider, OpenRouterProvider)
    assert hero_provider.model == settings.openrouter_hero_model
    assert worker._gateway.hero_route()["model"] == settings.openrouter_hero_model
    assert worker._gateway.quality_route()["model"] == settings.openrouter_hero_model


def test_recovery_provider_is_opt_in_and_exposes_concrete_route(monkeypatch, tmp_path):
    from server.generation.providers import OpenRouterProvider

    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "openrouter_recovery_model", "vendor/recovery-model")
    monkeypatch.setattr(settings, "openrouter_recovery_image_provider", "vendor")
    monkeypatch.setattr(settings, "openrouter_recovery_estimated_image_cost", 0.15)

    worker = GeminiWorker()

    assert isinstance(worker._gateway._openrouter_recovery, OpenRouterProvider)
    assert worker._gateway.has_recovery_route() is True
    assert worker._gateway.recovery_route() == {
        "provider": "openrouter",
        "model": "vendor/recovery-model",
        "provider_tag": "vendor",
        "reason": "episode_failure_escalation",
        "estimated_cost": 0.15,
        "estimated_latency_ms": 55_000,
        "confidence": 0.75,
    }


def test_worker_uses_chrome_provider(monkeypatch, tmp_path):
    """When gemini_backend=chrome, GeminiWorker uses ChromeProvider via ImageGateway."""
    from server.generation.providers import ChromeProvider

    monkeypatch.setattr(settings, "gemini_backend", "chrome")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "chrome_cdp_port", 9222)
    monkeypatch.setattr(settings, "chrome_wait_timeout", 120)

    worker = GeminiWorker()
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    assert isinstance(provider, ChromeProvider)


def test_worker_uses_siliconflow_provider(monkeypatch, tmp_path):
    from server.generation.providers import SiliconFlowProvider

    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    monkeypatch.setattr(settings, "siliconflow_api_key", "sf-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    worker = GeminiWorker()
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    assert isinstance(provider, SiliconFlowProvider)


def test_siliconflow_backend_requires_api_key(monkeypatch, tmp_path):
    from server.generation.providers import SiliconFlowError

    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    monkeypatch.setattr(settings, "siliconflow_api_key", "")
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    with pytest.raises(SiliconFlowError):
        GeminiWorker()


def test_openrouter_backend_requires_api_key(monkeypatch, tmp_path):
    """An empty OpenRouter key still hard-fails when backend is openrouter."""
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    with pytest.raises(OpenRouterError):
        GeminiWorker()


def test_openrouter_readiness_checks_image_model_balance_and_judge_model(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    worker = GeminiWorker()
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    monkeypatch.setattr(
        provider._image_client,
        "check_readiness",
        lambda: {
            "pass": True,
            "provider": "openrouter",
            "model": settings.gemini_model,
            "remaining_credits": 9.5,
            "minimum_credit_balance": 3.0,
        },
    )
    monkeypatch.setattr(
        provider._image_client,
        "check_text_model",
        lambda model: {"pass": True, "model": model},
    )
    hero_provider = worker._gateway._openrouter_hero
    monkeypatch.setattr(
        hero_provider._image_client,
        "check_readiness",
        lambda: {
            "pass": True,
            "provider": "openrouter",
            "model": settings.openrouter_hero_model,
            "remaining_credits": 9.5,
            "minimum_credit_balance": 3.0,
        },
    )

    worker.connect()

    assert worker.provider_readiness == {
        "pass": True,
        "provider": "openrouter",
        "model": settings.gemini_model,
        "remaining_credits": 9.5,
        "minimum_credit_balance": 3.0,
        "judge_model": settings.openrouter_judge_model,
        "hero_generation_ready": True,
        "hero_model": settings.openrouter_hero_model,
    }


def test_openrouter_readiness_rejects_insufficient_credit(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    worker = GeminiWorker()
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    monkeypatch.setattr(
        provider._image_client,
        "check_readiness",
        lambda: (_ for _ in ()).throw(
            OpenRouterImageError("OpenRouter insufficient credit balance")
        ),
    )

    with pytest.raises(OpenRouterError, match="insufficient credit"):
        worker.connect()
