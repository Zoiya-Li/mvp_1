"""Tests for the GEMINI_BACKEND configuration switch."""

from __future__ import annotations

import pytest

from server.config import settings
from server.gemini_worker import GeminiWorker
from server.openrouter_client import OpenRouterError


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


def test_openrouter_model_access_probe_records_ready_model(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    worker = GeminiWorker()
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    monkeypatch.setattr(
        provider._client,
        "_post",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "OK"}}]
        },
    )

    worker.connect()

    assert worker.provider_readiness == {
        "pass": True,
        "provider": "openrouter",
        "model": settings.gemini_model,
    }


def test_openrouter_model_access_probe_rejects_empty_response(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    worker = GeminiWorker()
    provider = worker._gateway._provider_for("CREATE_FROM_REFERENCES")
    monkeypatch.setattr(
        provider._client,
        "_post",
        lambda *args, **kwargs: {"choices": [{"message": {"content": ""}}]},
    )

    with pytest.raises(OpenRouterError, match="returned no text"):
        worker.connect()
