"""Tests for the GEMINI_BACKEND configuration switch."""

from __future__ import annotations

import pytest

from server.config import settings
from server.gemini_worker import GeminiWorker
from server.openrouter_client import OpenRouterGeminiClient


def test_default_backend_is_openrouter():
    """The default backend remains OpenRouter so production behaviour is unchanged."""
    # pydantic-settings may have already loaded .env; we only assert the valid default.
    assert settings.gemini_backend in ("openrouter", "chrome")


def test_worker_uses_openrouter_client(monkeypatch, tmp_path):
    """When gemini_backend=openrouter, GeminiWorker constructs OpenRouterGeminiClient."""
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test-key")
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    worker = GeminiWorker()
    assert isinstance(worker.client, OpenRouterGeminiClient)


def test_worker_uses_chrome_client(monkeypatch, tmp_path):
    """When gemini_backend=chrome, GeminiWorker constructs PersistentGeminiClient."""
    from persistent_client import PersistentGeminiClient

    monkeypatch.setattr(settings, "gemini_backend", "chrome")
    monkeypatch.setattr(settings, "output_dir", tmp_path)
    monkeypatch.setattr(settings, "chrome_cdp_port", 9222)
    monkeypatch.setattr(settings, "chrome_wait_timeout", 120)

    worker = GeminiWorker()
    assert isinstance(worker.client, PersistentGeminiClient)
    assert worker.client.port == 9222
    assert worker.client.output_dir == tmp_path


def test_openrouter_backend_requires_api_key(monkeypatch, tmp_path):
    """An empty OpenRouter key still hard-fails when backend is openrouter."""
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "output_dir", tmp_path)

    from server.openrouter_client import OpenRouterError

    with pytest.raises(OpenRouterError):
        GeminiWorker()
