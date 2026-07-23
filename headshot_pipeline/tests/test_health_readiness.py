"""Liveness and strict production-readiness contract tests."""

from __future__ import annotations

import json

import pytest

from server.config import settings
from server.main import health, launch_ready, public_config, queue, ready


@pytest.mark.asyncio
async def test_health_is_liveness_and_reports_worker_state(monkeypatch):
    monkeypatch.setattr(queue, "_worker", None)

    payload = await health()

    assert payload["status"] == "ok"
    assert payload["generation_ready"] is False


@pytest.mark.asyncio
async def test_ready_rejects_missing_generation_worker(monkeypatch):
    monkeypatch.setattr(queue, "_worker", None)
    monkeypatch.setattr(queue, "_worker_readiness_error", "regional restriction")
    monkeypatch.setattr(settings, "app_environment", "development")

    response = await ready()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["checks"]["generation_worker"] is False
    assert payload["provider_error"] == "regional restriction"


@pytest.mark.asyncio
async def test_ready_accepts_development_worker(monkeypatch):
    worker = type(
        "ReadyWorker",
        (),
        {"provider_readiness": {"pass": True}},
    )()
    monkeypatch.setattr(queue, "_worker", worker)
    monkeypatch.setattr(settings, "app_environment", "development")

    response = await ready()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "ready"


def test_production_config_errors_are_release_blocking(monkeypatch):
    monkeypatch.setattr(settings, "app_environment", "production")
    monkeypatch.setattr(settings, "gemini_backend", "openrouter")
    monkeypatch.setattr(settings, "_session_secret_generated", True)
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    monkeypatch.setattr(settings, "payment_mock_enabled", True)
    monkeypatch.setattr(settings, "apple_iap_environment", "sandbox")
    monkeypatch.setattr(settings, "apple_app_id", None)
    monkeypatch.setattr(settings, "face_swap_enabled", False)

    errors = settings.production_readiness_errors()

    assert "SESSION_SECRET_KEY must be persistent in production" in errors
    assert "OPENROUTER_API_KEY is missing" in errors
    assert "PAYMENT_MOCK_ENABLED must be off in production" in errors
    assert "APPLE_IAP_ENVIRONMENT must be production" in errors
    assert "APPLE_APP_ID is missing" in errors


def test_production_requires_siliconflow_key_when_selected(monkeypatch):
    monkeypatch.setattr(settings, "app_environment", "production")
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    monkeypatch.setattr(settings, "siliconflow_api_key", "")
    monkeypatch.setattr(settings, "face_swap_enabled", False)

    errors = settings.production_readiness_errors()

    assert "SILICONFLOW_API_KEY is missing" in errors
    assert "OPENROUTER_API_KEY is missing" not in errors


@pytest.mark.asyncio
async def test_launch_ready_reports_staging_and_apple_payment_blockers(
    monkeypatch, tmp_path,
):
    worker = type("ReadyWorker", (), {"provider_readiness": {"pass": True}})()
    monkeypatch.setattr(queue, "_worker", worker)
    monkeypatch.setattr(queue, "_worker_readiness_error", None)
    monkeypatch.setattr(settings, "app_environment", "staging")
    monkeypatch.setattr(settings, "gemini_backend", "siliconflow")
    monkeypatch.setattr(settings, "siliconflow_api_key", "sf-test")
    monkeypatch.setattr(settings, "payment_mock_enabled", False)
    monkeypatch.setattr(settings, "paddle_environment", "sandbox")
    monkeypatch.setattr(settings, "paddle_api_key", "")
    monkeypatch.setattr(settings, "paddle_client_token", "")
    monkeypatch.setattr(settings, "paddle_webhook_secret", "")
    monkeypatch.setattr(settings, "paddle_price_standard_id", "")
    monkeypatch.setattr(settings, "paddle_price_premium_id", "")
    monkeypatch.setattr(settings, "face_swap_enabled", False)
    monkeypatch.setattr(settings, "apple_root_cert_dir", tmp_path / "no-apple-certs")

    response = await launch_ready()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["status"] == "not_launch_ready"
    assert payload["checks"]["generation_worker"] is True
    assert payload["checks"]["production_environment"] is False
    assert payload["checks"]["apple_iap_configured"] is False
    assert "APP_ENVIRONMENT must be production" in payload["configuration_errors"]
    assert "Apple root certificates are missing" in payload["configuration_errors"]
    assert payload["web_payment"]["required_for_ios_launch"] is False
    assert "PADDLE_API_KEY is missing" in payload["web_payment"]["configuration_errors"]


@pytest.mark.asyncio
async def test_public_config_fail_closes_checkout_without_exposing_secrets(monkeypatch):
    monkeypatch.setattr(settings, "payment_mock_enabled", False)
    monkeypatch.setattr(settings, "paddle_api_key", "")
    monkeypatch.setattr(settings, "paddle_client_token", "")
    monkeypatch.setattr(settings, "paddle_webhook_secret", "")
    monkeypatch.setattr(settings, "paddle_price_standard_id", "")
    monkeypatch.setattr(settings, "paddle_price_premium_id", "")

    payload = await public_config()

    assert payload["checkout_available"] is False
    assert "paddle_api_key" not in payload
    assert "paddle_webhook_secret" not in payload


@pytest.mark.asyncio
async def test_public_config_allows_explicit_local_mock_checkout(monkeypatch):
    monkeypatch.setattr(settings, "payment_mock_enabled", True)

    payload = await public_config()

    assert payload["checkout_available"] is True
