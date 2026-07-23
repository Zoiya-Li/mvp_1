"""PortraitAI FastAPI application."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import portrait_storage, storage
from .config import settings
from .job_queue import queue  # singleton

# Import routers (they also import `queue` from job_queue — same singleton)
from .router_sessions import router as sessions_router
from .router_jobs import router as jobs_router
from .router_ws import router as ws_router
from .router_postprocess import router as postprocess_router
from .router_payment import router as payment_router, webhook_router
from .router_portrait_v2 import router as portrait_v2_router
from .router_admin import router as admin_router
from .payment import PaymentService
from .apple_iap import apple_iap_verifier


async def _run_retention_sweep_once() -> None:
    """Apply layered source, generated-media, and metadata retention."""
    now = storage.utcnow()
    source_cutoff = (now - timedelta(days=settings.source_retention_days)).isoformat()
    generated_cutoff = (now - timedelta(days=settings.generated_retention_days)).isoformat()
    metadata_cutoff = (now - timedelta(days=settings.metadata_retention_days)).isoformat()

    for row in storage.list_stale_sessions(source_cutoff):
        sid = row["session_id"]
        await queue.expire_session_sources(sid)
        project = portrait_storage.project_for_legacy_session(sid)
        if project:
            paths = portrait_storage.expire_project_sources(
                project["project_id"], project["user_id"],
            )
            private_dir = (
                settings.upload_dir / "portrait-v2" /
                project["user_id"] / project["project_id"]
            )
            await asyncio.to_thread(shutil.rmtree, private_dir, True)
            for raw_path in paths:
                await asyncio.to_thread(Path(raw_path).unlink, missing_ok=True)

    for row in storage.list_stale_sessions(generated_cutoff):
        sid = row["session_id"]
        project = portrait_storage.project_for_legacy_session(sid)
        if project:
            portrait_storage.delete_project_data(project["project_id"], project["user_id"])
        await queue.expire_session_outputs(sid)

    for row in storage.list_stale_sessions(metadata_cutoff):
        await queue.delete_session(row["session_id"])


async def _retention_sweep():
    """Run the layered retention sweep hourly."""
    while True:
        await asyncio.sleep(3600)
        try:
            await _run_retention_sweep_once()
            print("   retention sweep completed")
        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"⚠ retention sweep error: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize the configured image worker. Shutdown: stop."""
    print("🚀 Starting FlashShot API server...")
    active_model = (
        settings.siliconflow_image_model
        if settings.gemini_backend == "siliconflow"
        else settings.gemini_model
        if settings.gemini_backend == "openrouter"
        else "gemini-web-ui"
    )
    print(f"   Image model: {active_model}")
    provider_key_set = (
        bool(settings.siliconflow_api_key)
        if settings.gemini_backend == "siliconflow"
        else bool(settings.openrouter_api_key)
        if settings.gemini_backend == "openrouter"
        else True
    )
    print(f"   Image backend: {settings.gemini_backend}")
    print(f"   Provider key: {'set' if provider_key_set else 'MISSING — generation will fail'}")
    print(f"   Data dir: {settings.data_dir}")
    print(f"   Payment mock: {'ON (dev only)' if settings.payment_mock_enabled else 'OFF'}")
    print(
        "   Apple IAP: "
        f"{'configured' if not apple_iap_verifier.configuration_errors() else 'NOT configured'}"
    )
    print(f"   Paddle: {'configured' if PaymentService.is_paddle_configured() else 'NOT configured (mock or dev)'}")

    await queue.start()
    if queue.generation_ready:
        print("✓ Generation worker ready")
    else:
        print("⚠ API started without a generation-ready worker")

    sweep_task = asyncio.create_task(_retention_sweep())

    yield

    sweep_task.cancel()
    print("⏹ Shutting down...")
    await queue.stop()


app = FastAPI(
    title="FlashShot API",
    description="FlashShot - private, quality-gated AI portrait stories",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

access_logger = logging.getLogger("flashshot.access")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,80}", request_id):
        request_id = uuid.uuid4().hex
    started = time.monotonic()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        access_logger.info(json.dumps({
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "duration_ms": round((time.monotonic() - started) * 1000, 1),
        }, separators=(",", ":")))

app.include_router(sessions_router)
app.include_router(jobs_router)
app.include_router(ws_router)
app.include_router(postprocess_router)
app.include_router(payment_router)
# The webhook lives under /api/payments (NOT session-scoped) so Paddle can hit
# it without an owner token — auth is by HMAC-SHA256 signature instead.
app.include_router(webhook_router)
app.include_router(portrait_v2_router)
app.include_router(admin_router)


@app.get("/api/health")
async def health():
    await queue.refresh_provider_readiness(max_age_seconds=30)
    apple_errors = apple_iap_verifier.configuration_errors()
    return {
        "status": "ok",
        "queue_length": queue.queue_length(),
        "is_busy": queue.is_busy,
        "generation_ready": queue.generation_ready,
        "apple_iap_ready": not apple_errors,
        "web_payment_ready": PaymentService.is_paddle_configured(),
    }


@app.get("/api/ready")
async def ready():
    """Strict release/load-balancer readiness, including production config."""
    await queue.refresh_provider_readiness(max_age_seconds=30)
    config_errors = settings.production_readiness_errors()
    checks = {
        "generation_worker": queue.generation_ready,
        "production_config": not config_errors,
    }
    is_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
            "configuration_errors": config_errors,
            "provider_error": queue.worker_readiness_error,
        },
    )


@app.get("/api/launch-ready")
async def launch_ready():
    """Paid-production launch gate, independent of the current environment."""
    await queue.refresh_provider_readiness(max_age_seconds=30)
    config_errors = settings.launch_readiness_errors()
    checks = {
        "generation_worker": queue.generation_ready,
        "production_environment": settings.app_environment == "production",
        "apple_iap_configured": not apple_iap_verifier.configuration_errors(),
        "launch_config": not config_errors,
    }
    is_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "status": "launch_ready" if is_ready else "not_launch_ready",
            "checks": checks,
            "configuration_errors": config_errors,
            "provider_error": queue.worker_readiness_error,
            "web_payment": {
                "configured": PaymentService.is_paddle_configured(),
                "required_for_ios_launch": False,
                "configuration_errors": settings.web_payment_readiness_errors(),
            },
        },
    )


@app.get("/api/config/public")
async def public_config():
    """Public, unauthenticated site config — things the marketing/footer UI needs
    before any session exists (ICP 备案号 for the footer, etc.). No secrets here.
    """
    return {
        "icp_beian": settings.icp_beian,
        "paddle_client_token": settings.paddle_client_token,
        "paddle_environment": settings.paddle_environment,
        "checkout_available": (
            PaymentService.is_paddle_configured() or PaymentService.is_mock()
        ),
    }
