"""PortraitAI FastAPI application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import storage
from .config import settings
from .job_queue import queue  # singleton

# Import routers (they also import `queue` from job_queue — same singleton)
from .router_sessions import router as sessions_router
from .router_jobs import router as jobs_router
from .router_ws import router as ws_router
from .router_postprocess import router as postprocess_router
from .router_payment import router as payment_router, webhook_router
from .payment import PaymentService


async def _retention_sweep():
    """Periodically delete sessions older than ``retention_days``.

    Runs hourly; sweeps both the on-disk files and the SQLite row. This bounds
    how long we keep user face photos on disk (privacy + disk hygiene).
    """
    while True:
        await asyncio.sleep(3600)
        try:
            cutoff = (storage.utcnow() - timedelta(days=settings.retention_days)).isoformat()
            stale = storage.list_stale_sessions(cutoff)
            for row in stale:
                sid = row["session_id"]
                await queue.delete_session(sid)
                print(f"   🗑 retention sweep removed stale session {sid}")
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
    description="FlashShot — AI professional headshot generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(jobs_router)
app.include_router(ws_router)
app.include_router(postprocess_router)
app.include_router(payment_router)
# The webhook lives under /api/payments (NOT session-scoped) so Paddle can hit
# it without an owner token — auth is by HMAC-SHA256 signature instead.
app.include_router(webhook_router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "queue_length": queue.queue_length(),
        "is_busy": queue.is_busy,
        "generation_ready": queue.generation_ready,
    }


@app.get("/api/ready")
async def ready():
    """Strict release/load-balancer readiness, including production config."""
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
    config_errors = settings.launch_readiness_errors()
    checks = {
        "generation_worker": queue.generation_ready,
        "production_environment": settings.app_environment == "production",
        "payment_configured": PaymentService.is_paddle_configured(),
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
    }
