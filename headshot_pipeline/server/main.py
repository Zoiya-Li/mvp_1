"""PortraitAI FastAPI application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """Startup: init the OpenRouter Gemini worker. Shutdown: stop."""
    print("🚀 Starting FlashShot API server...")
    print(f"   Gemini model: {settings.gemini_model}")
    print(f"   OpenRouter key: {'set' if settings.openrouter_api_key else 'MISSING — generation will fail'}")
    print(f"   Data dir: {settings.data_dir}")
    print(f"   Payment mock: {'ON (dev only)' if settings.payment_mock_enabled else 'OFF'}")
    print(f"   Paddle: {'configured' if PaymentService.is_paddle_configured() else 'NOT configured (mock or dev)'}")

    await queue.start()
    print("✓ Worker started")

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
    }


@app.get("/api/config/public")
async def public_config():
    """Public, unauthenticated site config — things the marketing/footer UI needs
    before any session exists (ICP 备案号 for the footer, etc.). No secrets here.
    """
    return {"icp_beian": settings.icp_beian}
