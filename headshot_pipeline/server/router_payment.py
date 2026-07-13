"""Payment endpoints — create order, poll status, Paddle webhook.

SECURITY MODEL:
  - ``create_payment`` / ``get_session_payment`` are ownership-gated (caller
    must hold the session's owner token).
  - ``poll_payment_status`` ONLY reads status. It NEVER upgrades a tier. This
    closes the old forgery hole where any client could flip their tier by
    polling after a mock auto-confirm.
  - The ONLY tier-upgrade path is the Paddle webhook, whose HMAC-SHA256
    signature is verified against ``PADDLE_WEBHOOK_SECRET`` before
    ``apply_paid_webhook`` runs. See ``payment.apply_paid_webhook``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .job_queue import queue
from .models import (
    CreatePaymentRequest,
    PaymentResponse,
    PaymentStatusResponse,
    PricingTier,
    TIER_LIMITS,
)
from . import payment as payment_mod
from .payment import PaymentService
from .security import require_owner

router = APIRouter(prefix="/api/sessions", tags=["payment"])
webhook_router = APIRouter(prefix="/api/payments", tags=["payment-webhook"])

# Replay window: reject webhooks whose timestamp is older than this (seconds).
# Paddle signs `ts:raw_body`; an attacker can't forge a fresh signature without
# the secret, but we still cap staleness to limit replay.
PADDLE_WEBHOOK_MAX_AGE_SECONDS = 300


@router.post("/{session_id}/payment", response_model=PaymentResponse)
async def create_payment(
    session_id: str, req: CreatePaymentRequest, state=Depends(require_owner)
):
    """Create a payment order for upgrading tier.

    Returns an approved Paddle.js ``checkout_url`` the browser redirects to. Order
    creation calls Paddle's API (network) so it runs off the event loop.
    """
    # Don't allow downgrading
    tier_order = [PricingTier.free, PricingTier.standard, PricingTier.premium]
    current_idx = tier_order.index(state.tier)
    new_idx = tier_order.index(req.tier)
    if new_idx <= current_idx:
        raise HTTPException(400, f"Already on {state.tier.value} or higher")

    try:
        # Offload the (possibly network-bound) order creation to a thread so a
        # slow Paddle response never blocks the event loop.
        record = await asyncio.to_thread(
            PaymentService.create_order, session_id, req.tier
        )
    except RuntimeError as exc:
        # Payment not configured (no mock + no Paddle creds).
        raise HTTPException(503, str(exc))
    return record


@router.get("/{session_id}/payment", response_model=PaymentResponse | None)
async def get_session_payment(session_id: str, state=Depends(require_owner)):
    """Get the active (paid) payment for a session."""
    return PaymentService.get_payment_for_session(session_id)


@router.get(
    "/{session_id}/payment/{payment_id}/status",
    response_model=PaymentStatusResponse,
)
async def poll_payment_status(
    session_id: str,
    payment_id: str,
    state=Depends(require_owner),
):
    """Poll payment status (frontend calls this every few seconds).

    READ-ONLY. Does not upgrade the tier — only the verified webhook does that.
    The tier in the response reflects whatever the webhook has already applied.
    """
    record = PaymentService.get_payment(payment_id)
    if not record or record.session_id != session_id:
        raise HTTPException(404, "Payment not found")
    return PaymentStatusResponse(
        payment_id=record.payment_id,
        status=record.status,
        tier=record.tier,
    )


@router.get("/pricing/tiers")
async def get_pricing_tiers():
    """Return all pricing tiers with their limits."""
    result = []
    for tier in PricingTier:
        limits = TIER_LIMITS[tier]
        result.append({
            "tier": tier.value,
            "label": limits["label"],
            "price_cents": limits["price_cents"],
            "max_styles": limits["max_styles"],
            "max_revisions": limits["max_revisions"],
            "allow_id_photo": limits["allow_id_photo"],
            "allow_bg_replace": limits["allow_bg_replace"],
            "allow_hd_download": limits["allow_hd_download"],
        })
    return {"tiers": result}


# ── Paddle webhook (the ONLY tier-upgrade path) ──────────

def verify_paddle_signature(raw_body: bytes, paddle_signature: str,
                            secret: str) -> bool:
    """Verify a Paddle webhook signature.

    Header format: ``Paddle-Signature: ts=<unix_ts>;h1=<hex_hmac_sha256>``.
    Signed payload: ``"<ts>:<raw_body>"`` (raw body bytes, pre-JSON-parse),
    HMAC-SHA256 keyed by the webhook secret, hex-encoded. Constant-time
    compare. Also enforces a freshness window on ``ts`` to bound replay.
    """
    if not paddle_signature or not secret:
        return False
    parts: dict[str, list[str]] = {}
    for chunk in paddle_signature.split(";"):
        if "=" in chunk:
            key, val = chunk.split("=", 1)
            parts.setdefault(key.strip(), []).append(val.strip())
    timestamps = parts.get("ts", [])
    signatures = parts.get("h1", [])
    if len(timestamps) != 1 or not signatures:
        return False
    ts = timestamps[0]

    # Freshness: reject stale timestamps.
    try:
        ts_int = int(ts)
    except ValueError:
        return False
    if abs(int(time.time()) - ts_int) > PADDLE_WEBHOOK_MAX_AGE_SECONDS:
        return False

    signed = f"{ts}:".encode("utf-8") + raw_body
    expected = hmac.new(
        secret.encode("utf-8"), signed, hashlib.sha256
    ).hexdigest()
    return any(hmac.compare_digest(expected, signature) for signature in signatures)


@webhook_router.post("/paddle/webhook")
async def paddle_webhook(request: Request):
    """Paddle payment webhook (Merchant of Record).

    Verifies the HMAC-SHA256 ``Paddle-Signature`` over the raw body, then on a
    completed/paid one-time transaction marks the matching payment paid and
    promotes the session tier. All other events are acknowledged (200) so
    Paddle stops retrying. Signature failure → 401 so Paddle retries.
    """
    raw_body = await request.body()
    signature = request.headers.get("Paddle-Signature", "")

    if not verify_paddle_signature(
        raw_body, signature, payment_mod.settings.paddle_webhook_secret
    ):
        raise HTTPException(401, "signature verification failed")

    try:
        envelope = json.loads(raw_body)
    except Exception:
        raise HTTPException(400, "malformed callback")

    event_type = envelope.get("event_type", "")
    data = envelope.get("data") or {}
    custom = data.get("custom_data") or {}
    payment_id = custom.get("payment_id")

    # Only a captured one-time payment upgrades the tier.
    if event_type in ("transaction.completed", "transaction.paid"):
        if payment_id:
            # amount_cents=None: Paddle pins the price server-side via price_id
            # (see payment.py docstring); the HMAC + payment_id binding is the
            # security guarantee, not an amount check.
            try:
                queue.apply_payment_tier_upgrade(
                    payment_id,
                    None,
                    provider_transaction_id=data.get("id"),
                )
            except ValueError:
                # A future strict-amount gateway could land here. Paddle does
                # not, but keep the contract so the path is obvious.
                raise HTTPException(400, "amount mismatch")
        # Unknown/missing payment_id → still 200 so Paddle stops retrying an
        # old or irrelevant message.

    # Paddle Billing refund requests are adjustment records. Only approved
    # refund adjustments count as refunds for product metrics.
    if (
        event_type == "adjustment.updated"
        and data.get("action") == "refund"
        and data.get("status") == "approved"
    ):
        queue.apply_payment_refund(
            payment_id=payment_id,
            provider_transaction_id=data.get("transaction_id"),
        )

    return JSONResponse({"success": True})
