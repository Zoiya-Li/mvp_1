"""Payment service — Paddle (Merchant of Record) for the overseas build.

SECURITY MODEL (unchanged from the WeChat version it replaces):
  - The ONLY place a session tier is upgraded is ``apply_paid_webhook``, which
    is called exclusively from the signature-verified Paddle webhook
    (``router_payment.paddle_webhook``). Client polling NEVER upgrades a tier —
    it only *reads* status.
  - Mock auto-confirm is OFF by default. Set PAYMENT_MOCK_ENABLED=1 for local
    dev only; in production the real Paddle webhook is the source of truth.
  - All payment records persist to SQLite (storage.py) so a restart does not
    lose paid orders.

PADDLE INTEGRATION (Billing, paddle.com):
  Requires env config (placeholders until you open a Paddle account):
    PADDLE_API_KEY            — server API key (sandbox or production)
    PADDLE_WEBHOOK_SECRET     — webhook signing key from the dashboard
    PADDLE_ENVIRONMENT        — "sandbox" | "production"
    PADDLE_PRICE_STANDARD_ID  — price_id (pri_...) for the $5 Standard tier
    PADDLE_PRICE_PREMIUM_ID   — price_id (pri_...) for the $10 Pro tier
    PADDLE_RETURN_URL         — post-checkout redirect target

  When those are present, ``create_order`` calls Paddle's transactions API to
  obtain a ``checkout.url`` on the approved Paddle.js page and the
  webhook verifies the HMAC-SHA256 ``Paddle-Signature`` before marking paid.
  When they are absent AND mock is off, ``create_order`` raises a clear error —
  it never silently gives away premium.

WHY NO AMOUNT VERIFICATION (unlike the WeChat path it replaces):
  Paddle is a Merchant of Record: the charged amount is pinned server-side by
  the price_id we send, and the buyer cannot alter it. The webhook event is
  HMAC-authenticated and bound to our order via ``custom_data.payment_id``.
  There is no client-supplied-amount vector (as there was with WeChat's
  callback), so ``apply_paid_webhook`` is called with amount_cents=None.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

from .config import settings
from .models import (
    PaymentResponse,
    PaymentStatus,
    PricingTier,
    TIER_LIMITS,
)
from . import storage


# ── In-memory cache of payment records (backed by SQLite) ──
# We keep a hot dict for fast polling reads, but every write also hits SQLite
# so state survives restarts. On startup, _load_from_db() repopulates this.
_payments: dict[str, PaymentResponse] = {}


def _payment_response_from_row(row) -> PaymentResponse:
    return PaymentResponse(
        payment_id=row["payment_id"],
        session_id=row["session_id"],
        tier=PricingTier(row["tier"]),
        status=PaymentStatus(row["status"]),
        checkout_url=None,
        amount_cents=row["amount_cents"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _load_from_db():
    """Repopulate the in-memory cache from SQLite (called at startup)."""
    try:
        with storage.get_conn() as conn:
            rows = conn.execute(
                "SELECT payment_id, session_id, tier, status, amount_cents, "
                "created_at FROM payments"
            ).fetchall()
        for r in rows:
            _payments[r["payment_id"]] = _payment_response_from_row(r)
    except Exception:
        # DB not initialized yet — init_db() runs first in practice
        pass


class PaymentService:
    """Creates and manages payment orders."""

    @staticmethod
    def is_mock() -> bool:
        """Mock mode is explicit opt-in. Default OFF (was silently ON before)."""
        return bool(settings.payment_mock_enabled)

    @staticmethod
    def is_paddle_configured() -> bool:
        return all([
            settings.paddle_api_key,
            settings.paddle_client_token,
            settings.paddle_webhook_secret,
            _paddle_price_id_for_tier(PricingTier.standard),
            _paddle_price_id_for_tier(PricingTier.premium),
        ])

    @staticmethod
    def schedule_mock_confirmation(payment_id: str, delay: int = 5) -> bool:
        """Schedule dev-only payment confirmation on the caller's event loop.

        Gateway order creation runs in a worker thread in the HTTP routes. A
        worker thread has no running asyncio loop, so the route calls this once
        it returns to FastAPI's loop. Keeping the helper here also preserves
        direct async development callers of ``create_order``.
        """
        if not PaymentService.is_mock():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        loop.create_task(_mock_auto_confirm(payment_id, delay=delay))
        return True

    @staticmethod
    def create_order(
        session_id: str,
        tier: PricingTier,
        return_params: dict[str, str] | None = None,
    ) -> PaymentResponse:
        limits = TIER_LIMITS[tier]
        payment_id = f"pay_{uuid.uuid4().hex[:8]}"

        checkout_url: str | None = None
        if PaymentService.is_mock():
            # Local dev only — auto-confirms (PAYMENT_MOCK_ENABLED=1).
            checkout_url = None
        elif PaymentService.is_paddle_configured():
            checkout_url = _create_paddle_checkout(
                payment_id, session_id, tier, return_params=return_params,
            )
        else:
            # Neither mock nor Paddle configured — refuse to create an order
            # rather than silently handing out premium. The user gets a clear
            # error telling them exactly which env vars are missing.
            raise RuntimeError(
                "Payment not configured: set PADDLE_API_KEY, "
                "PADDLE_CLIENT_TOKEN, PADDLE_WEBHOOK_SECRET, "
                "PADDLE_PRICE_STANDARD_ID and "
                "PADDLE_PRICE_PREMIUM_ID (or PAYMENT_MOCK_ENABLED=1 for local "
                "dev). See server/payment.py for the full Paddle setup."
            )

        record = PaymentResponse(
            payment_id=payment_id,
            session_id=session_id,
            tier=tier,
            status=PaymentStatus.pending,
            checkout_url=checkout_url,
            amount_cents=limits["price_cents"],
            created_at=storage.utcnow(),
        )
        _payments[payment_id] = record
        storage.save_payment(
            payment_id, session_id, tier.value, PaymentStatus.pending.value,
            limits["price_cents"], record.created_at,
        )

        # Direct async callers can schedule immediately. HTTP routes create the
        # order in a worker thread and schedule again after returning to their
        # event loop; this call is therefore a no-op for those routes.
        PaymentService.schedule_mock_confirmation(payment_id)

        return record

    @staticmethod
    def get_payment(payment_id: str) -> PaymentResponse | None:
        return _payments.get(payment_id)

    @staticmethod
    def get_payment_for_session(session_id: str) -> PaymentResponse | None:
        for p in _payments.values():
            if p.session_id == session_id and p.status == PaymentStatus.paid:
                return p
        return None

    @staticmethod
    def apply_paid_webhook(
        payment_id: str,
        amount_cents: int | None = None,
        provider_transaction_id: str | None = None,
                           ) -> PaymentResponse | None:
        """The ONLY method that marks a payment paid. Called from the
        signature-verified Paddle webhook. Idempotent across retries.

        amount_cents is accepted for API parity but NOT enforced for Paddle
        (see module docstring: price is pinned by price_id server-side, so
        there is no tamper vector the way WeChat's client-supplied callback
        had). A future non-Paddle gateway that does carry a client amount can
        still pass it here to enforce.
        """
        record = _payments.get(payment_id)
        if not record:
            return None
        if amount_cents is not None and amount_cents != record.amount_cents:
            # Kept for callers that want strict amount checks. The Paddle
            # webhook passes None, so this branch is not hit in production.
            raise ValueError(
                f"Amount mismatch: expected {record.amount_cents}, "
                f"got {amount_cents}"
            )
        record.status = PaymentStatus.paid
        storage.mark_payment_paid(payment_id, provider_transaction_id)
        return record

    @staticmethod
    def apply_refunded_webhook(
        payment_id: str | None = None,
        provider_transaction_id: str | None = None,
    ) -> PaymentResponse | None:
        """Mark a payment refunded from an approved Paddle refund adjustment.

        Paddle refund adjustments identify the related transaction. We prefer
        our own payment_id when present, but fall back to the stored Paddle
        transaction id captured by the paid webhook.
        """
        record: PaymentResponse | None = None
        if payment_id:
            record = _payments.get(payment_id)
            if record is None:
                row = storage.load_payment_row(payment_id)
                if row is not None:
                    record = _payment_response_from_row(row)
                    _payments[record.payment_id] = record
        elif provider_transaction_id:
            row = storage.load_payment_row_by_transaction_id(provider_transaction_id)
            if row is not None:
                record = _payment_response_from_row(row)
                _payments[record.payment_id] = record

        if record is None:
            return None
        record.status = PaymentStatus.refunded
        storage.mark_payment_refunded(record.payment_id)
        return record

    @staticmethod
    def get_tier_limits(tier: PricingTier) -> dict:
        return TIER_LIMITS[tier]


async def _mock_auto_confirm(payment_id: str, delay: int = 5):
    """Mock only: auto-confirm after delay. No effect unless mock is on.

    Goes through the SAME tier-upgrade path as the real webhook
    (queue.apply_payment_tier_upgrade) so dev mock behaves like production.
    """
    await asyncio.sleep(delay)
    if PaymentService.is_mock():
        # Lazy import to avoid a module-load cycle (job_queue imports payment).
        from .job_queue import queue
        queue.apply_payment_tier_upgrade(payment_id)


# ── Paddle Billing ──────────────────────────────────────

def _paddle_price_id_for_tier(tier: PricingTier) -> str | None:
    """Map a tier to its configured Paddle price_id (or None if unset)."""
    if tier == PricingTier.standard:
        return settings.paddle_price_standard_id or None
    if tier == PricingTier.premium:
        return settings.paddle_price_premium_id or None
    return None


def _paddle_api_base() -> str:
    """Paddle API base URL per environment."""
    env = (settings.paddle_environment or "sandbox").lower()
    return (
        "https://api.paddle.com"
        if env == "production"
        else "https://sandbox-api.paddle.com"
    )


def _create_paddle_checkout(
    payment_id: str,
    session_id: str,
    tier: PricingTier,
    return_params: dict[str, str] | None = None,
) -> str:
    """Create a Paddle transaction and return its approved payment-link URL.

    The order is bound to our internal payment record via ``custom_data`` so
    the webhook can route it back to the right session. Built with stdlib only
    — Paddle signs webhooks with HMAC-SHA256 (not RSA), so unlike WeChat there
    is no ``cryptography`` dependency.

    The expected Paddle Billing response shape is:
        {"data": {"id": "txn_...", "status": "ready",
                  "checkout": {"url": "https://checkout.paddle.com/..."}}}
    We validate that shape and fall back to a regex scan of the JSON only to
    guard against minor drift in Paddle's field naming.
    """
    price_id = _paddle_price_id_for_tier(tier)
    if not price_id:
        raise RuntimeError(
            f"No Paddle price_id configured for tier {tier.value}"
        )

    return_url = settings.paddle_return_url
    if return_params:
        separator = "&" if urllib.parse.urlparse(return_url).query else "?"
        return_url = return_url + separator + urllib.parse.urlencode({
            **return_params,
            "order": payment_id,
        })
    body = json.dumps({
        "items": [{"price_id": price_id, "quantity": 1}],
        "currency_code": "USD",
        "custom_data": {"payment_id": payment_id, "session_id": session_id},
        "checkout": {"url": return_url},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{_paddle_api_base()}/transactions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.paddle_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(
            f"Paddle checkout request failed ({exc.code}): {detail}"
        ) from exc

    # Defensive response-shape validation. The transaction id and status are
    # sanity checks; the checkout URL is the value we actually need.
    if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
        raise RuntimeError(
            "Paddle response missing expected 'data' object: "
            f"{json.dumps(data)[:500]}"
        )
    txn = data["data"]
    txn_id = txn.get("id")
    txn_status = txn.get("status")
    if not isinstance(txn_id, str) or not txn_id.startswith("txn_"):
        raise RuntimeError(
            f"Paddle response missing valid transaction id: {json.dumps(data)[:500]}"
        )
    if not isinstance(txn_status, str):
        raise RuntimeError(
            f"Paddle response missing transaction status: {json.dumps(data)[:500]}"
        )

    checkout = _extract_checkout_url(data)
    if not checkout:
        raise RuntimeError(
            "Paddle response had no checkout URL; inspect the response shape "
            f"(got: {json.dumps(data)[:500]})."
        )
    return checkout


def _extract_checkout_url(data: dict) -> str | None:
    """Pull the approved checkout URL out of a Paddle transaction response.

    Primary path: ``data.checkout.url`` (Paddle Billing). Fallback: scan for a
    checkout.paddle.com URL anywhere in the JSON — defends against minor shape
    drift. The returned URL is validated to belong to Paddle's checkout domain.
    """
    top = data.get("data") if isinstance(data, dict) else None
    candidates: list[str] = []
    if isinstance(top, dict):
        checkout = top.get("checkout")
        if isinstance(checkout, dict) and isinstance(checkout.get("url"), str):
            candidates.append(checkout["url"])
        if isinstance(top.get("checkout_url"), str):
            candidates.append(top["checkout_url"])

    if not candidates:
        text = json.dumps(data)
        candidates.extend(re.findall(r"https://[^\s\"']+", text))

    for url in candidates:
        if _is_paddle_checkout_url(url):
            return url
    return None


def _is_paddle_checkout_url(url: str) -> bool:
    """Accept Paddle-hosted links or our configured approved Paddle.js page."""
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {
        "checkout.paddle.com",
        "sandbox-checkout.paddle.com",
        "pay.paddle.io",
        "sandbox.pay.paddle.io",
    }:
        return True

    expected = urllib.parse.urlparse(settings.paddle_return_url)
    query = urllib.parse.parse_qs(parsed.query)
    transaction_ids = query.get("_ptxn", [])
    return (
        expected.scheme == "https"
        and host == (expected.hostname or "").lower()
        and parsed.path.rstrip("/") == expected.path.rstrip("/")
        and any(value.startswith("txn_") for value in transaction_ids)
    )


def check_tier_permission(session, feature: str) -> bool:
    """Check if the session's current tier allows a feature.

    feature: "id_photo", "bg_replace", "hd_download", "revise", "multi_style"
    """
    limits = TIER_LIMITS[session.tier]

    if feature == "id_photo":
        return limits["allow_id_photo"]
    elif feature == "bg_replace":
        return limits["allow_bg_replace"]
    elif feature == "hd_download":
        return limits["allow_hd_download"]
    elif feature == "revise":
        return session.revisions_used < limits["max_revisions"]
    elif feature == "multi_style":
        # For multi-style, check max_styles > 1
        return limits["max_styles"] > 1
    return False
