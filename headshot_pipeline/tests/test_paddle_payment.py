"""Paddle payment regression tests (Task #65) — NO network, NO Chrome.

Focuses on the security-critical HMAC verification (the ONLY tier-upgrade gate)
and the pure helpers. The webhook route is thin glue over these; it is covered
by the signature tests below plus the manual sandbox run at deploy.

Covers:
  verify_paddle_signature
    - accepts a correctly signed body
    - rejects: wrong secret, tampered body, stale timestamp, malformed header,
      missing ts/h1, empty inputs
  _extract_checkout_url
    - primary path data.checkout.url
    - fallback scan for a checkout.paddle.com URL
    - returns None when absent
  _paddle_api_base — sandbox vs production
  PaymentService.is_paddle_configured — False with empty default env
  TIER_LIMITS pricing — $5 Standard / $10 Pro (USD), English labels

Run:  python -m pytest headshot_pipeline/tests/test_paddle_payment.py -q
  or:  python headshot_pipeline/tests/test_paddle_payment.py
"""

from __future__ import annotations

import hashlib
import hmac
import sys
import time
from pathlib import Path

# Make the package importable whether run from mvp_1/ or the pipeline dir.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.router_payment import (  # noqa: E402
    PADDLE_WEBHOOK_MAX_AGE_SECONDS,
    verify_paddle_signature,
)
from server import payment as payment_mod, storage  # noqa: E402
from server.config import settings  # noqa: E402
from server.payment import (  # noqa: E402
    PaymentService,
    _extract_checkout_url,
    _paddle_api_base,
)
from server.models import PaymentStatus, PricingTier, TIER_LIMITS  # noqa: E402


SECRET = "whsec_test_signing_key_0123456789"


def _sign(body: bytes, secret: str, ts: int | None = None) -> str:
    """Reproduce Paddle's exact signing: ``ts:raw_body`` HMAC-SHA256 hex."""
    ts = int(time.time()) if ts is None else ts
    signed = f"{ts}:".encode("utf-8") + body
    sig = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    return f"ts={ts};h1={sig}"


BODY = (
    b'{"event_type":"transaction.completed","data":{"custom_data":'
    b'{"payment_id":"pay_abc12345","session_id":"sess_xyz"}}}'
)


# ── verify_paddle_signature ────────────────────────────

def test_signature_accepts_valid():
    header = _sign(BODY, SECRET)
    assert verify_paddle_signature(BODY, header, SECRET) is True


def test_signature_rejects_wrong_secret():
    header = _sign(BODY, SECRET)
    assert verify_paddle_signature(BODY, header, "whsec_DIFFERENT_key") is False


def test_signature_rejects_tampered_body():
    header = _sign(BODY, SECRET)
    tampered = BODY.replace(b"pay_abc12345", b"pay_zzzzzzzz")
    assert verify_paddle_signature(tampered, header, SECRET) is False


def test_signature_rejects_stale_timestamp():
    old_ts = int(time.time()) - PADDLE_WEBHOOK_MAX_AGE_SECONDS - 60
    header = _sign(BODY, SECRET, ts=old_ts)
    assert verify_paddle_signature(BODY, header, SECRET) is False


def test_signature_accepts_fresh_within_window():
    # Exactly at the edge minus a second should still pass.
    edge_ts = int(time.time()) - PADDLE_WEBHOOK_MAX_AGE_SECONDS + 5
    header = _sign(BODY, SECRET, ts=edge_ts)
    assert verify_paddle_signature(BODY, header, SECRET) is True


def test_signature_rejects_future_timestamp():
    future_ts = int(time.time()) + PADDLE_WEBHOOK_MAX_AGE_SECONDS + 60
    header = _sign(BODY, SECRET, ts=future_ts)
    assert verify_paddle_signature(BODY, header, SECRET) is False


def test_signature_rejects_malformed_header():
    good = _sign(BODY, SECRET)
    assert verify_paddle_signature(BODY, "not-a-real-signature", SECRET) is False
    # Missing h1
    assert verify_paddle_signature(BODY, "ts=12345", SECRET) is False
    # Missing ts
    assert verify_paddle_signature(BODY, "h1=abc", SECRET) is False
    # Non-numeric ts
    assert verify_paddle_signature(
        BODY, "ts=notanint;h1=abc", SECRET
    ) is False
    # Garbage where good was expected
    assert good  # sanity


def test_signature_rejects_empty_inputs():
    # Empty header → no ts/h1 to parse → reject.
    assert verify_paddle_signature(BODY, "", SECRET) is False
    # Empty secret → the `not secret` guard rejects before any HMAC math.
    assert verify_paddle_signature(BODY, _sign(BODY, SECRET), "") is False
    # NOTE: a correctly-signed EMPTY body is mathematically valid HMAC (anyone
    # holding the secret can sign empty bytes). That is correct behavior, not a
    # hole — the route still 400s on ``json.loads(b"")`` before any tier
    # upgrade, so an empty body can never raise a tier regardless of signature.


# ── _extract_checkout_url ──────────────────────────────

def test_extract_primary_path():
    data = {"data": {"checkout": {"url": "https://checkout.paddle.com/txn_123"}}}
    assert _extract_checkout_url(data) == "https://checkout.paddle.com/txn_123"


def test_extract_fallback_scan():
    # Shape drift: no nested checkout.url, but a valid URL lives somewhere.
    data = {"data": {"weird": "https://checkout.paddle.com/abc?x=1"}}
    assert _extract_checkout_url(data) == "https://checkout.paddle.com/abc?x=1"


def test_extract_returns_none_when_absent():
    assert _extract_checkout_url({"data": {"checkout": {}}}) is None
    assert _extract_checkout_url({"data": {}}) is None
    assert _extract_checkout_url({}) is None
    assert _extract_checkout_url({"data": {"checkout": {"url": 123}}}) is None


# ── _paddle_api_base ───────────────────────────────────

def test_api_base_environment(monkeypatch):
    from server import config

    monkeypatch.setattr(config.settings, "paddle_environment", "sandbox")
    assert _paddle_api_base() == "https://sandbox-api.paddle.com"
    monkeypatch.setattr(config.settings, "paddle_environment", "production")
    assert _paddle_api_base() == "https://api.paddle.com"
    # Unknown env falls back to sandbox (safe default — never charge by accident).
    monkeypatch.setattr(config.settings, "paddle_environment", "garbage")
    assert _paddle_api_base() == "https://sandbox-api.paddle.com"


# ── config default + pricing ───────────────────────────

def test_paddle_not_configured_by_default():
    # With the placeholder env (no real keys), the service must refuse to mint
    # orders — it must NEVER silently hand out premium.
    assert PaymentService.is_paddle_configured() is False


def test_pricing_is_usd_and_english_labels():
    assert TIER_LIMITS[PricingTier.standard]["price_cents"] == 500   # $5
    assert TIER_LIMITS[PricingTier.premium]["price_cents"] == 1000   # $10
    assert TIER_LIMITS[PricingTier.free]["price_cents"] == 0
    assert TIER_LIMITS[PricingTier.standard]["label"] == "Standard"
    assert TIER_LIMITS[PricingTier.premium]["label"] == "Pro"
    assert TIER_LIMITS[PricingTier.standard]["max_styles"] == 2
    assert TIER_LIMITS[PricingTier.premium]["max_styles"] == 2


def test_payment_refund_persists_by_provider_transaction_id(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(storage, "_DB_PATH", None)
    payment_mod._payments.clear()
    storage.init_db()
    created_at = storage.utcnow()
    storage.save_payment(
        "pay_refund",
        "s_refund",
        PricingTier.standard.value,
        PaymentStatus.pending.value,
        TIER_LIMITS[PricingTier.standard]["price_cents"],
        created_at,
    )
    payment_mod._load_from_db()

    paid = PaymentService.apply_paid_webhook(
        "pay_refund",
        provider_transaction_id="txn_refund_123",
    )
    assert paid is not None
    assert paid.status == PaymentStatus.paid
    row = storage.load_payment_row("pay_refund")
    assert row["status"] == "paid"
    assert row["provider_transaction_id"] == "txn_refund_123"

    refunded = PaymentService.apply_refunded_webhook(
        provider_transaction_id="txn_refund_123"
    )
    assert refunded is not None
    assert refunded.status == PaymentStatus.refunded
    assert storage.load_payment_row("pay_refund")["status"] == "refunded"


if __name__ == "__main__":
    # Allow running without pytest: execute every test_* function.
    import inspect
    import tempfile

    monkeypatch = type(  # minimal shim for the one monkeypatch test
        "MP", (), {"setattr": staticmethod(lambda obj, name, value: setattr(obj, name, value))}
    )()
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                params = inspect.signature(fn).parameters
                kwargs = {}
                tmp_ctx = None
                if "monkeypatch" in params:
                    kwargs["monkeypatch"] = monkeypatch
                if "tmp_path" in params:
                    tmp_ctx = tempfile.TemporaryDirectory()
                    kwargs["tmp_path"] = Path(tmp_ctx.name)
                fn(**kwargs)
                print(f"PASS {name}")
                if tmp_ctx:
                    tmp_ctx.cleanup()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
