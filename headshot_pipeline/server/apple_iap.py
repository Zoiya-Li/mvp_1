"""StoreKit 2 signed-transaction and notification verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import settings


class AppleIAPError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedAppleTransaction:
    transaction_id: str
    original_transaction_id: str | None
    product_id: str
    bundle_id: str
    environment: str
    purchased_at: datetime | None
    revoked_at: datetime | None


def _datetime_from_millis(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


class AppleIAPVerifier:
    def __init__(self) -> None:
        self._verifier = None
        self._signature: tuple[Any, ...] | None = None

    @staticmethod
    def configuration_errors() -> list[str]:
        errors = []
        if not settings.apple_bundle_id:
            errors.append("APPLE_BUNDLE_ID is missing")
        if not settings.apple_iap_product_id:
            errors.append("APPLE_IAP_PRODUCT_ID is missing")
        cert_dir = Path(settings.apple_root_cert_dir)
        if not cert_dir.exists() or not list(cert_dir.glob("*.cer")):
            errors.append("Apple root certificates are missing")
        if settings.apple_iap_environment == "production" and settings.apple_app_id is None:
            errors.append("APPLE_APP_ID is missing")
        return errors

    def _get_verifier(self):
        errors = self.configuration_errors()
        if errors:
            raise AppleIAPError("; ".join(errors))
        try:
            from appstoreserverlibrary.models.Environment import Environment
            from appstoreserverlibrary.signed_data_verifier import SignedDataVerifier
        except ImportError as exc:
            raise AppleIAPError("app-store-server-library is not installed") from exc
        cert_paths = sorted(Path(settings.apple_root_cert_dir).glob("*.cer"))
        signature = (
            tuple((str(path), path.stat().st_mtime_ns) for path in cert_paths),
            settings.apple_iap_environment,
            settings.apple_bundle_id,
            settings.apple_app_id,
        )
        if self._verifier is None or signature != self._signature:
            environment = (
                Environment.PRODUCTION
                if settings.apple_iap_environment == "production"
                else Environment.SANDBOX
            )
            self._verifier = SignedDataVerifier(
                [path.read_bytes() for path in cert_paths],
                True,
                environment,
                settings.apple_bundle_id,
                settings.apple_app_id,
            )
            self._signature = signature
        return self._verifier

    def verify_transaction(self, signed_transaction: str) -> VerifiedAppleTransaction:
        if not signed_transaction or len(signed_transaction) > 65_536:
            raise AppleIAPError("Invalid signed StoreKit transaction")
        try:
            decoded = self._get_verifier().verify_and_decode_signed_transaction(
                signed_transaction
            )
        except AppleIAPError:
            raise
        except Exception as exc:
            raise AppleIAPError("StoreKit transaction verification failed") from exc
        transaction_id = str(getattr(decoded, "transactionId", "") or "")
        product_id = str(getattr(decoded, "productId", "") or "")
        if not transaction_id or not product_id:
            raise AppleIAPError("StoreKit transaction is missing required fields")
        return VerifiedAppleTransaction(
            transaction_id=transaction_id,
            original_transaction_id=(
                str(getattr(decoded, "originalTransactionId", "") or "") or None
            ),
            product_id=product_id,
            bundle_id=str(getattr(decoded, "bundleId", "") or ""),
            environment=_enum_value(getattr(decoded, "environment", "")),
            purchased_at=_datetime_from_millis(getattr(decoded, "purchaseDate", None)),
            revoked_at=_datetime_from_millis(getattr(decoded, "revocationDate", None)),
        )

    def verify_notification(self, signed_payload: str) -> tuple[str, VerifiedAppleTransaction | None]:
        if not signed_payload or len(signed_payload) > 131_072:
            raise AppleIAPError("Invalid App Store notification")
        try:
            notification = self._get_verifier().verify_and_decode_notification(signed_payload)
        except AppleIAPError:
            raise
        except Exception as exc:
            raise AppleIAPError("App Store notification verification failed") from exc
        notification_type = _enum_value(getattr(notification, "notificationType", ""))
        data = getattr(notification, "data", None)
        signed_transaction = getattr(data, "signedTransactionInfo", None) if data else None
        transaction = self.verify_transaction(signed_transaction) if signed_transaction else None
        return notification_type, transaction


apple_iap_verifier = AppleIAPVerifier()
