"""Sign in with Apple identity-token verification."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from typing import Any

import httpx
import jwt

from .config import settings


class AppleIdentityError(ValueError):
    pass


class AppleIdentityVerifier:
    def __init__(self) -> None:
        self._jwks: dict[str, Any] | None = None
        self._jwks_expires_at = 0.0
        self._lock = threading.Lock()

    def _load_jwks(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._jwks is not None and now < self._jwks_expires_at:
                return self._jwks
            response = httpx.get(settings.apple_jwks_url, timeout=10.0)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload.get("keys"), list):
                raise AppleIdentityError("Apple identity keys are unavailable")
            self._jwks = payload
            self._jwks_expires_at = now + 6 * 3600
            return payload

    def verify(self, identity_token: str, raw_nonce: str) -> dict[str, Any]:
        if not identity_token or len(identity_token) > 16_384:
            raise AppleIdentityError("Invalid Apple identity token")
        if not raw_nonce or len(raw_nonce) > 512:
            raise AppleIdentityError("Invalid Apple sign-in nonce")
        try:
            header = jwt.get_unverified_header(identity_token)
        except jwt.PyJWTError as exc:
            raise AppleIdentityError("Malformed Apple identity token") from exc
        if header.get("alg") != "RS256" or not header.get("kid"):
            raise AppleIdentityError("Unsupported Apple identity signature")
        key_data = next(
            (key for key in self._load_jwks()["keys"] if key.get("kid") == header["kid"]),
            None,
        )
        if key_data is None:
            # Apple may have just rotated keys. Force one refresh.
            with self._lock:
                self._jwks_expires_at = 0
            key_data = next(
                (key for key in self._load_jwks()["keys"] if key.get("kid") == header["kid"]),
                None,
            )
        if key_data is None:
            raise AppleIdentityError("Unknown Apple identity signing key")
        try:
            claims = jwt.decode(
                identity_token,
                jwt.PyJWK.from_dict(key_data).key,
                algorithms=["RS256"],
                audience=settings.apple_client_id,
                issuer=settings.apple_identity_issuer,
                options={"require": ["exp", "iat", "iss", "aud", "sub", "nonce"]},
            )
        except jwt.PyJWTError as exc:
            raise AppleIdentityError("Apple identity token verification failed") from exc
        expected_nonce = hashlib.sha256(raw_nonce.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(str(claims.get("nonce", "")), expected_nonce):
            raise AppleIdentityError("Apple sign-in nonce mismatch")
        if not claims.get("sub"):
            raise AppleIdentityError("Apple identity subject is missing")
        return claims


apple_identity_verifier = AppleIdentityVerifier()
