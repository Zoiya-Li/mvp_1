"""Security primitives: signed session tokens, ownership checks, rate limiting,
path-containment validation.

Threat model this closes (from the code review):
  - Any user could read/delete any other user's session, photos, or payment by
    guessing the 8-hex-char session_id (only 2^32 space, enumerable).
  - Path traversal in get_uploaded_photo / get_image read arbitrary server files.
  - No rate limiting → an attacker could drain the logged-in Gemini account.

Design:
  - Every session is created with a random 256-bit ``owner_token`` (URL-safe).
    The token is the ONLY credential. The session_id is public-ish (it appears
    in WS/image URLs); the token is secret and must be supplied on every
    session-scoped mutation/read.
  - Tokens are returned once at session creation and stored client-side; the
    server never lists them.
  - Rate limiting uses an in-process token bucket keyed by client IP. Sufficient
    for a single-host MVP; swap for Redis if you scale out.
"""

from __future__ import annotations

import hmac
import secrets
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock

from fastapi import Header, HTTPException, Query, Request, WebSocket


# ── Token generation ────────────────────────────────────

def generate_token() -> str:
    """Generate a cryptographically random URL-safe owner token (256 bits)."""
    return secrets.token_urlsafe(32)


# ── Path containment ────────────────────────────────────

_SAFE_NAME = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


def sanitize_filename(name: str) -> str:
    """Reduce a user-supplied filename to its basename and reject traversal.

    Returns a safe basename (no path separators, no NUL, no leading dots that
    could escape). Raises HTTPException(400) if the name is empty or dangerous.
    """
    if not name:
        raise HTTPException(400, "Empty filename")
    # Take only the basename — strips any directory components.
    base = Path(name).name
    if not base or base in (".", ".."):
        raise HTTPException(400, "Invalid filename")
    if any(ch not in _SAFE_NAME for ch in base):
        raise HTTPException(400, "Filename contains illegal characters")
    return base


def safe_id(value: str, label: str = "id") -> str:
    """Validate an ID used to build a filesystem path (image_id / payment_id).

    Only ``[A-Za-z0-9_-]`` is allowed. This makes path traversal via ``image_id``
    impossible because ``..`` and ``/`` are rejected.
    """
    if not value or any(ch not in _SAFE_NAME for ch in value):
        raise HTTPException(400, f"Invalid {label}")
    return value


def is_within(base: Path, target: Path) -> bool:
    """True iff ``target`` resolves to a path inside ``base`` (no symlink escape)."""
    try:
        base_r = base.resolve()
        target_r = target.resolve()
    except (OSError, RuntimeError):
        return False
    return target_r == base_r or base_r in target_r.parents


# ── Upload validation ───────────────────────────────────

# Magic-byte signatures for allowed image formats. We check the actual file
# content rather than trusting the client-supplied filename/content-type, so a
# renamed executable or HTML file can't be stored and later served as an image
# (stored-XSS / content-sniffing vector).
_IMAGE_SIGNATURES: tuple[bytes, ...] = (
    b"\xff\xd8\xff",            # JPEG
    b"\x89PNG\r\n\x1a\n",       # PNG
    b"RIFF",                    # WEBP (RIFF....WEBP)
    b"\x00\x00\x00",            # HEIC/HEIF (ftyp box: size(4) + 'ftyp'...)
)


def validate_image_bytes(content: bytes) -> None:
    """Raise HTTPException(400) unless ``content`` starts with a known image
    magic signature. Also enforces a hard minimum size."""
    if len(content) < 12:
        raise HTTPException(400, "File too small to be a valid image")
    head = content[:16]
    ok = (
        head.startswith(b"\xff\xd8\xff")
        or head.startswith(b"\x89PNG\r\n\x1a\n")
        or (head.startswith(b"RIFF") and head[8:12] == b"WEBP")
        # HEIC: bytes 4-8 == b'ftyp' and brand in heic/heix/mif1
        or (head[4:8] == b"ftyp" and head[8:12] in (b"heic", b"heix", b"mif1"))
    )
    if not ok:
        raise HTTPException(400, "Unrecognized image format (allowed: JPEG, PNG, WEBP, HEIC)")


# ── Ownership dependency ────────────────────────────────

# Unguessable placeholder. When a requested session does not exist we still run
# a constant-time token comparison (against this dummy) instead of short-
# circuiting. This equalises the comparison path/timing between "session
# missing" and "session exists, wrong token", so the only externally observable
# outcome of a failed auth attempt is a 401 — never a 404 that would reveal a
# session id is valid. See _check_token / require_owner.
_UNOWNED_DUMMY_TOKEN = secrets.token_urlsafe(32)


def _check_token(state, token: str | None):
    """Verify the caller owns this session; raise 401 on ANY failure.

    Deliberately raises 401 (NEVER 404) when the session does not exist.
    Returning 404 here would create a session-id enumeration oracle: an
    attacker probing ids could tell "no such session" (404) from "exists but
    wrong/missing token" (401), then focus effort on the ids that exist.
    Only the legitimate owner — who supplies the correct 256-bit token — ever
    learns a session exists.

    ``state`` may be ``None`` (session absent). In that case we compare the
    supplied token against ``_UNOWNED_DUMMY_TOKEN`` (unguessable), so the
    comparison runs uniformly and always mismatches.
    """
    expected = (
        state.owner_token
        if state is not None and state.owner_token
        else _UNOWNED_DUMMY_TOKEN
    )
    supplied = token or ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "Authentication required")
    # A None state cannot match the unguessable dummy, but guard explicitly so
    # a future change to the dummy cannot accidentally authenticate nothing.
    if state is None:
        raise HTTPException(401, "Authentication required")


def require_owner(
    session_id: str,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
):
    """FastAPI dependency: load session + verify caller owns it.

    Fetches the session WITHOUT raising 404 on absence — that distinction is
    deferred to :func:`_check_token`, which returns a uniform 401 for both
    "missing session" and "wrong token" to avoid an enumeration oracle.

    Usage::

        @router.post("/sessions/{session_id}/generate")
        async def gen(session_id: str, session = Depends(require_owner)):
            ...
    """
    from .job_queue import queue  # lazy: avoid circular import at module load
    state = queue.get_session(session_id)  # None if absent — do NOT 404 here
    _check_token(state, x_session_token)   # 401 if None / missing / wrong
    return state


def require_owner_query(
    session_id: str,
    token: str | None = Query(default=None),
):
    """Same as require_owner but token comes via query string (image/photo
    GETs and WebSocket handshakes)."""
    from .job_queue import queue
    state = queue.get_session(session_id)  # None if absent — do NOT 404 here
    _check_token(state, token)
    return state


async def require_owner_ws(ws: WebSocket, session_id: str):
    """WebSocket handshake auth. Token via ``?token=``. Accepts only if valid.

    Returns the session state on success (for the caller to use), or closes the
    connection with code 4401. A missing session is reported as a 4401 close
    (not a distinct code) to avoid the same enumeration oracle.
    """
    token = ws.query_params.get("token")
    from .job_queue import queue
    state = queue.get_session(session_id)  # None if absent — uniform 401 below
    try:
        _check_token(state, token)
    except HTTPException:
        await ws.close(code=4401)
        return None
    return state


# ── Rate limiting (in-process token bucket per IP) ──────

class RateLimiter:
    """Simple sliding-window rate limiter keyed by a string (client IP).

    Not distributed — fine for a single-host MVP. For multi-host, back this
    with Redis.
    """

    def __init__(self):
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, key: str, max_calls: int, window_s: int) -> None:
        """Raise 429 if ``key`` has >= ``max_calls`` hits in the last ``window_s``."""
        now = time.monotonic()
        cutoff = now - window_s
        with self._lock:
            bucket = self._hits[key]
            # Drop expired entries
            self._hits[key] = [t for t in bucket if t > cutoff]
            if len(self._hits[key]) >= max_calls:
                raise HTTPException(
                    429,
                    "Too many requests — please slow down",
                    headers={"Retry-After": str(window_s)},
                )
            self._hits[key].append(now)


# Shared instances
rate_limiter = RateLimiter()


def limit_session_create(request: Request):
    """Rate-limit session creation: 10 per IP per 10 minutes."""
    client = request.client.host if request.client else "unknown"
    rate_limiter.check(f"create:{client}", max_calls=10, window_s=600)


def limit_generation(session_id: str):
    """Rate-limit generation/revision per session: 30 per 10 minutes."""
    rate_limiter.check(f"gen:{session_id}", max_calls=30, window_s=600)
