"""SQLite persistence layer for sessions and payments.

Why this exists (code review finding):
  All state lived in in-memory dicts. A restart (Mac Mini reboot, launchd
  reload) wiped every paid order and every session, leaving orphaned files on
  disk and users who paid with nothing to show for it.

Design:
  - SQLite is the durable source of truth for *tier*, *payment status*, and
    *session existence*. In-memory job/image state stays in the JobQueue (it is
    ephemeral by nature — jobs can be re-run, generated images live on disk).
  - Writes are append/upsert and idempotent. Reads are cheap (single row).
  - The DB lives under ``data_dir/shanxiang.db``.

This module only persists what matters across restarts:
  - session_id, owner_token, style, gender, tier, max_revisions, created_at,
    status, payment_id  (the sessions table)
  - payment records (id, session, tier, status, amount, created_at, paid_at)
  - generated-image metadata (id, session, prompt_id, turn, parent, operation,
    resemblance history, created_at)  (the generated_images table)
  - generation job outcomes, including failed shots that never reach the gallery
    (the generation_events table)

Generated image *pixels* live on disk under output_dir/session_id/. Their
*metadata* (resemblance score history, turn, parent links) is persisted here so
a backend restart can fully rehydrate a session — auth, paid tier, AND the
generated-image gallery with its scores — via JobQueue._hydrate_session(). Tier
and payment status are the revenue-critical fields that must survive a restart.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import settings


_DB_PATH: Path | None = None
_lock = threading.Lock()


def _db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = settings.data_dir / "shanxiang.db"
    return _DB_PATH


def utcnow() -> datetime:
    """Timezone-aware UTC now (replaces naive datetime.now() everywhere)."""
    return datetime.now(timezone.utc)


@contextmanager
def get_conn():
    """Yield a SQLite connection. Thread-safe via a per-call connection."""
    conn = sqlite3.connect(str(_db_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist. Called at startup."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id     TEXT PRIMARY KEY,
                owner_token    TEXT NOT NULL,
                style          TEXT NOT NULL,
                gender         TEXT NOT NULL,
                tier           TEXT NOT NULL DEFAULT 'free',
                max_revisions  INTEGER NOT NULL DEFAULT 1,
                status         TEXT NOT NULL DEFAULT 'created',
                payment_id     TEXT,
                consent_json   TEXT,
                created_at     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS payments (
                payment_id   TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                tier         TEXT NOT NULL,
                status       TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                created_at   TEXT NOT NULL,
                paid_at      TEXT,
                provider_transaction_id TEXT,
                refunded_at  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_payments_session
                ON payments(session_id);

            CREATE TABLE IF NOT EXISTS generated_images (
                image_id         TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                prompt_id        TEXT,
                turn             INTEGER NOT NULL DEFAULT 1,
                revised_image_id TEXT,
                parent_image_id  TEXT,
                operation        TEXT,
                resemblance_json TEXT,
                created_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_images_session
                ON generated_images(session_id);

            CREATE TABLE IF NOT EXISTS generation_events (
                event_id        TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                job_id          TEXT NOT NULL,
                prompt_id       TEXT,
                shot_spec_json  TEXT,
                metadata_json   TEXT,
                status          TEXT NOT NULL,
                failure_reason  TEXT,
                error           TEXT,
                result_image_id TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_generation_events_session
                ON generation_events(session_id);

            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                image_id    TEXT NOT NULL,
                event       TEXT NOT NULL,
                reason      TEXT,
                score       INTEGER,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_session
                ON user_feedback(session_id);

            CREATE INDEX IF NOT EXISTS idx_feedback_image
                ON user_feedback(session_id, image_id);
            """
        )
        _ensure_column(conn, "generation_events", "metadata_json", "metadata_json TEXT")
        _ensure_column(conn, "sessions", "consent_json", "consent_json TEXT")
        _ensure_column(conn, "sessions", "hero_preview_image_id", "hero_preview_image_id TEXT")
        _ensure_column(conn, "sessions", "unlocked", "unlocked INTEGER DEFAULT 0")
        _ensure_column(
            conn,
            "payments",
            "provider_transaction_id",
            "provider_transaction_id TEXT",
        )
        _ensure_column(conn, "payments", "refunded_at", "refunded_at TEXT")

    # API v2 has an additive schema so v1 sessions and paid orders continue to
    # work during the portrait-platform migration.
    from .portrait_catalog import ensure_theme_catalog
    from .portrait_storage import init_portrait_schema

    init_portrait_schema()
    ensure_theme_catalog()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str):
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


# ── Sessions ────────────────────────────────────────────

def save_session(
    session_id: str,
    owner_token: str,
    style: str,
    gender: str,
    created_at: datetime,
):
    with _lock, get_conn() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO sessions
               (session_id, owner_token, style, gender, tier, max_revisions,
                status, payment_id, created_at)
               VALUES (?, ?, ?, ?, 'free', 1, 'created', NULL, ?)""",
            (session_id, owner_token, style, gender, created_at.isoformat()),
        )


def update_session_tier(session_id: str, tier: str, max_revisions: int,
                        payment_id: str | None):
    with _lock, get_conn() as conn:
        conn.execute(
            """UPDATE sessions SET tier=?, max_revisions=?, payment_id=?
               WHERE session_id=?""",
            (tier, max_revisions, payment_id, session_id),
        )


def update_session_hero_preview(session_id: str, hero_preview_image_id: str | None,
                                unlocked: bool | None = None):
    with _lock, get_conn() as conn:
        if unlocked is not None:
            conn.execute(
                """UPDATE sessions SET hero_preview_image_id=?, unlocked=?
                   WHERE session_id=?""",
                (hero_preview_image_id, 1 if unlocked else 0, session_id),
            )
        else:
            conn.execute(
                "UPDATE sessions SET hero_preview_image_id=? WHERE session_id=?",
                (hero_preview_image_id, session_id),
            )


def update_session_status(session_id: str, status: str):
    with _lock, get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET status=? WHERE session_id=?",
            (status, session_id),
        )


def update_session_consent(session_id: str, consent: dict):
    with _lock, get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET consent_json=? WHERE session_id=?",
            (json.dumps(consent, ensure_ascii=False), session_id),
        )


def delete_session_row(session_id: str):
    with _lock, get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))


def list_stale_sessions(before_iso: str) -> list[sqlite3.Row]:
    """Sessions created before ``before_iso`` (for the 7-day cleanup sweep)."""
    with get_conn() as conn:
        return list(
            conn.execute(
                "SELECT session_id FROM sessions WHERE created_at < ?",
                (before_iso,),
            )
        )


def load_session_row(session_id: str) -> sqlite3.Row | None:
    """Return the full sessions row, or None. Used to rehydrate after a restart."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()


# ── Generated images ────────────────────────────────────


def save_generated_image(
    image_id: str,
    session_id: str,
    prompt_id: str | None,
    turn: int,
    revised_image_id: str | None,
    parent_image_id: str | None,
    operation: str | None,
    resemblance: dict | None,
    created_at: datetime,
):
    """Persist generated-image metadata so the gallery (with resemblance history)
    survives a backend restart. Pixels live on disk; this stores the metadata."""
    with _lock, get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO generated_images
               (image_id, session_id, prompt_id, turn, revised_image_id,
                parent_image_id, operation, resemblance_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_id, session_id, prompt_id, turn, revised_image_id,
                parent_image_id, operation,
                json.dumps(resemblance, ensure_ascii=False) if resemblance else None,
                created_at.isoformat(),
            ),
        )


def load_generated_images(session_id: str) -> list[sqlite3.Row]:
    """All generated-image metadata rows for a session, oldest first."""
    with get_conn() as conn:
        return list(
            conn.execute(
                "SELECT * FROM generated_images WHERE session_id=? "
                "ORDER BY created_at",
                (session_id,),
            )
        )


def delete_session_images(session_id: str):
    """Remove a session's image-metadata rows (cleanup on session delete)."""
    with _lock, get_conn() as conn:
        conn.execute(
            "DELETE FROM generated_images WHERE session_id=?", (session_id,)
        )


def mark_generated_image_operation(
    session_id: str,
    image_id: str,
    operation: str | None,
) -> None:
    """Persist a lifecycle marker without deleting feedback-linked metadata."""
    with _lock, get_conn() as conn:
        conn.execute(
            """UPDATE generated_images SET operation=?
               WHERE session_id=? AND image_id=?""",
            (operation, session_id, image_id),
        )


# ── Generation events ───────────────────────────────────

def save_generation_event(
    event_id: str,
    session_id: str,
    job_id: str,
    prompt_id: str | None,
    shot_spec: dict | None,
    status: str,
    failure_reason: str | None,
    error: str | None,
    result_image_id: str | None,
    created_at: datetime,
    completed_at: datetime | None,
    metadata: dict | None = None,
):
    """Persist a generation job outcome, including failed non-gallery shots."""
    with _lock, get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO generation_events
               (event_id, session_id, job_id, prompt_id, shot_spec_json,
                metadata_json, status, failure_reason, error, result_image_id,
                created_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                session_id,
                job_id,
                prompt_id,
                json.dumps(shot_spec, ensure_ascii=False) if shot_spec else None,
                json.dumps(metadata, ensure_ascii=False) if metadata else None,
                status,
                failure_reason,
                error,
                result_image_id,
                created_at.isoformat(),
                completed_at.isoformat() if completed_at else None,
            ),
        )


def load_generation_events(session_id: str) -> list[sqlite3.Row]:
    """All generation job outcomes for a session, oldest first."""
    with get_conn() as conn:
        return list(
            conn.execute(
                "SELECT * FROM generation_events WHERE session_id=? "
                "ORDER BY created_at",
                (session_id,),
            )
        )


def fail_interrupted_generation_events(
    session_id: str,
    *,
    completed_at: datetime,
) -> int:
    """Close jobs left processing when the single worker process restarted."""
    with _lock, get_conn() as conn:
        cursor = conn.execute(
            """UPDATE generation_events
               SET status='failed',
                   failure_reason='worker_interrupted',
                   error='Generation worker restarted before this shot completed',
                   completed_at=?
               WHERE session_id=? AND status='processing'""",
            (completed_at.isoformat(), session_id),
        )
        return int(cursor.rowcount)


def delete_session_generation_events(session_id: str):
    with _lock, get_conn() as conn:
        conn.execute(
            "DELETE FROM generation_events WHERE session_id=?", (session_id,)
        )


# ── User feedback ───────────────────────────────────────

def save_user_feedback(
    feedback_id: str,
    session_id: str,
    image_id: str,
    event: str,
    reason: str | None,
    score: int | None,
    created_at: datetime,
):
    with _lock, get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO user_feedback
               (feedback_id, session_id, image_id, event, reason, score,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                feedback_id,
                session_id,
                image_id,
                event,
                reason,
                score,
                created_at.isoformat(),
            ),
        )


def load_user_feedback(session_id: str) -> list[sqlite3.Row]:
    with get_conn() as conn:
        return list(
            conn.execute(
                "SELECT * FROM user_feedback WHERE session_id=? ORDER BY created_at",
                (session_id,),
            )
        )


def delete_session_feedback(session_id: str):
    with _lock, get_conn() as conn:
        conn.execute("DELETE FROM user_feedback WHERE session_id=?", (session_id,))


# ── Payments ────────────────────────────────────────────

def save_payment(payment_id: str, session_id: str, tier: str,
                 status: str, amount_cents: int, created_at: datetime):
    with _lock, get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO payments
               (payment_id, session_id, tier, status, amount_cents, created_at,
                paid_at, provider_transaction_id, refunded_at)
               VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)""",
            (payment_id, session_id, tier, status, amount_cents,
             created_at.isoformat()),
        )


def mark_payment_paid(payment_id: str, provider_transaction_id: str | None = None):
    paid_at = utcnow().isoformat()
    with _lock, get_conn() as conn:
        conn.execute(
            """UPDATE payments
               SET status='paid',
                   paid_at=?,
                   provider_transaction_id=COALESCE(?, provider_transaction_id)
               WHERE payment_id=?""",
            (paid_at, provider_transaction_id, payment_id),
        )


def mark_payment_refunded(payment_id: str):
    refunded_at = utcnow().isoformat()
    with _lock, get_conn() as conn:
        conn.execute(
            "UPDATE payments SET status='refunded', refunded_at=? WHERE payment_id=?",
            (refunded_at, payment_id),
        )


def load_payment_row(payment_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM payments WHERE payment_id=?", (payment_id,)
        ).fetchone()


def load_payment_row_by_transaction_id(
    provider_transaction_id: str,
) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM payments WHERE provider_transaction_id=?",
            (provider_transaction_id,),
        ).fetchone()
