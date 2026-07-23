"""Durable portrait-platform persistence built alongside the v1 tables.

SQLite remains the local/runtime store during the compatibility phase.  The
schema uses explicit ownership and immutable ledger entries so it can be moved
to PostgreSQL without changing the API contract.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Any

from . import storage


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def init_portrait_schema() -> None:
    with storage.get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS portrait_users (
                user_id       TEXT PRIMARY KEY,
                token_hash    TEXT NOT NULL UNIQUE,
                account_type  TEXT NOT NULL DEFAULT 'guest',
                display_name  TEXT,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portrait_identities (
                provider      TEXT NOT NULL,
                subject       TEXT NOT NULL,
                user_id       TEXT NOT NULL,
                email         TEXT,
                created_at    TEXT NOT NULL,
                PRIMARY KEY(provider, subject),
                FOREIGN KEY(user_id) REFERENCES portrait_users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_identities_user
                ON portrait_identities(user_id);

            CREATE TABLE IF NOT EXISTS portrait_themes (
                theme_id          TEXT PRIMARY KEY,
                slug              TEXT NOT NULL UNIQUE,
                title             TEXT NOT NULL,
                title_en          TEXT NOT NULL,
                tagline           TEXT NOT NULL,
                category          TEXT NOT NULL,
                cover_image       TEXT NOT NULL,
                preview_json      TEXT NOT NULL,
                use_cases_json    TEXT NOT NULL,
                source_style_key  TEXT NOT NULL UNIQUE,
                featured          INTEGER NOT NULL DEFAULT 0,
                sort_order        INTEGER NOT NULL DEFAULT 0,
                status            TEXT NOT NULL DEFAULT 'active',
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portrait_theme_versions (
                theme_version_id  TEXT PRIMARY KEY,
                theme_id          TEXT NOT NULL,
                version           INTEGER NOT NULL,
                blueprint_json    TEXT NOT NULL,
                status            TEXT NOT NULL DEFAULT 'active',
                created_at        TEXT NOT NULL,
                UNIQUE(theme_id, version),
                FOREIGN KEY(theme_id) REFERENCES portrait_themes(theme_id)
            );

            CREATE TABLE IF NOT EXISTS portrait_projects (
                project_id            TEXT PRIMARY KEY,
                user_id               TEXT NOT NULL,
                theme_id              TEXT,
                theme_version_id      TEXT,
                source                TEXT NOT NULL,
                status                TEXT NOT NULL,
                gender                TEXT NOT NULL,
                shared_recipe_id      TEXT,
                inspiration_asset_id  TEXT,
                inspiration_spec_json TEXT,
                hero_asset_id         TEXT,
                photo_set_id          TEXT,
                legacy_session_id     TEXT,
                preview_retries_used  INTEGER NOT NULL DEFAULT 0,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES portrait_users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_projects_user
                ON portrait_projects(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS portrait_assets (
                asset_id       TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                project_id     TEXT,
                asset_type     TEXT NOT NULL,
                storage_path   TEXT NOT NULL,
                mime_type      TEXT NOT NULL,
                metadata_json  TEXT NOT NULL DEFAULT '{}',
                created_at     TEXT NOT NULL,
                expires_at     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_assets_project
                ON portrait_assets(project_id, created_at);

            CREATE TABLE IF NOT EXISTS portrait_photo_sets (
                photo_set_id   TEXT PRIMARY KEY,
                project_id     TEXT NOT NULL,
                title          TEXT NOT NULL,
                status         TEXT NOT NULL,
                cover_asset_id TEXT,
                asset_ids_json TEXT NOT NULL DEFAULT '[]',
                created_at     TEXT NOT NULL,
                delivered_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS portrait_generation_runs (
                run_id          TEXT PRIMARY KEY,
                project_id      TEXT NOT NULL,
                run_type        TEXT NOT NULL,
                status          TEXT NOT NULL,
                blueprint_json  TEXT NOT NULL,
                cost_json       TEXT NOT NULL DEFAULT '{}',
                error_code      TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS portrait_ledger (
                entry_id       TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                amount         INTEGER NOT NULL,
                reason         TEXT NOT NULL,
                reference_id   TEXT,
                created_at     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_ledger_user
                ON portrait_ledger(user_id, created_at);

            CREATE TABLE IF NOT EXISTS portrait_orders (
                order_id                TEXT PRIMARY KEY,
                user_id                 TEXT NOT NULL,
                project_id              TEXT,
                provider                TEXT NOT NULL,
                provider_transaction_id TEXT,
                product_code            TEXT NOT NULL,
                amount_cents            INTEGER NOT NULL,
                currency                TEXT NOT NULL,
                status                  TEXT NOT NULL,
                created_at              TEXT NOT NULL,
                paid_at                 TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_portrait_orders_provider_tx
                ON portrait_orders(provider, provider_transaction_id)
                WHERE provider_transaction_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS portrait_apple_transactions (
                transaction_id          TEXT PRIMARY KEY,
                original_transaction_id TEXT,
                user_id                 TEXT NOT NULL,
                project_id              TEXT NOT NULL,
                product_id              TEXT NOT NULL,
                environment             TEXT NOT NULL,
                bundle_id               TEXT NOT NULL,
                status                  TEXT NOT NULL,
                signed_payload_hash     TEXT NOT NULL,
                purchased_at            TEXT,
                revoked_at              TEXT,
                created_at              TEXT NOT NULL,
                updated_at              TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_apple_tx_user
                ON portrait_apple_transactions(user_id, created_at);

            CREATE TABLE IF NOT EXISTS portrait_rate_events (
                event_id       TEXT PRIMARY KEY,
                scope          TEXT NOT NULL,
                subject_hash   TEXT NOT NULL,
                occurred_at    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_rate_events_lookup
                ON portrait_rate_events(scope, subject_hash, occurred_at);

            CREATE TABLE IF NOT EXISTS portrait_preview_grants (
                fingerprint_hash TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                granted_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS portrait_operational_events (
                event_id       TEXT PRIMARY KEY,
                event_type     TEXT NOT NULL,
                project_id     TEXT,
                transaction_id TEXT,
                metadata_json  TEXT NOT NULL DEFAULT '{}',
                created_at     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_ops_type_time
                ON portrait_operational_events(event_type, created_at);

            CREATE TABLE IF NOT EXISTS portrait_clean_export_requests (
                request_id      TEXT PRIMARY KEY,
                subject_hash    TEXT NOT NULL,
                project_id      TEXT NOT NULL,
                asset_id        TEXT NOT NULL,
                terms_version   TEXT NOT NULL,
                requested_at    TEXT NOT NULL,
                retain_until    TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_clean_exports_time
                ON portrait_clean_export_requests(requested_at);

            CREATE TABLE IF NOT EXISTS portrait_share_recipes (
                share_id          TEXT PRIMARY KEY,
                share_token       TEXT NOT NULL UNIQUE,
                user_id           TEXT NOT NULL,
                project_id        TEXT NOT NULL,
                theme_id          TEXT,
                title             TEXT NOT NULL,
                recipe_json       TEXT NOT NULL,
                include_portrait  INTEGER NOT NULL DEFAULT 0,
                hero_image_id     TEXT,
                status            TEXT NOT NULL DEFAULT 'active',
                created_at        TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_portrait_share_token
                ON portrait_share_recipes(share_token, status);
            """
        )
        project_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(portrait_projects)")
        }
        if "preview_retries_used" not in project_columns:
            conn.execute(
                "ALTER TABLE portrait_projects ADD COLUMN "
                "preview_retries_used INTEGER NOT NULL DEFAULT 0"
            )


def create_guest_user(
    now: datetime, *, preview_fingerprint: str | None = None,
) -> tuple[dict[str, Any], str]:
    user_id = _id("usr")
    token = secrets.token_urlsafe(32)
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO portrait_users (user_id, token_hash, created_at) VALUES (?, ?, ?)",
            (user_id, _hash_token(token), now.isoformat()),
        )
        grant_preview = preview_fingerprint is None
        if preview_fingerprint is not None:
            fingerprint_hash = _hash_token(preview_fingerprint)
            try:
                conn.execute(
                    """INSERT INTO portrait_preview_grants
                       (fingerprint_hash, user_id, granted_at) VALUES (?, ?, ?)""",
                    (fingerprint_hash, user_id, now.isoformat()),
                )
                grant_preview = True
            except sqlite3.IntegrityError:
                grant_preview = False
        if grant_preview:
            conn.execute(
                "INSERT INTO portrait_ledger (entry_id, user_id, amount, reason, created_at) "
                "VALUES (?, ?, 1, 'welcome_preview', ?)",
                (_id("led"), user_id, now.isoformat()),
            )
    return {"user_id": user_id, "created_at": now}, token


def user_for_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM portrait_users WHERE token_hash=?",
            (_hash_token(token),),
        ).fetchone()
    return dict(row) if row else None


def link_apple_identity(
    *, current_user_id: str, subject: str, email: str | None,
    display_name: str | None, now: datetime,
) -> tuple[dict[str, Any], str, bool]:
    """Link Apple identity, merging the current guest into an existing account."""
    if not subject:
        raise ValueError("Apple subject is required")
    new_token = secrets.token_urlsafe(32)
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT * FROM portrait_users WHERE user_id=?", (current_user_id,),
        ).fetchone()
        if not current:
            raise ValueError("Current user no longer exists")
        identity = conn.execute(
            "SELECT user_id FROM portrait_identities WHERE provider='apple' AND subject=?",
            (subject,),
        ).fetchone()
        target_user_id = identity["user_id"] if identity else current_user_id
        merged = target_user_id != current_user_id

        if identity is None:
            conn.execute(
                """INSERT INTO portrait_identities
                   (provider, subject, user_id, email, created_at)
                   VALUES ('apple', ?, ?, ?, ?)""",
                (subject, target_user_id, email, now.isoformat()),
            )
        elif email:
            conn.execute(
                """UPDATE portrait_identities SET email=?
                   WHERE provider='apple' AND subject=?""",
                (email, subject),
            )

        if merged:
            # A repeated anonymous workspace must never mint another free
            # preview when it is merged into an existing Apple account.
            conn.execute(
                "DELETE FROM portrait_ledger WHERE user_id=? AND reason='welcome_preview'",
                (current_user_id,),
            )
            for table in (
                "portrait_projects", "portrait_assets", "portrait_orders",
                "portrait_share_recipes", "portrait_ledger",
                "portrait_preview_grants", "portrait_apple_transactions",
            ):
                conn.execute(
                    f"UPDATE {table} SET user_id=? WHERE user_id=?",
                    (target_user_id, current_user_id),
                )
            conn.execute("DELETE FROM portrait_users WHERE user_id=?", (current_user_id,))

        conn.execute(
            """UPDATE portrait_users
               SET token_hash=?, account_type='apple',
                   display_name=COALESCE(?, display_name)
               WHERE user_id=?""",
            (_hash_token(new_token), display_name, target_user_id),
        )
        row = conn.execute(
            "SELECT * FROM portrait_users WHERE user_id=?", (target_user_id,),
        ).fetchone()
    return dict(row), new_token, merged


def check_rate_limit(
    *, scope: str, subject: str, max_calls: int,
    window_seconds: int, now: datetime,
) -> tuple[bool, int]:
    """Persist and atomically enforce a sliding-window limit."""
    subject_hash = _hash_token(subject)
    cutoff = now - timedelta(seconds=window_seconds)
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM portrait_rate_events WHERE occurred_at<?",
            ((now - timedelta(days=2)).isoformat(),),
        )
        rows = conn.execute(
            """SELECT occurred_at FROM portrait_rate_events
               WHERE scope=? AND subject_hash=? AND occurred_at>=?
               ORDER BY occurred_at""",
            (scope, subject_hash, cutoff.isoformat()),
        ).fetchall()
        if len(rows) >= max_calls:
            oldest = datetime.fromisoformat(rows[0]["occurred_at"])
            retry_after = max(1, int((oldest + timedelta(seconds=window_seconds) - now).total_seconds()))
            return False, retry_after
        conn.execute(
            """INSERT INTO portrait_rate_events
               (event_id, scope, subject_hash, occurred_at) VALUES (?, ?, ?, ?)""",
            (_id("rate"), scope, subject_hash, now.isoformat()),
        )
    return True, 0


def upsert_theme(theme: dict[str, Any], now: datetime) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_themes
               (theme_id, slug, title, title_en, tagline, category, cover_image,
                preview_json, use_cases_json, source_style_key, featured,
                sort_order, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
               ON CONFLICT(source_style_key) DO UPDATE SET
                 slug=excluded.slug, title=excluded.title,
                 title_en=excluded.title_en, tagline=excluded.tagline,
                 category=excluded.category, cover_image=excluded.cover_image,
                 preview_json=excluded.preview_json,
                 use_cases_json=excluded.use_cases_json,
                 featured=excluded.featured, sort_order=excluded.sort_order,
                 status='active',
                 updated_at=excluded.updated_at""",
            (
                theme["theme_id"], theme["slug"], theme["title"],
                theme["title_en"], theme["tagline"], theme["category"],
                theme["cover_image"], json.dumps(theme["preview_images"]),
                json.dumps(theme["use_cases"], ensure_ascii=False),
                theme["source_style_key"], int(theme["featured"]),
                theme["sort_order"], now.isoformat(), now.isoformat(),
            ),
        )
        stored = conn.execute(
            "SELECT theme_id FROM portrait_themes WHERE source_style_key=?",
            (theme["source_style_key"],),
        ).fetchone()
        theme_id = stored["theme_id"]
        conn.execute(
            """INSERT INTO portrait_theme_versions
               (theme_version_id, theme_id, version, blueprint_json, created_at)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(theme_id, version) DO UPDATE SET
                 blueprint_json=excluded.blueprint_json""",
            (
                f"thv_{theme_id.removeprefix('thm_')}_v1",
                theme_id,
                json.dumps(theme["blueprint"], ensure_ascii=False),
                now.isoformat(),
            ),
        )


def mark_unlisted_themes_legacy(
    active_source_keys: set[str], now: datetime,
) -> None:
    """Hide retired catalog entries without breaking projects that reference them."""
    if not active_source_keys:
        raise ValueError("At least one active catalog theme is required")
    placeholders = ",".join("?" for _ in active_source_keys)
    values = [now.isoformat(), *sorted(active_source_keys)]
    with storage.get_conn() as conn:
        conn.execute(
            f"""UPDATE portrait_themes
                SET status='legacy', updated_at=?
                WHERE status='active'
                  AND source_style_key NOT IN ({placeholders})""",
            values,
        )


def list_themes() -> list[dict[str, Any]]:
    with storage.get_conn() as conn:
        rows = conn.execute(
            """SELECT t.*, v.version AS active_version
               FROM portrait_themes t
               JOIN portrait_theme_versions v ON v.theme_id=t.theme_id
               WHERE t.status='active' AND v.status='active'
               ORDER BY t.featured DESC, t.sort_order, t.title_en"""
        ).fetchall()
    return [_theme_row(row) for row in rows]


def get_theme(identifier: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            """SELECT t.*, v.theme_version_id, v.version AS active_version,
                      v.blueprint_json
               FROM portrait_themes t
               JOIN portrait_theme_versions v ON v.theme_id=t.theme_id
               WHERE (t.theme_id=? OR t.slug=?)
                     AND t.status IN ('active', 'legacy')
                     AND v.status='active'
               ORDER BY v.version DESC LIMIT 1""",
            (identifier, identifier),
        ).fetchone()
    if not row:
        return None
    item = _theme_row(row)
    item["theme_version_id"] = row["theme_version_id"]
    item["blueprint"] = json.loads(row["blueprint_json"])
    return item


def _theme_row(row) -> dict[str, Any]:
    return {
        "theme_id": row["theme_id"],
        "slug": row["slug"],
        "title": row["title"],
        "title_en": row["title_en"],
        "tagline": row["tagline"],
        "category": row["category"],
        "cover_image": row["cover_image"],
        "preview_images": json.loads(row["preview_json"]),
        "use_cases": json.loads(row["use_cases_json"]),
        "featured": bool(row["featured"]),
        "source_style_key": row["source_style_key"],
        "active_version": row["active_version"],
    }


def create_project(user_id: str, payload: dict[str, Any], now: datetime) -> dict[str, Any]:
    project_id = _id("prj")
    shared = None
    if payload.get("source") == "shared_recipe":
        shared = get_share_recipe(payload.get("shared_recipe_id") or "")
        if not shared:
            raise ValueError("Shared recipe not found")
    theme_identifier = shared.get("theme_id") if shared else payload.get("theme_id")
    theme = get_theme(theme_identifier) if theme_identifier else None
    if payload.get("theme_id") and not theme:
        raise ValueError("Theme not found")
    status = "awaiting_references" if theme else "draft"
    presentation = (theme or {}).get("blueprint", {}).get("presentation")
    project_gender = (
        presentation if presentation in {"male", "female"}
        else payload["gender"]
    )
    values = (
        project_id, user_id, theme["theme_id"] if theme else None,
        theme.get("theme_version_id") if theme else None, payload["source"],
        status, project_gender, payload.get("shared_recipe_id"),
        now.isoformat(), now.isoformat(),
    )
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_projects
               (project_id, user_id, theme_id, theme_version_id, source, status,
                gender, shared_recipe_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        if shared and shared["recipe"].get("inspiration_spec"):
            conn.execute(
                """UPDATE portrait_projects SET inspiration_spec_json=?
                   WHERE project_id=? AND user_id=?""",
                (
                    json.dumps(shared["recipe"]["inspiration_spec"], ensure_ascii=False),
                    project_id, user_id,
                ),
            )
    return get_project(project_id, user_id)


def get_project(project_id: str, user_id: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM portrait_projects WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["inspiration_spec"] = (
        json.loads(item.pop("inspiration_spec_json"))
        if item.get("inspiration_spec_json") else None
    )
    item["preview_retries_used"] = int(item.get("preview_retries_used") or 0)
    item["preview_retries_remaining"] = max(
        0, 1 - item["preview_retries_used"]
    )
    return item


def list_projects(user_id: str) -> list[dict[str, Any]]:
    with storage.get_conn() as conn:
        rows = conn.execute(
            "SELECT project_id FROM portrait_projects WHERE user_id=? "
            "ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [project for row in rows if (
        project := get_project(row["project_id"], user_id)
    ) is not None]


def attach_legacy_session(
    *, project_id: str, user_id: str, session_id: str, gender: str,
    status: str, now: datetime,
) -> None:
    with storage.get_conn() as conn:
        cursor = conn.execute(
            """UPDATE portrait_projects
               SET legacy_session_id=?, gender=?, status=?, updated_at=?
               WHERE project_id=? AND user_id=?""",
            (session_id, gender, status, now.isoformat(), project_id, user_id),
        )
    if cursor.rowcount != 1:
        raise ValueError("Project not found")


def set_project_status(
    *, project_id: str, user_id: str, status: str, now: datetime,
) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            "UPDATE portrait_projects SET status=?, updated_at=? "
            "WHERE project_id=? AND user_id=?",
            (status, now.isoformat(), project_id, user_id),
        )


def set_project_preview_ready(
    *, project_id: str, user_id: str, image_id: str, now: datetime,
) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            """UPDATE portrait_projects
               SET status='preview_ready', hero_asset_id=?, updated_at=?
               WHERE project_id=? AND user_id=?""",
            (image_id, now.isoformat(), project_id, user_id),
        )


def reserve_project_preview_retry(
    *, project_id: str, user_id: str, now: datetime, max_retries: int = 1,
) -> int:
    """Atomically reserve one user-requested hero retry and return remaining."""
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT status, preview_retries_used FROM portrait_projects
               WHERE project_id=? AND user_id=?""",
            (project_id, user_id),
        ).fetchone()
        if not row:
            raise KeyError(project_id)
        used = int(row["preview_retries_used"] or 0)
        if row["status"] != "preview_ready":
            raise ValueError("Preview must be ready before requesting a closer match")
        if used >= max_retries:
            raise PermissionError("The complimentary closer-match retry has already been used")
        updated = conn.execute(
            """UPDATE portrait_projects
               SET preview_retries_used=preview_retries_used+1,
                   status='preview_generating', hero_asset_id=NULL, updated_at=?
               WHERE project_id=? AND user_id=?
                 AND status='preview_ready' AND preview_retries_used=?""",
            (now.isoformat(), project_id, user_id, used),
        )
        if updated.rowcount != 1:
            raise ValueError("Preview state changed; refresh and try again")
    return max(0, max_retries - used - 1)


def rollback_project_preview_retry(
    *, project_id: str, user_id: str, hero_asset_id: str, now: datetime,
) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            """UPDATE portrait_projects
               SET preview_retries_used=MAX(0, preview_retries_used-1),
                   status='preview_ready', hero_asset_id=?, updated_at=?
               WHERE project_id=? AND user_id=?""",
            (hero_asset_id, now.isoformat(), project_id, user_id),
        )


def save_reference_asset(
    *, user_id: str, project_id: str, storage_path: str, mime_type: str,
    quality: dict[str, Any], now: datetime,
) -> str:
    asset_id = _id("ast")
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_assets
               (asset_id, user_id, project_id, asset_type, storage_path,
                mime_type, metadata_json, created_at)
               VALUES (?, ?, ?, 'identity_reference', ?, ?, ?, ?)""",
            (
                asset_id, user_id, project_id, storage_path, mime_type,
                json.dumps({"quality": quality, "private": True}),
                now.isoformat(),
            ),
        )
    return asset_id


def get_asset(asset_id: str, user_id: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM portrait_assets WHERE asset_id=? AND user_id=?",
            (asset_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def spend_credit_once(
    *, user_id: str, reason: str, reference_id: str, now: datetime,
) -> bool:
    """Atomically debit one credit once for an idempotent operation."""
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT 1 FROM portrait_ledger WHERE user_id=? AND reason=? "
            "AND reference_id=?",
            (user_id, reason, reference_id),
        ).fetchone()
        if existing:
            return False
        balance = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS balance FROM portrait_ledger "
            "WHERE user_id=?",
            (user_id,),
        ).fetchone()["balance"]
        if int(balance) < 1:
            raise ValueError("No preview credit available")
        conn.execute(
            """INSERT INTO portrait_ledger
               (entry_id, user_id, amount, reason, reference_id, created_at)
               VALUES (?, ?, -1, ?, ?, ?)""",
            (_id("led"), user_id, reason, reference_id, now.isoformat()),
        )
    return True


def restore_credit_spend(
    *, user_id: str, reason: str, reference_id: str,
) -> bool:
    """Undo a debit only when job submission failed before work was queued."""
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """DELETE FROM portrait_ledger
               WHERE entry_id=(
                 SELECT entry_id FROM portrait_ledger
                 WHERE user_id=? AND reason=? AND reference_id=? AND amount=-1
                 ORDER BY created_at DESC LIMIT 1
               )""",
            (user_id, reason, reference_id),
        )
    return cursor.rowcount == 1


def save_portrait_order(
    *, order_id: str, user_id: str, project_id: str, product_code: str,
    amount_cents: int, status: str, now: datetime, provider: str = "paddle",
    provider_transaction_id: str | None = None, currency: str = "USD",
) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_orders
               (order_id, user_id, project_id, provider,
                provider_transaction_id, product_code,
                amount_cents, currency, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(order_id) DO UPDATE SET
                 status=excluded.status,
                 provider_transaction_id=COALESCE(
                    excluded.provider_transaction_id,
                    portrait_orders.provider_transaction_id
                 )""",
            (
                order_id, user_id, project_id, provider,
                provider_transaction_id, product_code,
                amount_cents, currency, status, now.isoformat(),
            ),
        )


def grant_support_entitlement(
    *, user_id: str, project_id: str, reason: str, now: datetime,
) -> dict[str, Any]:
    """Record an audited no-charge replacement entitlement."""
    if not reason.strip():
        raise ValueError("A support reason is required")
    order_id = _id("support")
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        project = conn.execute(
            "SELECT 1 FROM portrait_projects WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        ).fetchone()
        if not project:
            raise ValueError("Project not found")
        existing = conn.execute(
            """SELECT * FROM portrait_orders
               WHERE user_id=? AND project_id=? AND provider='support'
                     AND status='paid'
               ORDER BY created_at DESC LIMIT 1""",
            (user_id, project_id),
        ).fetchone()
        if existing:
            return dict(existing)
        conn.execute(
            """INSERT INTO portrait_orders
               (order_id, user_id, project_id, provider, product_code,
                amount_cents, currency, status, created_at, paid_at)
               VALUES (?, ?, ?, 'support', 'portrait_set_6', 0, 'USD',
                       'paid', ?, ?)""",
            (order_id, user_id, project_id, now.isoformat(), now.isoformat()),
        )
        conn.execute(
            """INSERT INTO portrait_operational_events
               (event_id, event_type, project_id, metadata_json, created_at)
               VALUES (?, 'support_entitlement', ?, ?, ?)""",
            (
                _id("ops"), project_id,
                json.dumps({"reason": reason.strip()[:500]}), now.isoformat(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM portrait_orders WHERE order_id=?", (order_id,),
        ).fetchone()
    return dict(row)


def record_operational_event(
    event_type: str, *, now: datetime, project_id: str | None = None,
    transaction_id: str | None = None, metadata: dict[str, Any] | None = None,
) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_operational_events
               (event_id, event_type, project_id, transaction_id,
                metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                _id("ops"), event_type, project_id, transaction_id,
                json.dumps(metadata or {}, ensure_ascii=True), now.isoformat(),
            ),
        )


def record_clean_export_request(
    *, user_id: str, project_id: str, asset_id: str,
    terms_version: str, now: datetime, retention_days: int,
) -> str:
    """Record the user-requested no-visible-label export for legal audit."""
    request_id = _id("clean")
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_clean_export_requests
               (request_id, subject_hash, project_id, asset_id, terms_version,
                requested_at, retain_until)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                _hash_token(user_id),
                project_id,
                asset_id,
                terms_version,
                now.isoformat(),
                (now + timedelta(days=retention_days)).isoformat(),
            ),
        )
    return request_id


def operational_metrics() -> dict[str, Any]:
    cutoff = (storage.utcnow() - timedelta(hours=24)).isoformat()
    with storage.get_conn() as conn:
        project_rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM portrait_projects GROUP BY status"
        ).fetchall()
        order_rows = conn.execute(
            """SELECT provider, status, COUNT(*) AS count
               FROM portrait_orders GROUP BY provider, status"""
        ).fetchall()
        event_rows = conn.execute(
            """SELECT event_type, COUNT(*) AS count
               FROM portrait_operational_events
               WHERE created_at >= ? GROUP BY event_type""",
            (cutoff,),
        ).fetchall()
        delivered = conn.execute(
            "SELECT COUNT(*) AS count FROM portrait_photo_sets WHERE status='delivered'"
        ).fetchone()["count"]
    return {
        "projects": {row["status"]: row["count"] for row in project_rows},
        "orders": {
            f"{row['provider']}.{row['status']}": row["count"]
            for row in order_rows
        },
        "events_24h": {row["event_type"]: row["count"] for row in event_rows},
        "delivered_six_photo_sets": delivered,
    }


def support_project_snapshot(project_id: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        project = conn.execute(
            "SELECT * FROM portrait_projects WHERE project_id=?", (project_id,),
        ).fetchone()
        if not project:
            return None
        orders = conn.execute(
            """SELECT order_id, provider, provider_transaction_id, product_code,
                      status, created_at, paid_at
               FROM portrait_orders WHERE project_id=? ORDER BY created_at""",
            (project_id,),
        ).fetchall()
        transactions = conn.execute(
            """SELECT transaction_id, original_transaction_id, product_id,
                      environment, status, purchased_at, revoked_at
               FROM portrait_apple_transactions WHERE project_id=?""",
            (project_id,),
        ).fetchall()
        events = conn.execute(
            """SELECT event_type, metadata_json, created_at
               FROM portrait_operational_events WHERE project_id=?
               ORDER BY created_at DESC LIMIT 50""",
            (project_id,),
        ).fetchall()
    item = dict(project)
    item.pop("inspiration_spec_json", None)
    return {
        "project": item,
        "orders": [dict(row) for row in orders],
        "apple_transactions": [dict(row) for row in transactions],
        "events": [
            {
                "event_type": row["event_type"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in events
        ],
    }


def claim_apple_transaction(
    *, user_id: str, project_id: str, transaction_id: str,
    original_transaction_id: str | None, product_id: str,
    environment: str, bundle_id: str, signed_payload: str,
    purchased_at: datetime | None, now: datetime,
) -> tuple[dict[str, Any], bool]:
    """Atomically bind one verified Apple transaction to exactly one project."""
    order_id = f"app_{transaction_id}"
    payload_hash = hashlib.sha256(signed_payload.encode("utf-8")).hexdigest()
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        project = conn.execute(
            "SELECT 1 FROM portrait_projects WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        ).fetchone()
        if not project:
            raise ValueError("Project not found")
        existing = conn.execute(
            "SELECT * FROM portrait_apple_transactions WHERE transaction_id=?",
            (transaction_id,),
        ).fetchone()
        if existing:
            if (
                existing["user_id"] != user_id
                or existing["project_id"] != project_id
                or existing["product_id"] != product_id
            ):
                raise ValueError("Transaction has already been claimed")
            order = conn.execute(
                "SELECT * FROM portrait_orders WHERE order_id=?", (order_id,),
            ).fetchone()
            return dict(order), False
        conn.execute(
            """INSERT INTO portrait_apple_transactions
               (transaction_id, original_transaction_id, user_id, project_id,
                product_id, environment, bundle_id, status,
                signed_payload_hash, purchased_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'paid', ?, ?, ?, ?)""",
            (
                transaction_id, original_transaction_id, user_id, project_id,
                product_id, environment, bundle_id, payload_hash,
                purchased_at.isoformat() if purchased_at else None,
                now.isoformat(), now.isoformat(),
            ),
        )
        conn.execute(
            """INSERT INTO portrait_orders
               (order_id, user_id, project_id, provider,
                provider_transaction_id, product_code, amount_cents,
                currency, status, created_at, paid_at)
               VALUES (?, ?, ?, 'apple', ?, ?, 0,
                       'USD', 'paid', ?, ?)""",
            (
                order_id, user_id, project_id, transaction_id, product_id,
                now.isoformat(), now.isoformat(),
            ),
        )
        order = conn.execute(
            "SELECT * FROM portrait_orders WHERE order_id=?", (order_id,),
        ).fetchone()
    return dict(order), True


def revoke_apple_transaction(
    *, transaction_id: str, revoked_at: datetime, now: datetime,
) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        tx = conn.execute(
            "SELECT * FROM portrait_apple_transactions WHERE transaction_id=?",
            (transaction_id,),
        ).fetchone()
        if not tx:
            return None
        conn.execute(
            """UPDATE portrait_apple_transactions
               SET status='refunded', revoked_at=?, updated_at=?
               WHERE transaction_id=?""",
            (revoked_at.isoformat(), now.isoformat(), transaction_id),
        )
        conn.execute(
            """UPDATE portrait_orders SET status='refunded'
               WHERE provider='apple' AND provider_transaction_id=?""",
            (transaction_id,),
        )
    return dict(tx)


def has_paid_project_entitlement(user_id: str, project_id: str) -> bool:
    with storage.get_conn() as conn:
        row = conn.execute(
            """SELECT 1 FROM portrait_orders
               WHERE user_id=? AND project_id=? AND status='paid'
               LIMIT 1""",
            (user_id, project_id),
        ).fetchone()
    return row is not None


def paid_project_order(user_id: str, project_id: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM portrait_orders
               WHERE user_id=? AND project_id=? AND status='paid'
               ORDER BY paid_at DESC, created_at DESC LIMIT 1""",
            (user_id, project_id),
        ).fetchone()
    return dict(row) if row else None


def get_portrait_order(order_id: str, user_id: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM portrait_orders WHERE order_id=? AND user_id=?",
            (order_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def update_portrait_order_status(
    *, order_id: str, user_id: str, status: str, paid_at: datetime | None = None,
) -> None:
    with storage.get_conn() as conn:
        conn.execute(
            "UPDATE portrait_orders SET status=?, paid_at=COALESCE(?, paid_at) "
            "WHERE order_id=? AND user_id=?",
            (status, paid_at.isoformat() if paid_at else None, order_id, user_id),
        )


def delete_project_data(project_id: str, user_id: str) -> dict[str, Any] | None:
    """Delete portrait media metadata while preserving financial records."""
    with storage.get_conn() as conn:
        project = conn.execute(
            "SELECT * FROM portrait_projects WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        ).fetchone()
        if not project:
            return None
        asset_rows = conn.execute(
            """SELECT storage_path, metadata_json FROM portrait_assets
               WHERE project_id=? AND user_id=?""",
            (project_id, user_id),
        ).fetchall()
        paths: list[str] = []
        for row in asset_rows:
            paths.append(row["storage_path"])
            try:
                clean_path = json.loads(row["metadata_json"]).get(
                    "clean_storage_path"
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                clean_path = None
            if clean_path:
                paths.append(str(clean_path))
        conn.execute(
            "UPDATE portrait_orders SET project_id=NULL WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        )
        conn.execute(
            "UPDATE portrait_share_recipes SET status='revoked' "
            "WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        )
        conn.execute("DELETE FROM portrait_generation_runs WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM portrait_photo_sets WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM portrait_assets WHERE project_id=? AND user_id=?", (project_id, user_id))
        conn.execute("DELETE FROM portrait_projects WHERE project_id=? AND user_id=?", (project_id, user_id))
    return {"paths": paths, "legacy_session_id": project["legacy_session_id"]}


def delete_user_record(user_id: str) -> None:
    pseudonym = "deleted_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:20]
    with storage.get_conn() as conn:
        conn.execute(
            "UPDATE portrait_orders SET user_id=? WHERE user_id=?",
            (pseudonym, user_id),
        )
        conn.execute(
            "UPDATE portrait_apple_transactions SET user_id=? WHERE user_id=?",
            (pseudonym, user_id),
        )
        conn.execute("DELETE FROM portrait_identities WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM portrait_ledger WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM portrait_users WHERE user_id=?", (user_id,))


def project_for_legacy_session(session_id: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM portrait_projects WHERE legacy_session_id=?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def expire_project_sources(project_id: str, user_id: str) -> list[str]:
    """Remove source-asset metadata and reset a project for fresh uploads."""
    with storage.get_conn() as conn:
        rows = conn.execute(
            """SELECT storage_path FROM portrait_assets
               WHERE project_id=? AND user_id=?
                 AND asset_type IN ('identity_reference', 'inspiration')""",
            (project_id, user_id),
        ).fetchall()
        conn.execute(
            """DELETE FROM portrait_assets
               WHERE project_id=? AND user_id=?
                 AND asset_type IN ('identity_reference', 'inspiration')""",
            (project_id, user_id),
        )
        conn.execute(
            """UPDATE portrait_projects
               SET inspiration_asset_id=NULL, inspiration_spec_json=NULL,
                   status=CASE WHEN source='private_inspiration'
                               THEN 'draft' ELSE 'awaiting_references' END,
                   updated_at=?
               WHERE project_id=? AND user_id=?""",
            (storage.utcnow().isoformat(), project_id, user_id),
        )
    return [row["storage_path"] for row in rows]


def create_share_recipe(
    *, user_id: str, project: dict[str, Any], title: str,
    recipe: dict[str, Any], include_portrait: bool,
    hero_image_id: str | None, now: datetime,
) -> dict[str, Any]:
    share_id = _id("shr")
    share_token = secrets.token_urlsafe(12)
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_share_recipes
               (share_id, share_token, user_id, project_id, theme_id, title,
                recipe_json, include_portrait, hero_image_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                share_id, share_token, user_id, project["project_id"],
                project.get("theme_id"), title,
                json.dumps(recipe, ensure_ascii=False), int(include_portrait),
                hero_image_id if include_portrait else None, now.isoformat(),
            ),
        )
    return get_share_recipe(share_token)


def deliver_photo_set(
    *, user_id: str, project_id: str, title: str,
    images: list[dict[str, str]], now: datetime,
) -> str:
    """Register exactly one immutable delivered set for a project."""
    if len(images) != 6:
        raise ValueError("A delivered portrait set must contain exactly six images")
    if len({image["image_id"] for image in images}) != 6:
        raise ValueError("A delivered portrait set cannot contain duplicate images")
    with storage.get_conn() as conn:
        project = conn.execute(
            "SELECT 1 FROM portrait_projects WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        ).fetchone()
        if not project:
            raise ValueError("Project not found")
        existing = conn.execute(
            "SELECT photo_set_id FROM portrait_photo_sets WHERE project_id=?",
            (project_id,),
        ).fetchone()
        if existing:
            return existing["photo_set_id"]
        asset_ids: list[str] = []
        for image in images:
            asset_id = _id("ast")
            asset_ids.append(asset_id)
            conn.execute(
                """INSERT INTO portrait_assets
                   (asset_id, user_id, project_id, asset_type, storage_path,
                    mime_type, metadata_json, created_at)
                   VALUES (?, ?, ?, 'generated_portrait', ?, ?, ?, ?)""",
                (
                    asset_id, user_id, project_id, image["storage_path"],
                    image.get("mime_type", "image/png"),
                    json.dumps({
                        "legacy_image_id": image["image_id"],
                        "clean_storage_path": image.get("clean_storage_path"),
                    }),
                    now.isoformat(),
                ),
            )
        photo_set_id = _id("set")
        conn.execute(
            """INSERT INTO portrait_photo_sets
               (photo_set_id, project_id, title, status, cover_asset_id,
                asset_ids_json, created_at, delivered_at)
               VALUES (?, ?, ?, 'delivered', ?, ?, ?, ?)""",
            (
                photo_set_id, project_id, title,
                asset_ids[0] if asset_ids else None,
                json.dumps(asset_ids), now.isoformat(), now.isoformat(),
            ),
        )
        conn.execute(
            """UPDATE portrait_projects
               SET status='delivered', photo_set_id=?, hero_asset_id=?, updated_at=?
               WHERE project_id=? AND user_id=?""",
            (
                photo_set_id, asset_ids[0] if asset_ids else None,
                now.isoformat(), project_id, user_id,
            ),
        )
    return photo_set_id


def get_photo_set(
    photo_set_id: str, project_id: str, user_id: str,
) -> dict[str, Any] | None:
    """Return an owned delivered set with its assets in presentation order."""
    with storage.get_conn() as conn:
        row = conn.execute(
            """SELECT ps.* FROM portrait_photo_sets ps
               JOIN portrait_projects p ON p.project_id=ps.project_id
               WHERE ps.photo_set_id=? AND ps.project_id=? AND p.user_id=?
                 AND ps.status='delivered'""",
            (photo_set_id, project_id, user_id),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        asset_ids = json.loads(item.pop("asset_ids_json"))
        assets: list[dict[str, Any]] = []
        for position, asset_id in enumerate(asset_ids):
            asset = conn.execute(
                """SELECT asset_id, mime_type FROM portrait_assets
                   WHERE asset_id=? AND project_id=? AND user_id=?
                     AND asset_type='generated_portrait'""",
                (asset_id, project_id, user_id),
            ).fetchone()
            if asset:
                assets.append({**dict(asset), "position": position})
    item["assets"] = assets
    return item


def get_share_recipe(identifier: str) -> dict[str, Any] | None:
    with storage.get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM portrait_share_recipes
               WHERE (share_token=? OR share_id=?) AND status='active'""",
            (identifier, identifier),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["recipe"] = json.loads(item.pop("recipe_json"))
    item["include_portrait"] = bool(item["include_portrait"])
    return item


def save_inspiration(
    *, user_id: str, project_id: str, storage_path: str, mime_type: str,
    spec: dict[str, Any] | None, now: datetime,
) -> str:
    asset_id = _id("ast")
    with storage.get_conn() as conn:
        conn.execute(
            """INSERT INTO portrait_assets
               (asset_id, user_id, project_id, asset_type, storage_path,
                mime_type, metadata_json, created_at)
               VALUES (?, ?, ?, 'inspiration', ?, ?, ?, ?)""",
            (
                asset_id, user_id, project_id, storage_path, mime_type,
                json.dumps({"private": True, "redistribution": False}),
                now.isoformat(),
            ),
        )
        conn.execute(
            """UPDATE portrait_projects
               SET inspiration_asset_id=?, inspiration_spec_json=?,
                   source='private_inspiration', status='awaiting_references',
                   updated_at=? WHERE project_id=? AND user_id=?""",
            (
                asset_id,
                json.dumps(spec, ensure_ascii=False) if spec else None,
                now.isoformat(), project_id, user_id,
            ),
        )
    return asset_id


def credit_balance(user_id: str) -> int:
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS balance FROM portrait_ledger WHERE user_id=?",
            (user_id,),
        ).fetchone()
    return int(row["balance"])
