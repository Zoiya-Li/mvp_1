"""Learning layer — feedback-driven threshold calibration and policy adaptation.

This module turns user feedback ("looks like me", "not like me", downloaded,
selected) into training signals that adjust evaluation thresholds and router
policies over time.  It is the bridge between static rule-based decisions and
an adaptive optimization system.

Design principles:
1.  Feedback is stored per-session and aggregated globally.
2.  Threshold adjustments are conservative (small deltas, bounded ranges).
3.  All changes are logged and reversible.
4.  No PII — only image_id, event type, and score are stored.
"""

from __future__ import annotations

import sqlite3
import threading
import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings


@dataclass(frozen=True)
class FeedbackLabel:
    """A single user feedback label tied to a generated image."""

    image_id: str
    session_id: str
    event: str
    score: int | None = None  # 0=negative, 1=neutral, 2=positive
    reason: str | None = None
    created_at: str | None = None


@dataclass
class ThresholdCalibration:
    """Current calibrated thresholds derived from feedback history."""

    identity_pass_threshold: float = 8.0
    identity_repair_threshold: float = 7.0
    identity_cosine_accept: float = 0.45
    quality_accept_threshold: float = 8.0
    # How many feedback samples contributed to this calibration
    sample_count: int = 0
    feedback_version: int = 0
    # When the calibration was last updated
    updated_at: str | None = None


class LearningLayer:
    """Feedback-driven calibration and policy adaptation."""

    # Conservative adjustment bounds
    MIN_IDENTITY_PASS = 7.0
    MAX_IDENTITY_PASS = 9.0
    MIN_IDENTITY_REPAIR = 6.0
    MAX_IDENTITY_REPAIR = 8.0
    MIN_COSINE = 0.35
    MAX_COSINE = 0.55
    DELTA_IDENTITY = 0.05
    DELTA_COSINE = 0.02

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or self._default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._calibration: ThresholdCalibration | None = None

    @staticmethod
    def _default_db_path() -> Path:
        return settings.data_dir / "learning_layer.db"

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback_labels (
                    label_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id      TEXT NOT NULL,
                    session_id    TEXT NOT NULL,
                    event         TEXT NOT NULL,
                    score         INTEGER,
                    reason        TEXT,
                    created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_image
                    ON feedback_labels(image_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_session
                    ON feedback_labels(session_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_event
                    ON feedback_labels(event);

                CREATE TABLE IF NOT EXISTS threshold_calibration (
                    id                    INTEGER PRIMARY KEY CHECK (id = 1),
                    identity_pass         REAL NOT NULL DEFAULT 8.0,
                    identity_repair       REAL NOT NULL DEFAULT 7.0,
                    identity_cosine       REAL NOT NULL DEFAULT 0.45,
                    quality_accept        REAL NOT NULL DEFAULT 8.0,
                    sample_count          INTEGER NOT NULL DEFAULT 0,
                    feedback_version      INTEGER NOT NULL DEFAULT 0,
                    updated_at            TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS policy_adjustments (
                    adjustment_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    field                 TEXT NOT NULL,
                    old_value             REAL NOT NULL,
                    new_value             REAL NOT NULL,
                    reason                TEXT NOT NULL,
                    sample_count_at_time  INTEGER NOT NULL,
                    created_at            TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pipeline_outcomes (
                    outcome_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    failure_class         TEXT NOT NULL,
                    action                TEXT NOT NULL,
                    strategy              TEXT NOT NULL,
                    route_mode            TEXT NOT NULL,
                    model                 TEXT,
                    shot_profile          TEXT NOT NULL,
                    passed                INTEGER NOT NULL,
                    identity_score        REAL,
                    quality_score         REAL,
                    cost                  REAL,
                    created_at            TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pipeline_outcome_context
                    ON pipeline_outcomes(failure_class, shot_profile, action);
                CREATE INDEX IF NOT EXISTS idx_pipeline_outcome_strategy
                    ON pipeline_outcomes(strategy, route_mode, model);
                """
            )
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(threshold_calibration)"
                ).fetchall()
            }
            if "feedback_version" not in columns:
                conn.execute(
                    """
                    ALTER TABLE threshold_calibration
                    ADD COLUMN feedback_version INTEGER NOT NULL DEFAULT 0
                    """
                )
            # Seed the single calibration row if absent
            conn.execute(
                """
                INSERT OR IGNORE INTO threshold_calibration
                (id, identity_pass, identity_repair, identity_cosine, quality_accept, sample_count, updated_at)
                VALUES (1, 8.0, 7.0, 0.45, 8.0, 0, ?)
                """,
                (datetime.now(timezone.utc).isoformat(),),
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Feedback ingestion ────────────────────────────────────

    def record_feedback(
        self,
        image_id: str,
        session_id: str,
        event: str,
        score: int | None = None,
        reason: str | None = None,
    ) -> None:
        """Store a user feedback label."""
        with self._conn() as conn:
            if event in {"looks_like_me", "not_like_me"}:
                # Identity feedback is a mutable label, not an append-only vote.
                # Keep the latest answer per image so repeated UI submissions
                # cannot overweight one portrait in global calibration.
                conn.execute(
                    """
                    DELETE FROM feedback_labels
                    WHERE image_id = ?
                      AND event IN ('looks_like_me', 'not_like_me')
                    """,
                    (image_id,),
                )
            conn.execute(
                """
                INSERT INTO feedback_labels
                (image_id, session_id, event, score, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    session_id,
                    event,
                    score,
                    reason,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        # Invalidate cached calibration
        self._calibration = None

    def feedback_for_image(self, image_id: str) -> list[FeedbackLabel]:
        """Return all feedback labels for a given image."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback_labels WHERE image_id = ? ORDER BY created_at",
                (image_id,),
            ).fetchall()
        return [
            FeedbackLabel(
                image_id=r["image_id"],
                session_id=r["session_id"],
                event=r["event"],
                score=r["score"],
                reason=r["reason"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def feedback_stats(self, since: str | None = None) -> dict:
        """Aggregate feedback statistics."""
        sql = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN event = 'looks_like_me' THEN 1 ELSE 0 END) as likes,
                SUM(CASE WHEN event = 'not_like_me' THEN 1 ELSE 0 END) as dislikes,
                SUM(CASE WHEN event = 'downloaded' THEN 1 ELSE 0 END) as downloads,
                SUM(CASE WHEN event = 'selected' THEN 1 ELSE 0 END) as selections,
                SUM(CASE WHEN event = 'bad_artifacts' THEN 1 ELSE 0 END) as artifacts
            FROM feedback_labels
        """
        params: tuple = ()
        if since:
            sql += " WHERE created_at > ?"
            params = (since,)
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        total = row["total"] or 0
        likes = row["likes"] or 0
        dislikes = row["dislikes"] or 0
        identity_feedback = likes + dislikes
        return {
            "total": total,
            "likes": likes,
            "dislikes": dislikes,
            "downloads": row["downloads"] or 0,
            "selections": row["selections"] or 0,
            "artifacts": row["artifacts"] or 0,
            "identity_accuracy": round(likes / identity_feedback, 4) if identity_feedback else None,
            "not_like_me_rate": round(dislikes / identity_feedback, 4) if identity_feedback else None,
        }

    # ── Threshold calibration ─────────────────────────────────

    def get_calibration(self) -> ThresholdCalibration:
        """Return current calibrated thresholds (cached)."""
        if self._calibration is not None:
            return self._calibration
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM threshold_calibration WHERE id = 1"
            ).fetchone()
        self._calibration = ThresholdCalibration(
            identity_pass_threshold=row["identity_pass"],
            identity_repair_threshold=row["identity_repair"],
            identity_cosine_accept=row["identity_cosine"],
            quality_accept_threshold=row["quality_accept"],
            sample_count=row["sample_count"],
            feedback_version=row["feedback_version"],
            updated_at=row["updated_at"],
        )
        return self._calibration

    def _adjust_field(
        self,
        conn: sqlite3.Connection,
        field: str,
        current: float,
        delta: float,
        min_val: float,
        max_val: float,
        reason: str,
        sample_count: int,
    ) -> float:
        """Apply a bounded delta and log the adjustment."""
        new_val = max(min_val, min(max_val, round(current + delta, 3)))
        if new_val == current:
            return current
        conn.execute(
            f"UPDATE threshold_calibration SET {field} = ?, updated_at = ? WHERE id = 1",
            (new_val, datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            """
            INSERT INTO policy_adjustments
            (field, old_value, new_value, reason, sample_count_at_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (field, current, new_val, reason, sample_count, datetime.now(timezone.utc).isoformat()),
        )
        return new_val

    def calibrate(self) -> ThresholdCalibration:
        """Run calibration from accumulated feedback and return new thresholds.

        Rules:
        - not_like_me rate > 15%  → lower identity_pass (more strict)
        - not_like_me rate < 5%   → raise identity_pass (can be more lenient)
        - likes / (likes + dislikes) > 90% → raise identity_repair threshold
        - likes / (likes + dislikes) < 70% → lower identity_repair threshold
        """
        stats = self.feedback_stats()
        cal = self.get_calibration()
        total = stats["total"]
        with self._conn() as conn:
            version_row = conn.execute(
                "SELECT COALESCE(MAX(label_id), 0) AS version FROM feedback_labels"
            ).fetchone()
        feedback_version = int(version_row["version"] or 0)
        if total < 10 or feedback_version <= cal.feedback_version:
            # Require a minimum sample and never re-apply a calibration to the
            # same dataset after a restart or repeated scheduler call.
            return cal

        not_like_me_rate = stats.get("not_like_me_rate") or 0.0
        identity_accuracy = stats.get("identity_accuracy") or 0.0

        with self._conn() as conn:
            # Adjust identity_pass_threshold
            if not_like_me_rate > 0.15:
                cal.identity_pass_threshold = self._adjust_field(
                    conn, "identity_pass", cal.identity_pass_threshold,
                    self.DELTA_IDENTITY, self.MIN_IDENTITY_PASS, self.MAX_IDENTITY_PASS,
                    f"not_like_me_rate={not_like_me_rate:.2%} > 15%", total,
                )
            elif not_like_me_rate < 0.05:
                cal.identity_pass_threshold = self._adjust_field(
                    conn, "identity_pass", cal.identity_pass_threshold,
                    -self.DELTA_IDENTITY, self.MIN_IDENTITY_PASS, self.MAX_IDENTITY_PASS,
                    f"not_like_me_rate={not_like_me_rate:.2%} < 5%", total,
                )

            # Adjust identity_repair_threshold
            if identity_accuracy > 0.90:
                cal.identity_repair_threshold = self._adjust_field(
                    conn, "identity_repair", cal.identity_repair_threshold,
                    self.DELTA_IDENTITY, self.MIN_IDENTITY_REPAIR, self.MAX_IDENTITY_REPAIR,
                    f"identity_accuracy={identity_accuracy:.2%} > 90%", total,
                )
            elif identity_accuracy < 0.70:
                cal.identity_repair_threshold = self._adjust_field(
                    conn, "identity_repair", cal.identity_repair_threshold,
                    -self.DELTA_IDENTITY, self.MIN_IDENTITY_REPAIR, self.MAX_IDENTITY_REPAIR,
                    f"identity_accuracy={identity_accuracy:.2%} < 70%", total,
                )

            # Adjust cosine threshold in sync with identity_pass
            if not_like_me_rate > 0.15:
                cal.identity_cosine_accept = self._adjust_field(
                    conn, "identity_cosine", cal.identity_cosine_accept,
                    self.DELTA_COSINE, self.MIN_COSINE, self.MAX_COSINE,
                    f"not_like_me_rate={not_like_me_rate:.2%} > 15%", total,
                )
            elif not_like_me_rate < 0.05:
                cal.identity_cosine_accept = self._adjust_field(
                    conn, "identity_cosine", cal.identity_cosine_accept,
                    -self.DELTA_COSINE, self.MIN_COSINE, self.MAX_COSINE,
                    f"not_like_me_rate={not_like_me_rate:.2%} < 5%", total,
                )

            # Update sample count
            conn.execute(
                """
                UPDATE threshold_calibration
                SET sample_count = ?, feedback_version = ?, updated_at = ?
                WHERE id = 1
                """,
                (total, feedback_version, datetime.now(timezone.utc).isoformat()),
            )

        cal.sample_count = total
        cal.feedback_version = feedback_version
        cal.updated_at = datetime.now(timezone.utc).isoformat()
        self._calibration = cal
        return cal

    def adjustment_history(self, limit: int = 20) -> list[dict]:
        """Return recent threshold adjustments for audit."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM policy_adjustments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "field": r["field"],
                "old_value": r["old_value"],
                "new_value": r["new_value"],
                "reason": r["reason"],
                "sample_count": r["sample_count_at_time"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ── Recovery-strategy memory ──────────────────────────────

    def record_pipeline_outcome(
        self,
        *,
        failure_class: str,
        action: str,
        strategy: str,
        route_mode: str,
        shot_profile: str,
        passed: bool,
        model: str | None = None,
        identity_score: float | None = None,
        quality_score: float | None = None,
        cost: float | None = None,
    ) -> None:
        """Persist one action outcome without storing image data or identity."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_outcomes
                (failure_class, action, strategy, route_mode, model,
                 shot_profile, passed, identity_score, quality_score, cost,
                 created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    failure_class or "unknown_quality",
                    action,
                    strategy,
                    route_mode or "primary",
                    model,
                    shot_profile or "default",
                    int(bool(passed)),
                    identity_score,
                    quality_score,
                    cost,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def strategy_stats(
        self,
        *,
        failure_class: str | None = None,
        shot_profile: str | None = None,
    ) -> list[dict]:
        """Return auditable pass/cost statistics for recovery strategies."""
        where = []
        params: list[str] = []
        if failure_class:
            where.append("failure_class = ?")
            params.append(failure_class)
        if shot_profile:
            where.append("shot_profile = ?")
            params.append(shot_profile)
        sql = """
            SELECT failure_class, action, strategy, route_mode, model,
                   shot_profile, COUNT(*) AS samples, SUM(passed) AS passes,
                   AVG(identity_score) AS mean_identity_score,
                   AVG(quality_score) AS mean_quality_score,
                   AVG(cost) AS mean_cost
            FROM pipeline_outcomes
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += """
            GROUP BY failure_class, action, strategy, route_mode, model,
                     shot_profile
            ORDER BY samples DESC, passes DESC
        """
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        results = []
        for row in rows:
            samples = int(row["samples"] or 0)
            passes = int(row["passes"] or 0)
            results.append({
                "failure_class": row["failure_class"],
                "action": row["action"],
                "strategy": row["strategy"],
                "route_mode": row["route_mode"],
                "model": row["model"],
                "shot_profile": row["shot_profile"],
                "samples": samples,
                "passes": passes,
                "pass_rate": round(passes / samples, 4) if samples else None,
                "wilson_lower_bound": round(
                    self._wilson_lower_bound(passes, samples), 4
                ),
                "mean_identity_score": row["mean_identity_score"],
                "mean_quality_score": row["mean_quality_score"],
                "mean_cost": row["mean_cost"],
            })
        return results

    def strategy_adjustment(
        self,
        *,
        failure_class: str,
        action: str,
        shot_profile: str,
        minimum_samples: int = 20,
    ) -> dict:
        """Return a bounded policy multiplier only after enough evidence."""
        matching = [
            row for row in self.strategy_stats(
                failure_class=failure_class,
                shot_profile=shot_profile,
            )
            if row["action"] == action
        ]
        samples = sum(int(row["samples"]) for row in matching)
        passes = sum(int(row["passes"]) for row in matching)
        if samples < minimum_samples:
            return {
                "active": False,
                "samples": samples,
                "passes": passes,
                "multiplier": 1.0,
                "reason": "insufficient_strategy_evidence",
            }
        lower_bound = self._wilson_lower_bound(passes, samples)
        # A conservative 15% maximum movement keeps learned evidence below hard
        # gates and the episode ladder in authority.
        multiplier = max(0.85, min(1.15, 0.85 + 0.30 * lower_bound))
        return {
            "active": True,
            "samples": samples,
            "passes": passes,
            "pass_rate": round(passes / samples, 4),
            "wilson_lower_bound": round(lower_bound, 4),
            "multiplier": round(multiplier, 4),
            "reason": "evidence_weighted_strategy_prior",
        }

    @staticmethod
    def _wilson_lower_bound(passes: int, samples: int, z: float = 1.96) -> float:
        if samples <= 0:
            return 0.0
        proportion = passes / samples
        denominator = 1 + z * z / samples
        centre = proportion + z * z / (2 * samples)
        margin = z * math.sqrt(
            (proportion * (1 - proportion) + z * z / (4 * samples)) / samples
        )
        return max(0.0, (centre - margin) / denominator)
