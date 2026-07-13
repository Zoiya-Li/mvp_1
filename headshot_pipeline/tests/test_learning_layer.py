"""Tests for the learning layer — feedback-driven threshold calibration."""

from __future__ import annotations

import sqlite3

import pytest
from pathlib import Path

from server.learning import LearningLayer, ThresholdCalibration, FeedbackLabel
from server.evaluation import EvaluationService


def _make_layer(tmp_path: Path) -> LearningLayer:
    return LearningLayer(db_path=tmp_path / "learning_test.db")


class TestFeedbackIngestion:
    def test_record_and_retrieve(self, tmp_path):
        ll = _make_layer(tmp_path)
        ll.record_feedback("img_1", "sess_1", "looks_like_me", score=2)
        labels = ll.feedback_for_image("img_1")
        assert len(labels) == 1
        assert labels[0].event == "looks_like_me"
        assert labels[0].score == 2

    def test_multiple_labels_same_image(self, tmp_path):
        ll = _make_layer(tmp_path)
        ll.record_feedback("img_1", "sess_1", "looks_like_me", score=2)
        ll.record_feedback("img_1", "sess_1", "downloaded")
        labels = ll.feedback_for_image("img_1")
        assert len(labels) == 2

    def test_latest_identity_label_replaces_previous_vote(self, tmp_path):
        ll = _make_layer(tmp_path)
        ll.record_feedback("img_1", "sess_1", "looks_like_me", score=2)
        ll.record_feedback("img_1", "sess_1", "not_like_me", score=0)

        labels = ll.feedback_for_image("img_1")

        assert len(labels) == 1
        assert labels[0].event == "not_like_me"

    def test_feedback_stats(self, tmp_path):
        ll = _make_layer(tmp_path)
        ll.record_feedback("img_1", "sess_1", "looks_like_me", score=2)
        ll.record_feedback("img_2", "sess_1", "not_like_me", score=0)
        ll.record_feedback("img_3", "sess_2", "downloaded")
        stats = ll.feedback_stats()
        assert stats["total"] == 3
        assert stats["likes"] == 1
        assert stats["dislikes"] == 1
        assert stats["downloads"] == 1
        assert stats["identity_accuracy"] == 0.5
        assert stats["not_like_me_rate"] == 0.5

    def test_empty_stats(self, tmp_path):
        ll = _make_layer(tmp_path)
        stats = ll.feedback_stats()
        assert stats["total"] == 0
        assert stats["identity_accuracy"] is None

    def test_existing_calibration_schema_is_migrated_in_place(self, tmp_path):
        db_path = tmp_path / "legacy_learning.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE threshold_calibration (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    identity_pass REAL NOT NULL DEFAULT 8.0,
                    identity_repair REAL NOT NULL DEFAULT 7.0,
                    identity_cosine REAL NOT NULL DEFAULT 0.45,
                    quality_accept REAL NOT NULL DEFAULT 8.0,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )

        ll = LearningLayer(db_path=db_path)

        assert ll.get_calibration().feedback_version == 0
        with sqlite3.connect(db_path) as conn:
            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(threshold_calibration)"
                )
            }
        assert "feedback_version" in columns


class TestCalibration:
    def test_default_calibration(self, tmp_path):
        ll = _make_layer(tmp_path)
        cal = ll.get_calibration()
        assert cal.identity_pass_threshold == 8.0
        assert cal.identity_repair_threshold == 7.0
        assert cal.identity_cosine_accept == 0.45
        assert cal.sample_count == 0
        assert cal.feedback_version == 0

    def test_calibrate_not_enough_data(self, tmp_path):
        ll = _make_layer(tmp_path)
        # Only 5 labels — below the 10-sample threshold
        for i in range(5):
            ll.record_feedback(f"img_{i}", "sess_1", "looks_like_me")
        cal = ll.calibrate()
        assert cal.identity_pass_threshold == 8.0  # unchanged

    def test_calibrate_high_not_like_me(self, tmp_path):
        ll = _make_layer(tmp_path)
        # 20 labels, 4 likes + 16 dislikes = 20% not_like_me rate
        for i in range(4):
            ll.record_feedback(f"img_{i}", "sess_1", "looks_like_me")
        for i in range(4, 20):
            ll.record_feedback(f"img_{i}", "sess_1", "not_like_me")
        cal = ll.calibrate()
        # not_like_me_rate=20% > 15% → identity_pass should decrease (more strict)
        # Wait: not_like_me_rate high means users think images DON'T look like them
        # So we need to be MORE strict → INCREASE threshold
        assert cal.identity_pass_threshold > 8.0
        assert cal.identity_cosine_accept > 0.45

    def test_calibrate_low_not_like_me(self, tmp_path):
        ll = _make_layer(tmp_path)
        # 20 labels, 19 likes + 1 dislike = 5% not_like_me rate
        for i in range(19):
            ll.record_feedback(f"img_{i}", "sess_1", "looks_like_me")
        ll.record_feedback("img_19", "sess_1", "not_like_me")
        cal = ll.calibrate()
        # not_like_me_rate=5% — borderline, should not increase (or may stay)
        # Actually 1/20 = 5%, which is NOT < 5%, so no change expected
        assert cal.identity_pass_threshold == 8.0

    def test_calibrate_very_low_not_like_me(self, tmp_path):
        ll = _make_layer(tmp_path)
        # 20 labels, 20 likes + 0 dislikes = 0% not_like_me
        for i in range(20):
            ll.record_feedback(f"img_{i}", "sess_1", "looks_like_me")
        cal = ll.calibrate()
        # not_like_me_rate=0% < 5% → identity_pass should decrease (more lenient)
        assert cal.identity_pass_threshold < 8.0
        assert cal.identity_cosine_accept < 0.45

    def test_calibrate_high_identity_accuracy(self, tmp_path):
        ll = _make_layer(tmp_path)
        # 20 labels, 19 likes + 1 dislike = 95% accuracy
        for i in range(19):
            ll.record_feedback(f"img_{i}", "sess_1", "looks_like_me")
        ll.record_feedback("img_19", "sess_1", "not_like_me")
        cal = ll.calibrate()
        # identity_accuracy=95% > 90% → identity_repair should increase
        assert cal.identity_repair_threshold > 7.0

    def test_calibrate_low_identity_accuracy(self, tmp_path):
        ll = _make_layer(tmp_path)
        # 20 labels, 10 likes + 10 dislikes = 50% accuracy
        for i in range(10):
            ll.record_feedback(f"img_{i}", "sess_1", "looks_like_me")
        for i in range(10, 20):
            ll.record_feedback(f"img_{i}", "sess_1", "not_like_me")
        cal = ll.calibrate()
        # identity_accuracy=50% < 70% → identity_repair should decrease
        assert cal.identity_repair_threshold < 7.0

    def test_adjustment_history(self, tmp_path):
        ll = _make_layer(tmp_path)
        for i in range(20):
            ll.record_feedback(f"img_{i}", "sess_1", "not_like_me")
        ll.calibrate()
        history = ll.adjustment_history()
        assert len(history) > 0
        assert history[0]["field"] in ("identity_pass", "identity_cosine")
        assert "old_value" in history[0]
        assert "new_value" in history[0]

    def test_calibration_bounds(self, tmp_path):
        ll = _make_layer(tmp_path)
        # Push threshold to the limit with extreme feedback
        for _ in range(5):
            for i in range(20):
                ll.record_feedback(f"img_{_}_{i}", "sess_1", "not_like_me")
            ll.calibrate()
        cal = ll.get_calibration()
        # Should be bounded at MAX_IDENTITY_PASS (high not_like_me → increase threshold)
        assert cal.identity_pass_threshold <= LearningLayer.MAX_IDENTITY_PASS
        assert cal.identity_cosine_accept <= LearningLayer.MAX_COSINE

    def test_caching(self, tmp_path):
        ll = _make_layer(tmp_path)
        cal1 = ll.get_calibration()
        cal2 = ll.get_calibration()
        assert cal1 is cal2  # same object (cached)
        # After recording feedback, cache should be invalidated
        ll.record_feedback("img_1", "sess_1", "looks_like_me")
        cal3 = ll.get_calibration()
        assert cal3 is not cal1

    def test_calibration_is_idempotent_without_new_samples(self, tmp_path):
        ll = _make_layer(tmp_path)
        for i in range(20):
            ll.record_feedback(f"img_{i}", "sess_1", "not_like_me")

        first = ll.calibrate().identity_pass_threshold
        second = ll.calibrate().identity_pass_threshold

        assert second == first

    def test_changed_identity_label_recalibrates_without_count_growth(self, tmp_path):
        ll = _make_layer(tmp_path)
        for i in range(20):
            ll.record_feedback(f"img_{i}", "sess_1", "not_like_me")
        first = ll.calibrate()
        first_version = first.feedback_version
        first_history_size = len(ll.adjustment_history())

        ll.record_feedback("img_0", "sess_1", "looks_like_me")
        second = ll.calibrate()

        assert second.sample_count == 20
        assert second.feedback_version > first_version
        assert len(ll.adjustment_history()) > first_history_size

    def test_calibrated_delta_preserves_shot_geometry_profile(self, tmp_path):
        ll = _make_layer(tmp_path)
        for i in range(20):
            ll.record_feedback(f"img_{i}", "sess_1", "not_like_me")
        ll.calibrate()

        thresholds = EvaluationService(ll)._get_identity_thresholds(
            {"shot_id": "full_body"}
        )

        assert thresholds["calibrated"] is True
        assert thresholds["calibration_sample_count"] == 20
        assert thresholds["identity_pass_threshold"] == pytest.approx(7.05)
        assert thresholds["identity_repair_threshold"] == pytest.approx(5.95)
