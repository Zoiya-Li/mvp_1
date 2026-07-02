"""Tests for reference-photo intake quality gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.input_quality import (  # noqa: E402
    assess_reference_photo,
    summarize_reference_identity_embeddings,
    summarize_reference_set,
)


def test_blank_reference_photo_fails_before_generation(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), blank)

    result = assess_reference_photo(path)

    assert result["pass"] is False
    assert "no_face" in result["issues"]


def test_off_center_reference_face_fails_before_generation(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    path = tmp_path / "off_center.png"
    cv2.imwrite(str(path), image)

    class FakeCascade:
        def __init__(self, *_args, **_kwargs):
            pass

        def detectMultiScale(self, *_args, **_kwargs):
            return [(500, 250, 96, 96)]

    monkeypatch.setattr(cv2, "CascadeClassifier", FakeCascade)

    result = assess_reference_photo(path)

    assert result["pass"] is False
    assert "face_off_center" in result["issues"]
    assert result["measurements"]["face_count"] == 1


def test_reference_set_requires_four_quality_photos():
    photo_quality = {
        "a.jpg": {"filename": "a.jpg", "pass": True, "issues": []},
        "b.jpg": {"filename": "b.jpg", "pass": True, "issues": []},
        "c.jpg": {"filename": "c.jpg", "pass": False, "issues": ["too_blurry"]},
    }

    summary = summarize_reference_set(photo_quality, min_photos=4)

    assert summary["pass"] is False
    assert summary["status"] == "request_better_reference"
    assert summary["allowed_actions"] == ["ACCEPT", "REQUEST_BETTER_REFERENCE"]
    assert summary["agent_action"] == {
        "action": "REQUEST_BETTER_REFERENCE",
        "reason": "need_at_least_4_photos",
        "state": "INPUT_CHECK",
        "executed": True,
    }
    assert "need_at_least_4_photos" in summary["issues"]
    assert "need_at_least_4_quality_photos" in summary["issues"]
    assert "c.jpg:too_blurry" in summary["issues"]


def test_reference_set_passes_with_four_quality_photos():
    photo_quality = {
        f"{idx}.jpg": {"filename": f"{idx}.jpg", "pass": True, "issues": []}
        for idx in range(4)
    }

    summary = summarize_reference_set(photo_quality, min_photos=4)

    assert summary["pass"] is True
    assert summary["status"] == "pass"
    assert summary["agent_action"] == {
        "action": "ACCEPT",
        "reason": "reference_quality_pass",
        "state": "INPUT_CHECK",
        "executed": True,
    }
    assert summary["passed_photos"] == 4
    assert [
        record["role"] for record in summary["role_coverage"]
    ] == ["front_neutral", "front_smile", "left_45", "right_45"]


def test_reference_set_requires_first_four_identity_role_slots_to_pass():
    photo_quality = {
        "front.jpg": {"filename": "front.jpg", "pass": True, "issues": []},
        "smile.jpg": {"filename": "smile.jpg", "pass": False, "issues": ["too_blurry"]},
        "left.jpg": {"filename": "left.jpg", "pass": True, "issues": []},
        "right.jpg": {"filename": "right.jpg", "pass": True, "issues": []},
        "lifestyle.jpg": {"filename": "lifestyle.jpg", "pass": True, "issues": []},
    }

    summary = summarize_reference_set(photo_quality, min_photos=4)

    assert summary["pass"] is False
    assert summary["status"] == "request_better_reference"
    assert "front_smile:needs_quality_reference" in summary["issues"]
    assert summary["role_coverage"][1] == {
        "role": "front_smile",
        "filename": "smile.jpg",
        "pass": False,
        "issues": ["too_blurry"],
    }


def test_reference_identity_embeddings_detect_same_person_set():
    summary = summarize_reference_identity_embeddings({
        "front.jpg": [1.0, 0.0, 0.0],
        "smile.jpg": [0.95, 0.05, 0.0],
        "left.jpg": [0.9, 0.1, 0.0],
    })

    assert summary["pass"] is True
    assert summary["status"] == "pass"
    assert summary["measurements"]["embedding_count"] == 3


def test_reference_identity_embeddings_detect_mismatch():
    summary = summarize_reference_identity_embeddings({
        "front.jpg": [1.0, 0.0, 0.0],
        "other_person.jpg": [-1.0, 0.0, 0.0],
    })

    assert summary["pass"] is False
    assert summary["status"] == "fail"
    assert "reference_identity_mismatch" in summary["issues"]
    assert summary["measurements"]["weakest_pair"] == [
        "front.jpg",
        "other_person.jpg",
    ]


def test_reference_set_blocks_when_reference_identities_mismatch():
    photo_quality = {
        f"{idx}.jpg": {"filename": f"{idx}.jpg", "pass": True, "issues": []}
        for idx in range(4)
    }
    identity_consistency = {
        "status": "fail",
        "pass": False,
        "issues": ["reference_identity_mismatch"],
        "measurements": {"min_pairwise_cosine": 0.1},
    }

    summary = summarize_reference_set(
        photo_quality,
        min_photos=4,
        identity_consistency=identity_consistency,
    )

    assert summary["pass"] is False
    assert summary["status"] == "request_better_reference"
    assert "reference_identity_mismatch" in summary["issues"]
    assert summary["identity_consistency"] == identity_consistency


def test_reference_set_does_not_block_when_identity_checker_unavailable():
    photo_quality = {
        f"{idx}.jpg": {"filename": f"{idx}.jpg", "pass": True, "issues": []}
        for idx in range(4)
    }
    identity_consistency = {
        "status": "unchecked",
        "pass": True,
        "issues": [],
        "measurements": {},
        "notes": "identity_consistency_checker_unavailable",
    }

    summary = summarize_reference_set(
        photo_quality,
        min_photos=4,
        identity_consistency=identity_consistency,
    )

    assert summary["pass"] is True
    assert summary["identity_consistency"]["status"] == "unchecked"
