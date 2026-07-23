"""Tests for reference-photo intake quality gates."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server import input_quality  # noqa: E402
from server.input_quality import (  # noqa: E402
    assess_reference_diversity,
    assess_reference_photo,
    order_reference_paths_by_pose,
    summarize_reference_identity_embeddings,
    summarize_reference_pose_measurements,
    summarize_reference_set,
)


def test_reference_diversity_rejects_copied_and_near_duplicate_photos(tmp_path):
    from PIL import Image, ImageDraw

    original = tmp_path / "front.jpg"
    image = Image.new("RGB", (640, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((180, 120, 460, 500), fill="#9c725a")
    draw.rectangle((220, 520, 420, 760), fill="#315b77")
    image.save(original, "JPEG", quality=95)
    copies = []
    for index in range(3):
        copy = tmp_path / f"copy_{index}.jpg"
        image.save(copy, "JPEG", quality=92 - index)
        copies.append(copy)

    result = assess_reference_diversity([original, *copies], min_unique=4)

    assert result["pass"] is False
    assert result["status"] == "fail"
    assert "duplicate_reference" in result["issues"]
    assert result["measurements"]["unique_count"] == 1
    assert len(result["measurements"]["duplicate_pairs"]) == 3


def test_reference_diversity_rejects_resized_crop_derivative(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    rng = np.random.default_rng(42)
    original_pixels = rng.integers(0, 255, (900, 720, 3), dtype=np.uint8)
    for index in range(30):
        center = (60 + index * 19 % 600, 80 + index * 23 % 740)
        cv2.circle(original_pixels, center, 8 + index % 11, (20, 220, 80), 3)
    original = tmp_path / "original.jpg"
    cv2.imwrite(str(original), original_pixels)

    cropped_pixels = original_pixels[110:790, 90:630]
    cropped_pixels = cv2.resize(cropped_pixels, (720, 900))
    crop = tmp_path / "crop.jpg"
    cv2.imwrite(str(crop), cropped_pixels, [cv2.IMWRITE_JPEG_QUALITY, 88])

    unrelated = []
    for index in range(2):
        pixels = rng.integers(0, 255, (900, 720, 3), dtype=np.uint8)
        path = tmp_path / f"unrelated_{index}.jpg"
        cv2.imwrite(str(path), pixels)
        unrelated.append(path)

    result = assess_reference_diversity(
        [original, crop, *unrelated], min_unique=4,
    )

    assert result["pass"] is False
    assert result["measurements"]["unique_count"] == 3
    derivative = result["measurements"]["duplicate_pairs"][0]
    assert derivative["match_method"] == "local_features"
    assert derivative["local_inlier_count"] >= 24


def test_reference_set_blocks_duplicate_reference_summary():
    photo_quality = {
        f"{index}.jpg": {
            "filename": f"{index}.jpg", "pass": True, "issues": []
        }
        for index in range(4)
    }
    diversity = {
        "status": "fail",
        "pass": False,
        "issues": ["duplicate_reference"],
        "measurements": {"unique_count": 2},
    }

    result = summarize_reference_set(
        photo_quality,
        min_photos=4,
        diversity=diversity,
    )

    assert result["pass"] is False
    assert "duplicate_reference" in result["issues"]
    assert result["diversity"] == diversity


def test_reference_set_allows_one_similar_front_expression_pair_with_side_views():
    photo_quality = {
        f"{index}.jpg": {
            "filename": f"{index}.jpg", "pass": True, "issues": []
        }
        for index in range(4)
    }
    diversity = {
        "status": "fail",
        "pass": False,
        "issues": ["duplicate_reference"],
        "measurements": {
            "unique_count": 3,
            "duplicate_pairs": [{"left": "0.jpg", "right": "1.jpg"}],
        },
    }
    pose = {
        "status": "pass",
        "pass": True,
        "issues": [],
        "measurements": {
            "role_assignments": {
                "front": "0.jpg",
                "front_secondary": "1.jpg",
                "side_a": "2.jpg",
                "side_b": "3.jpg",
            }
        },
    }

    result = summarize_reference_set(
        photo_quality,
        min_photos=4,
        diversity=diversity,
        pose_diversity=pose,
    )

    assert result["pass"] is True
    assert result["diversity_exception"]["applied"] is True
    assert result["agent_action"]["action"] == "ACCEPT"


def test_front_pair_exception_requires_proven_left_and_right_views():
    photo_quality = {
        f"{index}.jpg": {
            "filename": f"{index}.jpg", "pass": True, "issues": []
        }
        for index in range(4)
    }
    diversity = {
        "status": "fail",
        "pass": False,
        "issues": ["duplicate_reference"],
        "measurements": {
            "unique_count": 3,
            "duplicate_pairs": [{"left": "0.jpg", "right": "1.jpg"}],
        },
    }
    pose = {
        "status": "fail",
        "pass": False,
        "issues": ["insufficient_pose_diversity"],
        "measurements": {"role_assignments": {}},
    }

    result = summarize_reference_set(
        photo_quality,
        min_photos=4,
        diversity=diversity,
        pose_diversity=pose,
    )

    assert result["pass"] is False
    assert "duplicate_reference" in result["issues"]
    assert "insufficient_pose_diversity" in result["issues"]


def test_reference_pose_diversity_requires_front_and_both_sides():
    result = summarize_reference_pose_measurements(
        {
            "front.jpg": [1.0, 2.0, 0.0],
            "left.jpg": [0.0, -24.0, 1.0],
            "right.jpg": [-2.0, 21.0, -1.0],
            "smile.jpg": [2.0, 5.0, 0.0],
        },
        photo_count=4,
    )

    assert result["pass"] is True
    assert result["status"] == "pass"
    assert result["measurements"]["has_front"] is True
    assert result["measurements"]["has_negative_side"] is True
    assert result["measurements"]["has_positive_side"] is True
    assert result["measurements"]["role_assignments"] == {
        "front": "front.jpg",
        "front_secondary": "smile.jpg",
        "side_a": "left.jpg",
        "side_b": "right.jpg",
    }


def test_reference_paths_follow_measured_pose_roles_not_upload_order(tmp_path):
    paths = [
        tmp_path / "right.jpg",
        tmp_path / "smile.jpg",
        tmp_path / "left.jpg",
        tmp_path / "front.jpg",
        tmp_path / "lifestyle.jpg",
    ]
    pose_diversity = {
        "measurements": {
            "role_assignments": {
                "front": "front.jpg",
                "front_secondary": "smile.jpg",
                "side_a": "left.jpg",
                "side_b": "right.jpg",
            }
        }
    }

    ordered = order_reference_paths_by_pose(paths, pose_diversity)

    assert [path.name for path in ordered] == [
        "front.jpg",
        "smile.jpg",
        "left.jpg",
        "right.jpg",
        "lifestyle.jpg",
    ]


def test_reference_pose_diversity_rejects_four_front_facing_crops():
    result = summarize_reference_pose_measurements(
        {
            "front.jpg": [0.0, 1.0, 0.0],
            "crop_1.jpg": [1.0, 3.0, 0.0],
            "crop_2.jpg": [-1.0, -2.0, 1.0],
            "crop_3.jpg": [2.0, 5.0, -1.0],
        },
        photo_count=4,
    )

    assert result["pass"] is False
    assert result["status"] == "fail"
    assert result["measurements"]["yaw_span"] == 7.0
    assert "insufficient_pose_diversity" in result["issues"]


def test_reference_pose_diversity_degrades_when_pose_model_is_unavailable():
    result = summarize_reference_pose_measurements(
        {"front.jpg": [0.0, 2.0, 0.0]},
        photo_count=4,
    )

    assert result["pass"] is True
    assert result["status"] == "unchecked"
    assert result["notes"] == "pose_estimator_unavailable_for_enough_references"


def test_reference_set_blocks_when_angles_are_too_similar():
    photo_quality = {
        f"{index}.jpg": {
            "filename": f"{index}.jpg", "pass": True, "issues": []
        }
        for index in range(4)
    }
    pose_diversity = {
        "status": "fail",
        "pass": False,
        "issues": ["insufficient_pose_diversity"],
        "measurements": {"yaw_span": 7.0},
    }

    result = summarize_reference_set(
        photo_quality,
        min_photos=4,
        pose_diversity=pose_diversity,
    )

    assert result["pass"] is False
    assert "insufficient_pose_diversity" in result["issues"]
    assert result["pose_diversity"] == pose_diversity


def test_blank_reference_photo_fails_before_generation(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), blank)
    monkeypatch.setattr(
        input_quality,
        "_detect_faces",
        lambda *_args: ([], "insightface", ""),
    )

    result = assess_reference_photo(path)

    assert result["pass"] is False
    assert "no_face" in result["issues"]


def test_off_center_reference_face_fails_before_generation(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    path = tmp_path / "off_center.png"
    cv2.imwrite(str(path), image)

    monkeypatch.setattr(
        input_quality,
        "_detect_faces",
        lambda *_args: ([(500, 250, 96, 96)], "insightface", ""),
    )

    result = assess_reference_photo(path)

    assert result["pass"] is False
    assert "face_off_center" in result["issues"]
    assert result["measurements"]["face_count"] == 1


def test_haar_negative_is_uncertain_instead_of_blocking_normal_photo(
    tmp_path,
    monkeypatch,
):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    image = np.random.randint(0, 255, (900, 700, 3), dtype=np.uint8)
    path = tmp_path / "side-profile.jpg"
    cv2.imwrite(str(path), image)
    monkeypatch.setattr(
        input_quality,
        "_detect_faces",
        lambda *_args: ([], "haar_fallback", "modern detector unavailable"),
    )

    result = assess_reference_photo(path)

    assert result["pass"] is True
    assert "no_face" not in result["issues"]
    assert "face_detection_uncertain" in result["issues"]
    assert result["measurements"]["face_detector"] == "haar_fallback"


def test_mediapipe_rechecks_when_insightface_returns_no_face(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    class EmptyInsightFace:
        @staticmethod
        def get(_image):
            return []

    relative_box = SimpleNamespace(
        xmin=0.25,
        ymin=0.2,
        width=0.5,
        height=0.55,
    )
    detection = SimpleNamespace(
        location_data=SimpleNamespace(relative_bounding_box=relative_box)
    )

    class FaceDetection:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def process(_image):
            return SimpleNamespace(detections=[detection])

    fake_mediapipe = SimpleNamespace(
        solutions=SimpleNamespace(
            face_detection=SimpleNamespace(FaceDetection=FaceDetection)
        )
    )
    monkeypatch.setattr(input_quality, "_get_identity_app", lambda: EmptyInsightFace())
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)

    image = np.zeros((800, 600, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces, detector, notes = input_quality._detect_faces(image, gray)

    assert detector == "mediapipe"
    assert faces == [(150, 160, 300, 440)]
    assert "insightface:no_faces" in notes


def test_modern_detector_box_is_used_for_geometry(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    image = np.random.randint(0, 255, (900, 700, 3), dtype=np.uint8)
    path = tmp_path / "portrait.jpg"
    cv2.imwrite(str(path), image)
    monkeypatch.setattr(
        input_quality,
        "_detect_faces",
        lambda *_args: ([(210, 180, 280, 320)], "insightface", ""),
    )

    result = assess_reference_photo(path)

    assert result["pass"] is True
    assert result["measurements"]["face_count"] == 1
    assert result["measurements"]["face_detector"] == "insightface"


def test_face_normalized_sharpness_ignores_smooth_background(
    tmp_path,
    monkeypatch,
):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    gray = np.full((800, 800), 145, dtype=np.uint8)
    face_box = (250, 190, 300, 360)
    cv2.ellipse(gray, (400, 370), (135, 170), 0, 0, 360, 175, -1)
    cv2.ellipse(gray, (350, 330), (28, 14), 0, 0, 360, 55, 3)
    cv2.ellipse(gray, (450, 330), (28, 14), 0, 0, 360, 55, 3)
    cv2.circle(gray, (350, 330), 6, 20, -1)
    cv2.circle(gray, (450, 330), 6, 20, -1)
    cv2.line(gray, (400, 345), (390, 420), 85, 4)
    cv2.ellipse(gray, (400, 460), (65, 24), 0, 5, 175, 70, 4)
    for offset in range(0, 260, 12):
        cv2.line(gray, (275 + offset, 245), (285 + offset, 260), 110, 1)

    path = tmp_path / "sharp-face-soft-background.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    monkeypatch.setattr(
        input_quality,
        "_detect_faces",
        lambda *_args: ([face_box], "insightface", ""),
    )

    result = assess_reference_photo(path)

    assert result["pass"] is True
    assert "too_blurry" not in result["issues"]
    assert result["measurements"]["sharpness_metric_source"] == "face_crop_256"
    assert len(result["measurements"]["failed_metrics"]) < 2


def test_face_normalized_sharpness_rejects_truly_blurred_face(
    tmp_path,
    monkeypatch,
):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    gray = np.full((800, 800), 145, dtype=np.uint8)
    face_box = (250, 190, 300, 360)
    cv2.ellipse(gray, (400, 370), (135, 170), 0, 0, 360, 175, -1)
    cv2.circle(gray, (350, 330), 9, 25, -1)
    cv2.circle(gray, (450, 330), 9, 25, -1)
    cv2.line(gray, (400, 345), (390, 420), 85, 5)
    cv2.ellipse(gray, (400, 460), (65, 24), 0, 5, 175, 70, 5)
    blurred = cv2.GaussianBlur(gray, (61, 61), 18)
    path = tmp_path / "blurred-face.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))
    monkeypatch.setattr(
        input_quality,
        "_detect_faces",
        lambda *_args: ([face_box], "insightface", ""),
    )

    result = assess_reference_photo(path)

    assert result["pass"] is False
    assert "too_blurry" in result["issues"]
    assert len(result["measurements"]["failed_metrics"]) >= 2


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
        "headline": "这张照片有些模糊",
        "guidance": "请选择眼睛和面部轮廓都清晰对焦的照片。",
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
