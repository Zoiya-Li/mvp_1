"""Local input-photo quality gates for reference selfies.

This is the intake-layer counterpart of generation QA: reject obviously bad
reference photos before spending image-model calls. It checks cheap objective
signals per image, then optionally verifies that the usable reference faces are
consistent with one task-local person. Embeddings are never returned or stored.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps


MIN_REFERENCE_DIM = 512
MIN_FACE_SIZE = 72
REFERENCE_FACE_SHARPNESS_SIZE = 256
MIN_FACE_BLUR_VARIANCE = 50.0
MIN_FACE_TENENGRAD = 1_500.0
MIN_FACE_EDGE_DENSITY = 0.03
MIN_FAILED_SHARPNESS_METRICS = 2
REFERENCE_IDENTITY_CONSISTENCY_MIN_COSINE = 0.35
REFERENCE_DUPLICATE_HASH_SIZE = 16
# A 16x16 average hash is intentionally only an exact/near-exact fallback.
# Portraits shot against the same wall can differ by expression while sharing
# the same low-frequency layout; crop/resize derivatives are caught by the
# stronger local-feature homography below.
REFERENCE_DUPLICATE_HAMMING_THRESHOLD = 3
REFERENCE_LOCAL_MATCH_MIN_INLIERS = 24
REFERENCE_LOCAL_MATCH_MIN_INLIER_RATIO = 0.72
REFERENCE_LOCAL_MATCH_MIN_COVERAGE = 0.18
REFERENCE_POSE_FRONT_MAX_ABS_YAW = 12.0
REFERENCE_POSE_SIDE_MIN_ABS_YAW = 15.0
REFERENCE_POSE_MIN_YAW_SPAN = 30.0
PRIMARY_REFERENCE_ROLES = [
    "front_neutral",
    "front_smile",
    "left_45",
    "right_45",
]
OPTIONAL_REFERENCE_ROLES = [
    "lifestyle",
    "side_profile",
]

REFERENCE_ISSUE_GUIDANCE = {
    "unreadable_image": (
        "这张照片无法打开",
        "请重新选择原始的 JPEG、HEIC、PNG 或 WebP 文件。",
    ),
    "resolution_too_low": (
        "这张照片尺寸太小",
        "请选择相机原图，不要使用缩略图或截图。",
    ),
    "too_blurry": (
        "这张照片有些模糊",
        "请选择眼睛和面部轮廓都清晰对焦的照片。",
    ),
    "no_face": (
        "暂时没有清楚识别到你的脸",
        "请选择更明亮、没有墨镜、重度滤镜或面部遮挡的照片。",
    ),
    "multiple_faces": (
        "照片中出现了多张人脸",
        "请选择只有你一个人的照片。",
    ),
    "face_too_small": (
        "你的脸离镜头有些远",
        "请选择一张更近的肖像，让面部在画面中更清楚。",
    ),
    "face_too_close": (
        "面部裁切得太紧",
        "请选择包含完整头部、周围留有少量空间的照片。",
    ),
    "face_off_center": (
        "面部太靠近画面边缘",
        "请选择完整面部自然位于画面内的照片。",
    ),
    "reference_identity_mismatch": (
        "这些照片可能不是同一个人",
        "请使用同一位成年人的四到六张照片。",
    ),
    "duplicate_reference": (
        "这张照片与另一张太相似",
        "请选择真正不同的角度或神情，不要使用重复照片。",
    ),
    "insufficient_pose_diversity": (
        "这些照片的角度太相似",
        "请补充一张清晰正面照，并分别向左右轻转各拍一张。",
    ),
    "missing_reference": (
        "还缺少一个必要角度",
        "请补充这个角度，帮助我们从不同方向保留你的真实五官。",
    ),
}

_IDENTITY_APP = None
_IDENTITY_APP_LOAD_FAILED = False


def _clamp_face_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _detect_faces(img, gray) -> tuple[list[tuple[int, int, int, int]], str, str]:
    """Detect faces with modern detectors before the compatibility fallback.

    InsightFace is already used for task-local identity consistency, so using
    its detector here avoids contradictory outcomes where Haar rejects a face
    that the identity model finds successfully. MediaPipe is the lightweight
    secondary detector. Haar remains available for degraded installations but
    its negative result is treated as uncertain by the caller.
    """
    height, width = img.shape[:2]
    notes: list[str] = []
    modern_detector_ran = False

    app = _get_identity_app()
    if app is not None:
        try:
            boxes = []
            for face in app.get(img):
                clamped = _clamp_face_box(face.bbox, width, height)
                if clamped:
                    boxes.append(clamped)
            modern_detector_ran = True
            if boxes:
                return boxes, "insightface", ""
            notes.append("insightface:no_faces")
        except Exception as exc:
            notes.append(f"insightface:{exc}")

    try:
        import cv2
        import mediapipe as mp

        with mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        ) as detector:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = detector.process(rgb).detections or []
        modern_detector_ran = True
        boxes = []
        for detection in detections:
            relative = detection.location_data.relative_bounding_box
            clamped = _clamp_face_box(
                (
                    relative.xmin * width,
                    relative.ymin * height,
                    (relative.xmin + relative.width) * width,
                    (relative.ymin + relative.height) * height,
                ),
                width,
                height,
            )
            if clamped:
                boxes.append(clamped)
        if boxes:
            return boxes, "mediapipe", "; ".join(notes)
        notes.append("mediapipe:no_faces")
    except Exception as exc:
        notes.append(f"mediapipe:{exc}")

    try:
        import cv2

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        boxes = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
        normalized = [tuple(int(value) for value in box) for box in boxes]
        if normalized:
            return normalized, "haar_fallback", "; ".join(notes)
        detector_name = "modern_ensemble" if modern_detector_ran else "haar_fallback"
        return [], detector_name, "; ".join(notes)
    except Exception as exc:
        notes.append(f"haar:{exc}")
        detector_name = "modern_ensemble" if modern_detector_ran else "unavailable"
        return [], detector_name, "; ".join(notes)


def _measure_face_sharpness(
    gray,
    face_box: tuple[int, int, int, int],
) -> dict:
    """Measure facial detail at a stable scale instead of scoring the background."""
    import cv2

    x, y, width, height = face_box
    pad_x = int(width * 0.08)
    pad_y = int(height * 0.08)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(gray.shape[1], x + width + pad_x)
    y2 = min(gray.shape[0], y + height + pad_y)
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return {
            "pass": False,
            "failed_metrics": ["face_crop_unavailable"],
        }

    canonical = cv2.resize(
        crop,
        (REFERENCE_FACE_SHARPNESS_SIZE, REFERENCE_FACE_SHARPNESS_SIZE),
        interpolation=(
            cv2.INTER_AREA
            if max(crop.shape[:2]) >= REFERENCE_FACE_SHARPNESS_SIZE
            else cv2.INTER_CUBIC
        ),
    )
    laplacian_variance = float(cv2.Laplacian(canonical, cv2.CV_64F).var())
    grad_x = cv2.Sobel(canonical, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(canonical, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float((grad_x * grad_x + grad_y * grad_y).mean())
    edge_density = float((cv2.Canny(canonical, 50, 120) > 0).mean())
    failed_metrics = []
    if laplacian_variance < MIN_FACE_BLUR_VARIANCE:
        failed_metrics.append("laplacian")
    if tenengrad < MIN_FACE_TENENGRAD:
        failed_metrics.append("tenengrad")
    if edge_density < MIN_FACE_EDGE_DENSITY:
        failed_metrics.append("edge_density")
    return {
        "pass": len(failed_metrics) < MIN_FAILED_SHARPNESS_METRICS,
        "face_blur_variance": round(laplacian_variance, 2),
        "face_tenengrad": round(tenengrad, 2),
        "face_edge_density": round(edge_density, 5),
        "sharpness_metric_source": "face_crop_256",
        "failed_metrics": failed_metrics,
    }
def assess_reference_photo(path: str | Path) -> dict:
    result = {
        "filename": Path(path).name,
        "status": "unknown",
        "pass": False,
        "issues": [],
        "measurements": {},
        "notes": "",
    }
    try:
        import cv2
    except Exception as exc:
        # Do not block onboarding if the optional local detector is unavailable.
        result.update({
            "status": "unchecked",
            "pass": True,
            "notes": f"input_quality_unavailable: {exc}",
        })
        return result

    img = cv2.imread(str(path))
    if img is None:
        result["status"] = "fail"
        result["issues"].append("unreadable_image")
        result["notes"] = "Could not read image"
        return result

    height, width = img.shape[:2]
    min_dim = min(width, height)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    result["measurements"].update({
        "width": width,
        "height": height,
        "min_dim": min_dim,
        "blur_variance": round(blur_var, 2),
    })

    if min_dim < MIN_REFERENCE_DIM:
        result["issues"].append("resolution_too_low")
    faces, detector_name, detector_notes = _detect_faces(img, gray)
    result["measurements"]["face_detector"] = detector_name
    if detector_notes:
        result["measurements"]["face_detector_fallback_reason"] = detector_notes[:200]
    if detector_name == "unavailable":
        result["issues"].append("face_detector_failed")

    face_count = int(len(faces))
    result["measurements"]["face_count"] = face_count
    if face_count == 0:
        if detector_name in {"insightface", "mediapipe", "modern_ensemble"}:
            result["issues"].append("no_face")
        else:
            result["issues"].append("face_detection_uncertain")
    elif face_count > 1:
        result["issues"].append("multiple_faces")
    else:
        x, y, fw, fh = [int(v) for v in faces[0]]
        sharpness = _measure_face_sharpness(gray, (x, y, fw, fh))
        result["measurements"].update({
            key: value for key, value in sharpness.items() if key != "pass"
        })
        if not sharpness["pass"]:
            result["issues"].append("too_blurry")
        face_area_ratio = (fw * fh) / float(width * height)
        cx = x + fw / 2.0
        cy = y + fh / 2.0
        center_dx = abs(cx - width / 2.0) / width
        center_dy = abs(cy - height / 2.0) / height
        result["measurements"].update({
            "face_area_ratio": round(face_area_ratio, 4),
            "face_center_dx": round(center_dx, 4),
            "face_center_dy": round(center_dy, 4),
        })
        if face_area_ratio < 0.035:
            result["issues"].append("face_too_small")
        if face_area_ratio > 0.65:
            result["issues"].append("face_too_close")
        if center_dx > 0.28 or center_dy > 0.28:
            result["issues"].append("face_off_center")

    blocking = {
        "unreadable_image",
        "resolution_too_low",
        "too_blurry",
        "no_face",
        "multiple_faces",
        "face_too_small",
        "face_too_close",
        "face_off_center",
    }
    blocking_found = sorted(blocking.intersection(result["issues"]))
    result["pass"] = not blocking_found
    result["status"] = "pass" if result["pass"] else "fail"
    if result["issues"]:
        result["notes"] = ", ".join(result["issues"])
    return result


def _normalize(vec: Sequence[float]) -> list[float] | None:
    values = [float(v) for v in vec]
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-8:
        return None
    return [v / norm for v in values]


def summarize_reference_identity_embeddings(
    embeddings_by_filename: dict[str, Sequence[float]],
    min_cosine: float = REFERENCE_IDENTITY_CONSISTENCY_MIN_COSINE,
) -> dict:
    """Summarize task-local same-person consistency without storing embeddings."""
    names: list[str] = []
    embeddings: list[list[float]] = []
    for name, embedding in embeddings_by_filename.items():
        normalized = _normalize(embedding)
        if normalized is None:
            continue
        names.append(name)
        embeddings.append(normalized)

    result = {
        "status": "unchecked",
        "pass": True,
        "issues": [],
        "measurements": {
            "embedding_count": len(embeddings),
            "min_required_embeddings": 2,
        },
        "notes": "",
    }
    if len(embeddings) < 2:
        result["notes"] = "need_at_least_2_face_embeddings"
        return result

    pairwise = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            cosine = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            pairwise.append((names[i], names[j], cosine))

    min_pair = min(pairwise, key=lambda item: item[2])
    avg_cosine = sum(item[2] for item in pairwise) / len(pairwise)
    result["measurements"].update({
        "pairwise_count": len(pairwise),
        "min_pairwise_cosine": round(min_pair[2], 4),
        "avg_pairwise_cosine": round(avg_cosine, 4),
        "min_cosine_threshold": min_cosine,
        "weakest_pair": [min_pair[0], min_pair[1]],
    })
    if min_pair[2] < min_cosine:
        result["status"] = "fail"
        result["pass"] = False
        result["issues"].append("reference_identity_mismatch")
        result["notes"] = (
            "Reference photos appear to contain different identities: "
            f"{min_pair[0]} vs {min_pair[1]}"
        )
    else:
        result["status"] = "pass"
    return result


def summarize_reference_pose_measurements(
    poses_by_filename: dict[str, Sequence[float]],
    *,
    photo_count: int,
) -> dict:
    """Require real front/left/right head evidence when pose is measurable.

    InsightFace reports ``pitch, yaw, roll`` in degrees. This gate intentionally
    degrades to unchecked/pass when fewer than three faces expose pose, so an
    optional detector outage never blocks a legitimate upload.
    """
    readable: list[tuple[str, float, float, float]] = []
    for filename, pose in poses_by_filename.items():
        values = [float(value) for value in pose]
        if len(values) < 3 or not all(math.isfinite(value) for value in values[:3]):
            continue
        readable.append((filename, values[0], values[1], values[2]))

    result = {
        "status": "unchecked",
        "pass": True,
        "issues": [],
        "measurements": {
            "photo_count": int(photo_count),
            "pose_count": len(readable),
            "min_required_pose_count": 3,
            "front_max_abs_yaw": REFERENCE_POSE_FRONT_MAX_ABS_YAW,
            "side_min_abs_yaw": REFERENCE_POSE_SIDE_MIN_ABS_YAW,
            "min_yaw_span": REFERENCE_POSE_MIN_YAW_SPAN,
        },
        "notes": "",
    }
    if len(readable) < 3:
        result["notes"] = "pose_estimator_unavailable_for_enough_references"
        return result

    front = min(readable, key=lambda item: abs(item[2]))
    negative = min(readable, key=lambda item: item[2])
    positive = max(readable, key=lambda item: item[2])
    reserved = {front[0], negative[0], positive[0]}
    front_secondary = min(
        (item for item in readable if item[0] not in reserved),
        key=lambda item: abs(item[2]),
        default=front,
    )
    yaw_values = [item[2] for item in readable]
    yaw_span = max(yaw_values) - min(yaw_values)
    has_front = abs(front[2]) <= REFERENCE_POSE_FRONT_MAX_ABS_YAW
    has_negative_side = negative[2] <= -REFERENCE_POSE_SIDE_MIN_ABS_YAW
    has_positive_side = positive[2] >= REFERENCE_POSE_SIDE_MIN_ABS_YAW
    passed = (
        has_front
        and has_negative_side
        and has_positive_side
        and yaw_span >= REFERENCE_POSE_MIN_YAW_SPAN
    )
    result["measurements"].update({
        "yaw_min": round(min(yaw_values), 2),
        "yaw_max": round(max(yaw_values), 2),
        "yaw_span": round(yaw_span, 2),
        "has_front": has_front,
        "has_negative_side": has_negative_side,
        "has_positive_side": has_positive_side,
        "role_assignments": {
            "front": front[0],
            "front_secondary": front_secondary[0],
            "side_a": negative[0],
            "side_b": positive[0],
        },
    })
    result["status"] = "pass" if passed else "fail"
    result["pass"] = passed
    if not passed:
        result["issues"].append("insufficient_pose_diversity")
        result["notes"] = "References do not prove front and both side angles"
    return result


def order_reference_paths_by_pose(
    photo_paths: Sequence[str | Path],
    pose_diversity: dict | None,
) -> list[Path]:
    """Put measured identity views into the canonical generation role order."""
    paths = [Path(path) for path in photo_paths]
    assignments = (
        ((pose_diversity or {}).get("measurements") or {}).get("role_assignments")
        or {}
    )
    canonical_names = [
        assignments.get("front"),
        assignments.get("front_secondary"),
        assignments.get("side_a"),
        assignments.get("side_b"),
    ]
    by_name = {path.name: path for path in paths}
    ordered: list[Path] = []
    for filename in canonical_names:
        path = by_name.get(str(filename or ""))
        if path is not None and path not in ordered:
            ordered.append(path)
    ordered.extend(path for path in paths if path not in ordered)
    return ordered


def _get_identity_app():
    global _IDENTITY_APP, _IDENTITY_APP_LOAD_FAILED
    if _IDENTITY_APP is not None:
        return _IDENTITY_APP
    if _IDENTITY_APP_LOAD_FAILED:
        return None
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _IDENTITY_APP = app
        return _IDENTITY_APP
    except Exception:
        _IDENTITY_APP_LOAD_FAILED = True
        return None


def assess_reference_identity_consistency(photo_paths: Sequence[str | Path]) -> dict:
    """Check whether reference photos look like one task-local person.

    This is a privacy-conscious 1:small-set check. It computes embeddings in
    memory, returns only aggregate measurements, and degrades to unchecked/pass
    when the optional local recognizer is unavailable.
    """
    paths = [Path(p) for p in photo_paths]
    result = {
        "status": "unchecked",
        "pass": True,
        "issues": [],
        "measurements": {
            "photo_count": len(paths),
            "embedding_count": 0,
        },
        "notes": "",
    }
    if len(paths) < 2:
        result["notes"] = "need_at_least_2_photos"
        return result

    app = _get_identity_app()
    if app is None:
        result["notes"] = "identity_consistency_checker_unavailable"
        return result

    try:
        import cv2
    except Exception as exc:
        result["notes"] = f"identity_consistency_dependencies_unavailable: {exc}"
        return result

    embeddings: dict[str, Sequence[float]] = {}
    poses: dict[str, Sequence[float]] = {}
    face_counts: dict[str, int] = {}
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            face_counts[path.name] = 0
            continue
        faces = app.get(img)
        face_counts[path.name] = len(faces)
        if not faces:
            continue
        face = max(
            faces,
            key=lambda f: float(
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            ),
        )
        embeddings[path.name] = face.normed_embedding
        pose = getattr(face, "pose", None)
        if pose is not None:
            poses[path.name] = pose

    summary = summarize_reference_identity_embeddings(embeddings)
    summary["pose_diversity"] = summarize_reference_pose_measurements(
        poses,
        photo_count=len(paths),
    )
    summary["measurements"].update({
        "photo_count": len(paths),
        "face_counts": face_counts,
    })
    return summary


def assess_reference_diversity(
    photo_paths: Sequence[str | Path],
    min_unique: int,
) -> dict:
    """Reject exact and near-duplicate uploads before assigning view roles."""
    paths = [Path(path) for path in photo_paths]
    result = {
        "status": "unchecked",
        "pass": True,
        "issues": [],
        "measurements": {
            "photo_count": len(paths),
            "min_unique": int(min_unique),
            "duplicate_pairs": [],
        },
        "notes": "",
    }

    hashes: list[tuple[Path, tuple[bool, ...]]] = []
    for path in paths:
        try:
            with Image.open(path) as source:
                gray = ImageOps.exif_transpose(source).convert("L").resize(
                    (REFERENCE_DUPLICATE_HASH_SIZE, REFERENCE_DUPLICATE_HASH_SIZE)
                )
                flattened = getattr(gray, "get_flattened_data", None)
                pixels = list(flattened() if flattened else gray.getdata())
        except Exception:
            continue
        average = sum(pixels) / max(len(pixels), 1)
        hashes.append((path, tuple(pixel >= average for pixel in pixels)))

    local_features: dict[Path, tuple[object, object, tuple[int, int]]] = {}
    try:
        import cv2
        import numpy as np

        orb = cv2.ORB_create(nfeatures=1_200, fastThreshold=10)
        for path, _image_hash in hashes:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            height, width = gray.shape[:2]
            scale = min(1.0, 960.0 / max(height, width))
            if scale < 1.0:
                gray = cv2.resize(
                    gray,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is not None and len(keypoints) >= 32:
                points = np.float32([keypoint.pt for keypoint in keypoints])
                local_features[path] = (points, descriptors, gray.shape[:2])
    except Exception:
        cv2 = None
        np = None

    def local_derivative_match(left_path: Path, right_path: Path) -> dict | None:
        if cv2 is None or np is None:
            return None
        left = local_features.get(left_path)
        right = local_features.get(right_path)
        if left is None or right is None:
            return None
        left_points, left_descriptors, left_shape = left
        right_points, right_descriptors, right_shape = right
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
        good = [
            first for pair in pairs if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.72 * second.distance
        ]
        if len(good) < REFERENCE_LOCAL_MATCH_MIN_INLIERS:
            return None
        source = np.float32([left_points[match.queryIdx] for match in good])
        target = np.float32([right_points[match.trainIdx] for match in good])
        _matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if mask is None:
            return None
        inliers = mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        inlier_ratio = inlier_count / max(len(good), 1)
        if inlier_count < REFERENCE_LOCAL_MATCH_MIN_INLIERS:
            return None

        def coverage(points, shape) -> float:
            selected = points[inliers]
            if len(selected) < 3:
                return 0.0
            x, y, width, height = cv2.boundingRect(selected)
            image_height, image_width = shape
            return (width * height) / max(float(image_width * image_height), 1.0)

        min_coverage = min(
            coverage(source, left_shape),
            coverage(target, right_shape),
        )
        if (
            inlier_ratio < REFERENCE_LOCAL_MATCH_MIN_INLIER_RATIO
            or min_coverage < REFERENCE_LOCAL_MATCH_MIN_COVERAGE
        ):
            return None
        return {
            "local_good_matches": len(good),
            "local_inlier_count": inlier_count,
            "local_inlier_ratio": round(inlier_ratio, 4),
            "local_min_coverage": round(min_coverage, 4),
        }

    duplicate_indexes: set[int] = set()
    duplicate_pairs: list[dict] = []
    for right in range(len(hashes)):
        for left in range(right):
            distance = sum(
                1 for a, b in zip(hashes[left][1], hashes[right][1]) if a != b
            )
            local_match = local_derivative_match(
                hashes[left][0], hashes[right][0]
            )
            if (
                distance <= REFERENCE_DUPLICATE_HAMMING_THRESHOLD
                or local_match is not None
            ):
                duplicate_indexes.add(right)
                pair = {
                    "left": hashes[left][0].name,
                    "right": hashes[right][0].name,
                    "hamming_distance": distance,
                    "match_method": (
                        "perceptual_hash"
                        if distance <= REFERENCE_DUPLICATE_HAMMING_THRESHOLD
                        else "local_features"
                    ),
                }
                if local_match:
                    pair.update(local_match)
                duplicate_pairs.append(pair)
                break

    unique_count = len(hashes) - len(duplicate_indexes)
    result["measurements"].update({
        "readable_count": len(hashes),
        "unique_count": unique_count,
        "duplicate_pairs": duplicate_pairs,
        "hash_size": REFERENCE_DUPLICATE_HASH_SIZE,
        "hamming_threshold": REFERENCE_DUPLICATE_HAMMING_THRESHOLD,
        "local_feature_images": len(local_features),
        "local_match_min_inliers": REFERENCE_LOCAL_MATCH_MIN_INLIERS,
        "local_match_min_inlier_ratio": REFERENCE_LOCAL_MATCH_MIN_INLIER_RATIO,
        "local_match_min_coverage": REFERENCE_LOCAL_MATCH_MIN_COVERAGE,
    })
    if duplicate_pairs or unique_count < min_unique:
        result["status"] = "fail"
        result["pass"] = False
        result["issues"].append("duplicate_reference")
        result["notes"] = (
            f"Only {unique_count} distinct reference photos were found; "
            f"{min_unique} are required"
        )
    else:
        result["status"] = "pass"
    return result


def _front_expression_duplicate_exception(
    diversity: dict | None,
    pose_diversity: dict | None,
) -> dict:
    """Allow one similar front neutral/smile pair when both side views exist."""
    result = {
        "applied": False,
        "reason": None,
        "front_pair": [],
    }
    if not diversity or diversity.get("pass", True):
        return result
    if not pose_diversity or not pose_diversity.get("pass"):
        return result
    measurements = diversity.get("measurements") or {}
    duplicate_pairs = measurements.get("duplicate_pairs") or []
    if len(duplicate_pairs) != 1 or int(measurements.get("unique_count") or 0) < 3:
        return result
    assignments = (pose_diversity.get("measurements") or {}).get(
        "role_assignments"
    ) or {}
    expected = {
        str(assignments.get("front") or ""),
        str(assignments.get("front_secondary") or ""),
    }
    actual = {
        str(duplicate_pairs[0].get("left") or ""),
        str(duplicate_pairs[0].get("right") or ""),
    }
    if "" in expected or actual != expected:
        return result
    result.update({
        "applied": True,
        "reason": "front_neutral_smile_pair_with_proven_side_views",
        "front_pair": sorted(actual),
    })
    return result


def summarize_reference_set(
    photo_quality: dict[str, dict],
    min_photos: int,
    identity_consistency: dict | None = None,
    diversity: dict | None = None,
    pose_diversity: dict | None = None,
    primary_roles: Sequence[str] | None = None,
) -> dict:
    records = list(photo_quality.values())
    passed = [r for r in records if r.get("pass")]
    failed = [r for r in records if not r.get("pass")]
    issues: list[str] = []
    diversity_exception = _front_expression_duplicate_exception(
        diversity,
        pose_diversity,
    )
    roles = list(primary_roles or PRIMARY_REFERENCE_ROLES[:min_photos])
    if len(records) < min_photos:
        issues.append(f"need_at_least_{min_photos}_photos")
    if len(passed) < min_photos:
        issues.append(f"need_at_least_{min_photos}_quality_photos")
    role_coverage = []
    for idx, role in enumerate(roles):
        rec = records[idx] if idx < len(records) else None
        role_record = {
            "role": role,
            "filename": rec.get("filename") if rec else None,
            "pass": bool(rec and rec.get("pass")),
            "issues": list(rec.get("issues") or []) if rec else ["missing_reference"],
        }
        blocking_issue = next(
            (
                issue for issue in role_record["issues"]
                if issue in REFERENCE_ISSUE_GUIDANCE
            ),
            None,
        )
        if blocking_issue:
            headline, guidance = REFERENCE_ISSUE_GUIDANCE[blocking_issue]
            role_record["headline"] = headline
            role_record["guidance"] = guidance
        role_coverage.append(role_record)
        if not role_record["pass"]:
            issues.append(f"{role}:needs_quality_reference")
    for rec in failed:
        name = rec.get("filename", "photo")
        for issue in rec.get("issues") or []:
            issues.append(f"{name}:{issue}")
    if identity_consistency and not identity_consistency.get("pass", True):
        for issue in identity_consistency.get("issues") or []:
            issues.append(issue)
    if (
        diversity
        and not diversity.get("pass", True)
        and not diversity_exception["applied"]
    ):
        for issue in diversity.get("issues") or []:
            issues.append(issue)
    if pose_diversity and not pose_diversity.get("pass", True):
        for issue in pose_diversity.get("issues") or []:
            issues.append(issue)
    unique_issues = list(dict.fromkeys(issues))
    passed_gate = not unique_issues
    agent_action = {
        "action": "ACCEPT" if passed_gate else "REQUEST_BETTER_REFERENCE",
        "reason": (
            "reference_quality_pass"
            if passed_gate else unique_issues[0]
        ),
        "state": "INPUT_CHECK",
        "executed": True,
    }
    return {
        "status": "pass" if passed_gate else "request_better_reference",
        "pass": passed_gate,
        "allowed_actions": ["ACCEPT", "REQUEST_BETTER_REFERENCE"],
        "agent_action": agent_action,
        "total_photos": len(records),
        "passed_photos": len(passed),
        "failed_photos": len(failed),
        "min_required": min_photos,
        "primary_roles_required": roles,
        "role_coverage": role_coverage,
        "identity_consistency": identity_consistency,
        "diversity": diversity,
        "diversity_exception": diversity_exception,
        "pose_diversity": pose_diversity,
        "issues": unique_issues,
    }
