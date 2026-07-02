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


MIN_REFERENCE_DIM = 512
MIN_FACE_SIZE = 72
MIN_BLUR_VARIANCE = 35.0
REFERENCE_IDENTITY_CONSISTENCY_MIN_COSINE = 0.35
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

_IDENTITY_APP = None
_IDENTITY_APP_LOAD_FAILED = False


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
    if blur_var < MIN_BLUR_VARIANCE:
        result["issues"].append("too_blurry")

    faces = []
    try:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )
    except Exception as exc:
        result["issues"].append("face_detector_failed")
        result["notes"] = str(exc)[:200]

    face_count = int(len(faces))
    result["measurements"]["face_count"] = face_count
    if face_count == 0:
        result["issues"].append("no_face")
    elif face_count > 1:
        result["issues"].append("multiple_faces")
    else:
        x, y, fw, fh = [int(v) for v in faces[0]]
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

    summary = summarize_reference_identity_embeddings(embeddings)
    summary["measurements"].update({
        "photo_count": len(paths),
        "face_counts": face_counts,
    })
    return summary


def summarize_reference_set(
    photo_quality: dict[str, dict],
    min_photos: int,
    identity_consistency: dict | None = None,
    primary_roles: Sequence[str] | None = None,
) -> dict:
    records = list(photo_quality.values())
    passed = [r for r in records if r.get("pass")]
    failed = [r for r in records if not r.get("pass")]
    issues: list[str] = []
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
        "issues": unique_issues,
    }
