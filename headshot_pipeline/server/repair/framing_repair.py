"""Deterministic portrait reframing that preserves the generated subject."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def reframe_small_face_region(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_face_ratio: float = 0.025,
    min_crop_scale: float = 0.72,
    max_crop_scale: float = 0.92,
) -> Path:
    """Crop a too-wide portrait toward its detected face at the same aspect ratio."""
    source = Path(input_path)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not read image: {source}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        raise ValueError("No face detected for local framing repair")

    x, y, face_width, face_height = [
        int(value)
        for value in max(
            faces,
            key=lambda face: int(face[2]) * int(face[3]),
        )
    ]
    face_ratio = (face_width * face_height) / float(width * height)
    crop_scale = float(np.sqrt(face_ratio / target_face_ratio))
    crop_scale = max(min_crop_scale, min(max_crop_scale, crop_scale))
    crop_width = max(1, int(width * crop_scale))
    crop_height = max(1, int(height * crop_scale))

    face_center_x = x + face_width / 2
    # Bias downward so a close portrait keeps the neck and shoulders rather
    # than becoming an accidental face-only crop.
    face_center_y = y + face_height / 2 + crop_height * 0.10
    left = max(0, min(width - crop_width, int(face_center_x - crop_width / 2)))
    top = max(0, min(height - crop_height, int(face_center_y - crop_height / 2)))
    crop = image[top : top + crop_height, left : left + crop_width]
    reframed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LANCZOS4)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), reframed):
        raise OSError(f"Could not write reframed image: {destination}")
    return destination
