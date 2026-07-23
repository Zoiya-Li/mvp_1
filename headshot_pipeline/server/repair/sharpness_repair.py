"""Deterministic face-detail repair that cannot alter portrait geometry."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sharpen_face_region(
    input_path: str | Path,
    output_path: str | Path,
    *,
    amount: float = 2.0,
    sigma: float = 1.0,
) -> Path:
    """Apply feathered unsharp masking around the largest detected face."""
    source = Path(input_path)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Could not read image: {source}")

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
        raise ValueError("No face detected for local sharpness repair")

    x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    center = (int(x + width / 2), int(y + height / 2))
    axes = (max(1, int(width * 0.72)), max(1, int(height * 0.78)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    feather = max(9, int(min(width, height) * 0.18))
    if feather % 2 == 0:
        feather += 1
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha = mask.astype(np.float32)[:, :, None] / 255.0
    repaired = (
        sharpened.astype(np.float32) * alpha
        + image.astype(np.float32) * (1.0 - alpha)
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), np.clip(repaired, 0, 255).astype(np.uint8)):
        raise OSError(f"Could not write repaired image: {destination}")
    return destination
