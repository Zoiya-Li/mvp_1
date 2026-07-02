"""Post-processing service: ID photo cropping, background replacement, upscaling.

CPU-optimized for Mac Mini M2 using:
- mediapipe for face detection
- rembg (U2-Net) for background segmentation
- OpenCV for image operations
"""

from __future__ import annotations

import enum
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# Enums & constants
# ──────────────────────────────────────────────

class IDPhotoSpec(str, enum.Enum):
    one_inch = "1寸"
    two_inch = "2寸"


# Standard sizes at 300 DPI
PHOTO_SPECS = {
    IDPhotoSpec.one_inch: (295, 413),   # 25×35mm
    IDPhotoSpec.two_inch: (413, 579),   # 35×49mm
}

# Standard background colors (RGB)
BG_COLORS = {
    "red":           (255, 40, 40),
    "blue":          (67, 142, 219),
    "white":         (255, 255, 255),
    "gradient_gray": None,  # special handling
}


# ──────────────────────────────────────────────
# Face detection
# ──────────────────────────────────────────────

def detect_face(pil_img: Image.Image) -> dict | None:
    """Detect the primary face and return bounding box + landmarks.

    Returns dict with keys: bbox (x,y,w,h), left_eye, right_eye, nose, center
    or None if no face found.
    """
    import mediapipe as mp

    mp_face = mp.solutions.face_detection
    img_array = np.array(pil_img.convert("RGB"))

    with mp_face.FaceDetection(
        model_selection=1,  # full-range model (up to 5m)
        min_detection_confidence=0.5,
    ) as detector:
        results = detector.process(img_array)

    if not results.detections:
        return None

    # Take the largest / highest-confidence detection
    det = results.detections[0]
    h, w = img_array.shape[:2]

    # Bounding box (relative coords → pixel coords)
    bbox_rel = det.location_data.relative_bounding_box
    bx = int(bbox_rel.xmin * w)
    by = int(bbox_rel.ymin * h)
    bw = int(bbox_rel.width * w)
    bh = int(bbox_rel.height * h)

    # Keypoints (relative coords)
    keypoints = det.location_data.relative_keypoints
    # mp_face.FaceKeyPoint indices: 0=right_eye, 1=left_eye, 2=nose_tip, 3=mouth, 4=right_ear, 5=left_ear
    left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))
    right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
    nose = (int(keypoints[2].x * w), int(keypoints[2].y * h))

    # Eye center = face horizontal center
    eye_center_x = (left_eye[0] + right_eye[0]) // 2
    eye_center_y = (left_eye[1] + right_eye[1]) // 2

    return {
        "bbox": (bx, by, bw, bh),
        "left_eye": left_eye,
        "right_eye": right_eye,
        "nose": nose,
        "center": (eye_center_x, eye_center_y),
    }


# ──────────────────────────────────────────────
# ID photo cropping
# ──────────────────────────────────────────────

def crop_id_photo(pil_img: Image.Image, spec: IDPhotoSpec) -> Image.Image:
    """Crop image to standard ID photo dimensions, face-centered.

    Chinese national standard (GB/T 33529-2017):
    - Head height ≈ 2/3 of photo height
    - Top of head ≈ 1/10 from top edge
    - Centered horizontally on face
    """
    target_w, target_h = PHOTO_SPECS[spec]
    target_ratio = target_w / target_h  # e.g., 295/413 ≈ 0.714

    face = detect_face(pil_img)
    if face is None:
        # Fallback: center crop
        return _center_crop(pil_img, target_w, target_h)

    img_w, img_h = pil_img.size
    bx, by, bw, bh = face["bbox"]
    fc_x, fc_y = face["center"]

    # Estimate head height from face bbox (face bbox is typically 80% of head)
    head_height = bh / 0.8

    # Target photo height such that head is 2/3 of photo
    photo_h_by_head = head_height / (2.0 / 3.0)

    # Target photo width from aspect ratio
    photo_w_by_head = photo_h_by_head * target_ratio

    # Also compute from the full image — don't exceed original
    photo_h_by_img = img_h
    photo_w_by_img = photo_h_by_img * target_ratio

    # Use the smaller of the two (don't upscale)
    if photo_h_by_head <= img_h:
        crop_h = int(photo_h_by_head)
        crop_w = int(photo_w_by_head)
    else:
        crop_h = int(min(photo_h_by_img, img_h))
        crop_w = int(min(crop_h * target_ratio, img_w))

    # Position: center on face horizontally, head top at 1/10 from top
    head_top_est = by - (head_height - bh) * 0.5  # estimate top of head
    crop_top = int(head_top_est - crop_h * 0.10)
    crop_left = int(fc_x - crop_w / 2)

    # Clamp to image bounds
    crop_top = max(0, min(crop_top, img_h - crop_h))
    crop_left = max(0, min(crop_left, img_w - crop_w))

    # If clamping shifted face off-center, adjust
    if crop_left == 0:
        pass  # already at left edge
    elif crop_left == img_w - crop_w:
        pass  # already at right edge

    cropped = pil_img.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))

    # Resize to exact target dimensions
    if cropped.size != (target_w, target_h):
        cropped = cropped.resize((target_w, target_h), Image.LANCZOS)

    return cropped


def _center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Fallback: center crop to target aspect ratio."""
    img_w, img_h = img.size
    target_ratio = target_w / target_h
    img_ratio = img_w / img_h

    if img_ratio > target_ratio:
        # Image is wider — crop width
        new_w = int(img_h * target_ratio)
        left = (img_w - new_w) // 2
        cropped = img.crop((left, 0, left + new_w, img_h))
    else:
        # Image is taller — crop height
        new_h = int(img_w / target_ratio)
        top = (img_h - new_h) // 2
        cropped = img.crop((0, top, img_w, top + new_h))

    return cropped.resize((target_w, target_h), Image.LANCZOS)


# ──────────────────────────────────────────────
# Background replacement
# ──────────────────────────────────────────────

_rembg_session = None
_rembg_lock = threading.Lock()


def _get_rembg_session():
    """Lazy-load rembg session (U2-Net model), thread-safe."""
    global _rembg_session
    if _rembg_session is None:
        with _rembg_lock:
            if _rembg_session is None:
                from rembg import new_session
                _rembg_session = new_session("u2net_human_seg")
    return _rembg_session


def remove_background(pil_img: Image.Image) -> Image.Image:
    """Remove background and return RGBA image with alpha mask."""
    from rembg import remove
    session = _get_rembg_session()
    return remove(pil_img, session=session)


def replace_background(pil_img: Image.Image, color_key: str) -> Image.Image:
    """Replace background with a solid color or gradient.

    Args:
        pil_img: Input RGB or RGBA image
        color_key: One of "red", "blue", "white", "gradient_gray"

    Returns:
        RGB image with replaced background
    """
    # Get foreground with alpha mask
    rgba = remove_background(pil_img.convert("RGB"))

    w, h = rgba.size
    alpha = np.array(rgba.split()[-1])  # alpha channel

    # Feather edges for smooth blending
    alpha_float = alpha.astype(np.float32) / 255.0
    alpha_float = cv2.GaussianBlur(alpha_float, (5, 5), 3)
    alpha_float = np.clip(alpha_float, 0, 1)

    # Create background
    if color_key == "gradient_gray":
        # Top-to-bottom gradient: light gray → medium gray
        top_color = np.array([220, 220, 220], dtype=np.float32)
        bottom_color = np.array([160, 160, 160], dtype=np.float32)
        bg = np.zeros((h, w, 3), dtype=np.float32)
        for row in range(h):
            t = row / max(h - 1, 1)
            bg[row, :] = top_color * (1 - t) + bottom_color * t
    else:
        rgb = BG_COLORS[color_key]
        bg = np.full((h, w, 3), rgb, dtype=np.float32)

    # Composite: foreground over background using alpha
    fg = np.array(rgba.convert("RGB"), dtype=np.float32)
    alpha_3d = alpha_float[:, :, np.newaxis]
    result = fg * alpha_3d + bg * (1 - alpha_3d)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


# ──────────────────────────────────────────────
# Combined post-processing service
# ──────────────────────────────────────────────

class PostProcessService:
    """Orchestrates post-processing operations for generated images."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crop_id_photo(self, image_path: Path, spec: IDPhotoSpec) -> Path:
        """Crop to ID photo dimensions. Returns path to saved result."""
        img = Image.open(image_path)
        result = crop_id_photo(img, spec)
        out_path = self.output_dir / f"{image_path.stem}_crop_{spec.value.replace('寸','in')}.png"
        result.save(out_path, "PNG", dpi=(300, 300))
        return out_path

    def replace_background(self, image_path: Path, color_key: str) -> Path:
        """Replace background. Returns path to saved result."""
        img = Image.open(image_path)
        result = replace_background(img, color_key)
        out_path = self.output_dir / f"{image_path.stem}_bg_{color_key}.png"
        result.save(out_path, "PNG", dpi=(300, 300))
        return out_path

    def crop_and_replace_bg(
        self, image_path: Path, spec: IDPhotoSpec, color_key: str
    ) -> Path:
        """Crop to ID photo dimensions then replace background."""
        img = Image.open(image_path)
        cropped = crop_id_photo(img, spec)
        result = replace_background(cropped, color_key)
        out_path = self.output_dir / f"{image_path.stem}_{spec.value.replace('寸','in')}_{color_key}.png"
        result.save(out_path, "PNG", dpi=(300, 300))
        return out_path
