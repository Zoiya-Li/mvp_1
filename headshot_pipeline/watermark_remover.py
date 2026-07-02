"""
Watermark Remover — Gemini 水印去除 (Template + Spike Verify + Shape Mask + LaMa)
====================================================================================

水印特征：
  - Gemini 生成的 4 角星/闪光图标 (✦ 形状)
  - 半透明白色叠加，alpha ≈ 0.3–0.5
  - 位于图片底部右侧区域（约 70-100% 横向，70-100% 纵向）
  - 位置漂移，尺寸随分辨率缩放

检测策略（两阶段）：
  1. 模板匹配：在底部右侧 30% 区域用正向+反向小模板搜索候选位置
  2. 尖峰验证：在候选位置检查 high-pass 尖峰密度，确认是真实水印

去除策略：
  3. 用形状 mask（带膨胀）覆盖检测位置
  4. Crop-paste LaMa inpainting

依赖：
  - simple-lama-inpainting (pip install simple-lama-inpainting)
  - opencv-python, numpy, Pillow
"""

from __future__ import annotations

import threading
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_TEMPLATE_PATH = _HERE / "wm_template.npy"          # (30,25) alpha template
_SHAPE_MASK_PATH = _HERE / "wm_shape_mask.npy"       # (72,65) binary mask (572-wide ref)

_REF_WIDTH = 572
_SEARCH_FRAC = 0.30
_MIN_SEARCH = 150

# Template matching
_TM_THRESH = 0.55

# Spike verification
_SPIKE_VERIFY_THRESH = 8   # pixels with diff > 8 in high-pass to confirm star
_SPIKE_VERIFY_COUNT = 5    # minimum spike count for verification

# Dilation
_DILATE_KERNEL = 15
_DILATE_ITERS = 6

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------
_lama = None
_tmpl_u8 = None
_tmpl_inv = None
_shape_mask = None
_lama_lock = threading.Lock()
_assets_lock = threading.Lock()


def _load_lama():
    global _lama
    if _lama is None:
        with _lama_lock:
            if _lama is None:
                from simple_lama_inpainting import SimpleLama
                _lama = SimpleLama()
    return _lama


def _load_assets():
    global _tmpl_u8, _tmpl_inv, _shape_mask
    if _tmpl_u8 is None or _shape_mask is None:
        with _assets_lock:
            if _tmpl_u8 is None:
                a = np.load(_TEMPLATE_PATH)
                _tmpl_u8 = (a * 255).astype(np.uint8)
                _tmpl_inv = 255 - _tmpl_u8
            if _shape_mask is None:
                _shape_mask = np.load(_SHAPE_MASK_PATH)
    return _tmpl_u8, _tmpl_inv, _shape_mask


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _find_watermark(gray: np.ndarray) -> tuple[float, tuple[int, int]] | None:
    """双模板匹配 + 尖峰验证。

    Returns:
        (score, (cx, cy)) — center of watermark in image coords, or None.
    """
    tmpl, inv, _ = _load_assets()
    th, tw = tmpl.shape
    h, w = gray.shape

    # Search region: bottom-right 30%
    sh = max(_MIN_SEARCH, int(h * _SEARCH_FRAC))
    sw = max(_MIN_SEARCH, int(w * _SEARCH_FRAC))
    sh = min(sh, h)
    sw = min(sw, w)

    search = gray[-sh:, -sw:]
    if search.shape[0] < th or search.shape[1] < tw:
        return None

    # Template matching (forward + inverse)
    fwd = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    inv_m = cv2.matchTemplate(search, inv, cv2.TM_CCOEFF_NORMED)

    _, fwd_s, _, fwd_loc = cv2.minMaxLoc(fwd)
    _, inv_s, _, inv_loc = cv2.minMaxLoc(inv_m)

    # Pick best
    if fwd_s >= inv_s:
        best_s, best_loc = fwd_s, fwd_loc
    else:
        best_s, best_loc = inv_s, inv_loc

    if best_s < _TM_THRESH:
        return None

    # Convert to image coordinates → center
    img_x = w - sw + best_loc[0]
    img_y = h - sh + best_loc[1]
    cx = img_x + tw // 2
    cy = img_y + th // 2

    # Spike verification at match location
    if not _verify_spikes(gray, cx, cy):
        # High-confidence match (score >= 0.65) passes without verification
        if best_s < 0.65:
            return None

    return best_s, (cx, cy)


def _verify_spikes(gray: np.ndarray, cx: int, cy: int) -> bool:
    """验证候选位置是否有星形尖峰模式。"""
    h, w = gray.shape
    r = 25
    y0 = max(0, cy - r)
    y1 = min(h, cy + r)
    x0 = max(0, cx - r)
    x1 = min(w, cx + r)
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return False

    blurred = cv2.GaussianBlur(region, (7, 7), 0)
    diff = np.abs(region.astype(float) - blurred.astype(float))
    count = np.sum(diff > _SPIKE_VERIFY_THRESH)
    return count >= _SPIKE_VERIFY_COUNT


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

def remove_gemini_watermark(img: Image.Image) -> Image.Image:
    """去除 Gemini 图片中的 ✦ 水印。

    Args:
        img: PIL Image (RGB or RGBA)

    Returns:
        PIL Image with watermark removed
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    if h < 100 or w < 100:
        return img

    has_alpha = arr.shape[2] == 4
    rgb = arr[:, :, :3].copy()

    # Detect
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result = _find_watermark(gray)
    if result is None:
        return img

    _, (cx, cy) = result

    # Build scaled + dilated shape mask
    _, _, shape_mask = _load_assets()
    sm_h_orig, sm_w_orig = shape_mask.shape
    scale = w / _REF_WIDTH

    if scale > 1.05:
        sm_h = max(1, int(sm_h_orig * scale))
        sm_w = max(1, int(sm_w_orig * scale))
        sm = cv2.resize(shape_mask, (sm_w, sm_h), interpolation=cv2.INTER_NEAREST)
    else:
        sm = shape_mask.copy()

    kernel = np.ones((_DILATE_KERNEL, _DILATE_KERNEL), np.uint8)
    sm = cv2.dilate(sm, kernel, iterations=_DILATE_ITERS)
    sm_h, sm_w = sm.shape

    # Place centered on detection
    mask_x = cx - sm_w // 2
    mask_y = cy - sm_h // 2
    full_mask = np.zeros((h, w), dtype=np.uint8)

    y0 = max(0, mask_y)
    y1 = min(h, mask_y + sm_h)
    x0 = max(0, mask_x)
    x1 = min(w, mask_x + sm_w)

    src_y = y0 - mask_y
    src_x = x0 - mask_x
    src_y1 = src_y + (y1 - y0)
    src_x1 = src_x + (x1 - x0)

    if src_y1 > sm.shape[0] or src_x1 > sm.shape[1]:
        return img

    full_mask[y0:y1, x0:x1] = sm[src_y:src_y1, src_x:src_x1]

    if full_mask.sum() == 0:
        return img

    # Crop-paste LaMa
    lama = _load_lama()
    pad = 80
    ys, xs = np.where(full_mask > 0)
    cy0 = max(0, ys.min() - pad)
    cy1 = min(h, ys.max() + pad + 1)
    cx0 = max(0, xs.min() - pad)
    cx1 = min(w, xs.max() + pad + 1)

    crop = rgb[cy0:cy1, cx0:cx1]
    crop_mask = full_mask[cy0:cy1, cx0:cx1]

    lama_result = lama(Image.fromarray(crop), Image.fromarray(crop_mask).convert("L"))
    lama_arr = np.array(lama_result)

    if lama_arr.shape[:2] != crop.shape[:2]:
        lama_result = lama_result.resize((crop.shape[1], crop.shape[0]))
        lama_arr = np.array(lama_result)

    rgb[cy0:cy1, cx0:cx1] = lama_arr

    if has_alpha:
        return Image.fromarray(np.dstack([rgb, arr[:, :, 3]]))
    return Image.fromarray(rgb)


def remove_watermark_from_file(filepath: str | Path) -> str:
    """从文件去除水印并原地覆盖保存。"""
    filepath = Path(filepath)
    img = Image.open(filepath)
    cleaned = remove_gemini_watermark(img)
    cleaned.save(filepath, format="PNG")
    return str(filepath)
