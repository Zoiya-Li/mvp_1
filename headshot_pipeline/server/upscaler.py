"""HD upscaler using Real-ESRGAN x2 via ONNX Runtime.

Lazily loads the model on first use. On M2 Pro, typical upscale
time is 3-8 seconds for a 512×768 input → 1024×1536 output.
"""

from __future__ import annotations

import threading
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

from PIL import Image

# Pillow >=9.1 moved resampling constants under Image.Resampling; older versions
# exposed them directly as Image.LANCZOS. Resolve once so the fallback works on
# both (using Image.Lanczos directly crashes on modern Pillow with
# AttributeError, turning a graceful degradation into a 500).
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

# Model path — will be downloaded by rembg or manually placed
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "realesrgan_x2.onnx"

_session: ort.InferenceSession | None = None
_load_lock = threading.Lock()


def _get_session() -> ort.InferenceSession:
    """Lazy-load ONNX model session (thread-safe double-checked locking)."""
    global _session
    if _session is not None:
        return _session
    with _load_lock:
        # Re-check inside the lock — another thread may have loaded it.
        if _session is not None:
            return _session
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")

        if not _MODEL_PATH.exists():
            raise RuntimeError(
                f"Real-ESRGAN model not found at {_MODEL_PATH}. "
                "Download from https://github.com/xinntao/Real-ESRGAN/releases "
                "and place in models/ directory."
            )

        providers = ort.get_available_providers()
        # Prefer CoreML > CPU on Mac, CUDA > CPU elsewhere
        if "CoreMLExecutionProvider" in providers:
            _session = ort.InferenceSession(str(_MODEL_PATH), providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        elif "CUDAExecutionProvider" in providers:
            _session = ort.InferenceSession(str(_MODEL_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            _session = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
        return _session


def upscale_image(input_path: Path, output_path: Path, scale: int = 2) -> tuple[Path, str]:
    """Upscale an image using Real-ESRGAN x2.

    Falls back to Lanczos resampling if the ONNX model is unavailable.

    Returns ``(output_path, method)`` where method is ``"realesrgan_x2"`` (true
    super-resolution) or ``"lanczos_x2"`` (interpolated fallback — produces a
    2× image but synthesises no new detail). The caller records ``method`` so a
    premium HD user / audit trail is never silently handed interpolated output
    labelled as super-resolution.
    """
    img = Image.open(input_path).convert("RGB")

    # Try ONNX upscaling first
    try:
        session = _get_session()
        result = _run_upscale(session, img, scale)
        result.save(output_path, format="PNG")
        return output_path, "realesrgan_x2"
    except Exception as e:
        print(f"⚠ ONNX upscale failed ({e}), falling back to Lanczos")
        # Fallback: Lanczos resampling
        new_w, new_h = img.width * scale, img.height * scale
        result = img.resize((new_w, new_h), _LANCZOS)
        result.save(output_path, format="PNG")
        return output_path, "lanczos_x2"


def _run_upscale(session: ort.InferenceSession, img: Image.Image, scale: int) -> Image.Image:
    """Run Real-ESRGAN inference on a PIL image."""
    # Preprocess: HWC uint8 → CHW float32, normalized to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
    img_np = np.expand_dims(img_np, axis=0)  # NCHW

    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_np})[0]

    # Postprocess: NCHW float32 → HWC uint8
    output = np.squeeze(output, axis=0)  # CHW
    output = np.transpose(output, (1, 2, 0))  # HWC
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(output, mode="RGB")
