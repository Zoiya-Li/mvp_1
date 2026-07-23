"""Identity repair strategies — face-swap fallback and future local restoration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..config import settings
from ..face_swap import FaceSwapper, FaceSwapResult


class FaceSwapRepair:
    """Swap the user's face onto a generated portrait as a final identity-preservation step."""

    def __init__(
        self,
        analysis_app_factory: Callable[[], object | None] | None = None,
        analysis_app_release: Callable[[], None] | None = None,
    ) -> None:
        self._swapper: FaceSwapper | None = None
        self._load_failed: bool = False
        self._analysis_app_factory = analysis_app_factory
        self._analysis_app_release = analysis_app_release

    def _get_swapper(self) -> FaceSwapper | None:
        if self._swapper is not None:
            return self._swapper
        if self._load_failed:
            return None
        if not settings.face_swap_enabled:
            return None
        model_path = settings.face_swap_model_path
        if not model_path.exists():
            print(f"⚠ Face-swap model not found at {model_path}; skipping.")
            self._load_failed = True
            return None
        try:
            print(f"Loading face-swap model from {model_path}...")
            analysis_app = (
                self._analysis_app_factory()
                if self._analysis_app_factory is not None
                else None
            )
            if self._analysis_app_factory is not None and analysis_app is None:
                raise RuntimeError("shared identity analysis model is unavailable")
            self._swapper = FaceSwapper(
                model_path,
                analysis_app=analysis_app,
                providers=["CPUExecutionProvider"],
                analysis_release=self._analysis_app_release,
            )
            print("✓ Face-swap model loaded")
            return self._swapper
        except Exception as exc:
            print(f"⚠ Failed to load face-swap model: {exc}")
            self._load_failed = True
            return None

    def apply(
        self,
        generated_path: str,
        user_photo_paths: list[str],
        title: str,
    ) -> FaceSwapResult:
        """Swap the user's face onto the generated portrait.

        Falls back to the original image if the model is unavailable or no faces
        are detected. The swapped image is written next to the generated image.
        """
        swapper = self._get_swapper()
        if swapper is None:
            return FaceSwapResult(
                output_path=Path(generated_path),
                swapped=False,
                message="Face swap disabled or model unavailable",
            )

        generated = Path(generated_path)
        output_path = generated.with_name(f"{generated.stem}_swapped{generated.suffix}")
        try:
            result = swapper.swap(
                user_photos=user_photo_paths,
                style_image=generated_path,
                output_path=output_path,
            )
            if result.swapped:
                print(f"✓ Face-swapped result saved to {result.output_path}")
            else:
                print(f"⚠ Face swap skipped: {result.message}")
            return result
        except Exception as exc:
            print(f"⚠ Face swap failed: {exc}")
            return FaceSwapResult(
                output_path=generated,
                swapped=False,
                message=f"Face swap failed: {exc}",
            )
        finally:
            # FaceSwapper unloads the large ONNX runtime after each inference.
            # Drop the wrapper too so the next repair reacquires a fresh shared
            # FaceAnalysis instance for detection.
            self._swapper = None

def public_repair_metadata(action: str, result: FaceSwapResult) -> dict:
    """Keep repair metadata useful without exposing local absolute paths."""
    return {
        "action": action,
        "applied": result.swapped,
        "message": result.message,
        "source_face_count": result.source_face_count,
        "target_face_count": result.target_face_count,
        "output_filename": result.output_path.name,
    }
