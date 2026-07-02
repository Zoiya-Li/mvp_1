"""InsightFace-based face swap post-processing.

Usage:
    from server.face_swap import FaceSwapper
    swapper = FaceSwapper(model_path='models/inswapper_128.onnx')
    out_path = swapper.swap(user_photos=['selfie.jpg'], style_image='style.jpg', output_path='out.jpg')
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model as get_insightface_model

logger = logging.getLogger(__name__)


@dataclass
class FaceSwapResult:
    output_path: Path
    swapped: bool
    message: str
    source_face_count: int = 0
    target_face_count: int = 0


class FaceSwapper:
    """Lightweight wrapper around InsightFace inswapper."""

    # Face detection size used for the face-analysis model.
    DEFAULT_DET_SIZE = (640, 640)

    def __init__(
        self,
        model_path: Union[str, Path] = "models/inswapper_128.onnx",
        det_size: Tuple[int, int] = DEFAULT_DET_SIZE,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            model_path: path to the inswapper ONNX model.
            det_size: detection input size for FaceAnalysis.
            providers: ONNX Runtime execution providers. Defaults to
                InsightFace defaults (usually tries CUDA then CPU).
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Face-swap model not found: {self.model_path}")

        self.det_size = det_size
        logger.info("Loading InsightFace analysis model...")
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0, det_size=det_size)
        logger.info("Loading inswapper model: %s", self.model_path)
        self._swapper = get_insightface_model(str(self.model_path))

    @staticmethod
    def _load_image(path: Union[str, Path]) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not read image: {p}")
        return img

    @staticmethod
    def _save_image(path: Union[str, Path], img: np.ndarray) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)

    @staticmethod
    def _face_area(face) -> float:
        bbox = face.bbox.astype(int)
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _face_center_distance(face, img_shape: Tuple[int, int]) -> float:
        bbox = face.bbox.astype(float)
        cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        img_cx, img_cy = img_shape[1] / 2.0, img_shape[0] / 2.0
        return float((cx - img_cx) ** 2 + (cy - img_cy) ** 2)

    @classmethod
    def _pick_best_face(cls, faces: List, img_shape: Tuple[int, int]) -> object:
        """Return the largest face closest to the image center."""
        if not faces:
            raise ValueError("No faces provided")
        if len(faces) == 1:
            return faces[0]

        # Rank by area first, break ties by centrality.
        ranked = sorted(
            faces,
            key=lambda f: (-cls._face_area(f), cls._face_center_distance(f, img_shape)),
        )
        return ranked[0]

    def _detect_faces(self, img: np.ndarray) -> List:
        return self._app.get(img)

    def swap(
        self,
        user_photos: Sequence[Union[str, Path]],
        style_image: Union[str, Path],
        output_path: Union[str, Path],
        source_index: Optional[int] = None,
    ) -> FaceSwapResult:
        """Swap the user's face into the style image.

        Args:
            user_photos: one or more reference photos of the same person. The
                largest/closest-to-frontal face will be selected automatically.
            style_image: generated style image where the face should be replaced.
            output_path: where to write the result.
            source_index: if provided, force using this index from ``user_photos``.

        Returns:
            FaceSwapResult metadata.
        """
        if not user_photos:
            raise ValueError("At least one user photo is required")

        target = self._load_image(style_image)
        target_faces = self._detect_faces(target)
        if not target_faces:
            return FaceSwapResult(
                output_path=Path(output_path),
                swapped=False,
                message="No face detected in style image",
                target_face_count=0,
            )
        target_face = self._pick_best_face(target_faces, target.shape)

        source_faces: List[Tuple[np.ndarray, object]] = []
        for photo in user_photos:
            img = self._load_image(photo)
            faces = self._detect_faces(img)
            if faces:
                source_faces.append((img, self._pick_best_face(faces, img.shape)))

        if not source_faces:
            return FaceSwapResult(
                output_path=Path(output_path),
                swapped=False,
                message="No face detected in any user photo",
                target_face_count=len(target_faces),
            )

        if source_index is not None:
            if not 0 <= source_index < len(source_faces):
                raise IndexError(f"source_index {source_index} out of range")
            source_img, source_face = source_faces[source_index]
        else:
            # Pick the largest source face for best identity signal.
            source_img, source_face = max(
                source_faces, key=lambda pair: self._face_area(pair[1])
            )

        result = self._swapper.get(target, target_face, source_face, paste_back=True)
        self._save_image(output_path, result)

        return FaceSwapResult(
            output_path=Path(output_path),
            swapped=True,
            message="Face swapped successfully",
            source_face_count=len(source_faces),
            target_face_count=len(target_faces),
        )
