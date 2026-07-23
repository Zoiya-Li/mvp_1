"""InsightFace-based face swap post-processing.

Usage:
    from server.face_swap import FaceSwapper
    swapper = FaceSwapper(model_path='models/inswapper_128.onnx')
    out_path = swapper.swap(user_photos=['selfie.jpg'], style_image='style.jpg', output_path='out.jpg')
"""

from __future__ import annotations

import logging
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

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
        analysis_app: FaceAnalysis | None = None,
        embedding_map_path: Union[str, Path, None] = None,
        analysis_release: Callable[[], None] | None = None,
    ) -> None:
        """
        Args:
            model_path: path to the inswapper ONNX model.
            det_size: detection input size for FaceAnalysis.
            providers: ONNX Runtime execution providers. Defaults to CPU for
                the shared VPS runtime.
            analysis_app: optional shared FaceAnalysis instance. Passing the
                evaluator's instance avoids loading buffalo_l twice.
            embedding_map_path: pre-extracted inswapper embedding map. Runtime
                loading fails closed when it is missing, because parsing the
                full ONNX graph here temporarily doubles model memory.
            analysis_release: releases the shared face-analysis sessions after
                source/target detection and before the swap runtime loads.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Face-swap model not found: {self.model_path}")

        self.det_size = det_size
        self.embedding_map_path = Path(
            embedding_map_path
            or self.model_path.with_suffix(".emap.npy")
        )
        self._analysis_release = analysis_release
        self._runtime_providers = runtime_providers = list(
            providers or ["CPUExecutionProvider"]
        )
        if not self.embedding_map_path.exists():
            raise FileNotFoundError(
                "Face-swap embedding map not found: "
                f"{self.embedding_map_path}. Run "
                "deploy/overseas-vps/extract_inswapper_emap.py before startup."
            )
        if analysis_app is None:
            logger.info("Loading InsightFace analysis model...")
            self._app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=runtime_providers,
            )
            self._app.prepare(ctx_id=0, det_size=det_size)
        else:
            logger.info("Reusing shared InsightFace analysis model")
            self._app = analysis_app
        self._swapper: INSwapper | None = None

    def _get_swapper(self) -> INSwapper:
        if self._swapper is None:
            logger.info("Loading inswapper model: %s", self.model_path)
            self._swapper = self._load_lean_swapper(self._runtime_providers)
        return self._swapper

    def _load_lean_swapper(self, providers: Sequence[str]) -> INSwapper:
        """Load inswapper without parsing a second in-memory ONNX graph."""
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=list(providers),
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if len(inputs) != 2 or len(outputs) != 1:
            raise RuntimeError("Configured face-swap model has an unexpected graph")

        swapper = INSwapper.__new__(INSwapper)
        swapper.model_file = str(self.model_path)
        swapper.session = session
        swapper.emap = np.load(self.embedding_map_path)
        swapper.input_mean = 0.0
        swapper.input_std = 255.0
        swapper.input_names = [item.name for item in inputs]
        swapper.output_names = [item.name for item in outputs]
        input_shape = inputs[0].shape
        swapper.input_shape = input_shape
        swapper.input_size = tuple(input_shape[2:4][::-1])
        return swapper

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

        source_faces: List[object] = []
        for photo in user_photos:
            img = self._load_image(photo)
            faces = self._detect_faces(img)
            if faces:
                source_faces.append(self._pick_best_face(faces, img.shape))

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
            source_face = source_faces[source_index]
        else:
            # Pick the largest source face for best identity signal.
            source_face = max(source_faces, key=self._face_area)

        # The detected Face objects retain the embeddings and landmarks needed
        # for swapping. Release buffalo_l before loading INSwapper so both ONNX
        # runtimes never occupy the shared host at the same time.
        self._app = None
        if self._analysis_release is not None:
            self._analysis_release()
        gc.collect()

        swapper = self._get_swapper()
        try:
            result = swapper.get(
                target, target_face, source_face, paste_back=True
            )
            self._save_image(output_path, result)
        finally:
            self._swapper = None
            del swapper
            gc.collect()

        return FaceSwapResult(
            output_path=Path(output_path),
            swapped=True,
            message="Face swapped successfully",
            source_face_count=len(source_faces),
            target_face_count=len(target_faces),
        )
