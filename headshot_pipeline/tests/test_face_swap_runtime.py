"""Regression tests for the bounded-memory INSwapper runtime loader."""

from __future__ import annotations

import sys
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.face_swap import FaceSwapper
from server.repair.framing_repair import reframe_small_face_region
from server.repair.sharpness_repair import sharpen_face_region


def test_face_swapper_requires_preextracted_embedding_map(tmp_path):
    model_path = tmp_path / "inswapper_128.onnx"
    model_path.write_bytes(b"model")

    with pytest.raises(FileNotFoundError, match="embedding map"):
        FaceSwapper(model_path, analysis_app=object())


def test_lean_swapper_disables_ort_memory_arenas(tmp_path, monkeypatch):
    model_path = tmp_path / "inswapper_128.onnx"
    model_path.write_bytes(b"model")
    embedding_map = np.eye(4, dtype=np.float32)
    np.save(model_path.with_suffix(".emap.npy"), embedding_map)
    captured = {}

    class FakeSession:
        def __init__(self, path, *, sess_options, providers):
            captured.update({
                "path": path,
                "options": sess_options,
                "providers": providers,
            })

        def get_inputs(self):
            return [
                SimpleNamespace(name="target", shape=[1, 3, 128, 128]),
                SimpleNamespace(name="source", shape=[1, 512]),
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

    import server.face_swap as face_swap_module

    monkeypatch.setattr(face_swap_module.ort, "InferenceSession", FakeSession)
    swapper = FaceSwapper(
        model_path,
        analysis_app=object(),
        providers=["CPUExecutionProvider"],
    )

    runtime = swapper._get_swapper()

    assert captured["path"] == str(model_path)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert captured["options"].enable_cpu_mem_arena is False
    assert captured["options"].enable_mem_pattern is False
    assert runtime.input_names == ["target", "source"]
    assert runtime.output_names == ["output"]
    assert np.array_equal(runtime.emap, embedding_map)


def test_face_sharpness_repair_changes_face_region_without_resizing(
    tmp_path, monkeypatch,
):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, (260, 220, 3), dtype=np.uint8)
    source = cv2.GaussianBlur(source, (0, 0), 2.0)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    cv2.imwrite(str(input_path), source)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[60, 55, 100, 120]])

    import server.repair.sharpness_repair as sharpness_module

    monkeypatch.setattr(
        sharpness_module.cv2, "CascadeClassifier", lambda *_args: FaceCascade()
    )
    sharpen_face_region(input_path, output_path)
    repaired = cv2.imread(str(output_path))

    assert repaired.shape == source.shape
    assert np.array_equal(repaired[0, 0], source[0, 0])
    assert np.mean(np.abs(repaired[80:160, 80:140].astype(int) - source[80:160, 80:140])) > 0


def test_face_sharpness_repair_default_matches_calibrated_minimum():
    assert signature(sharpen_face_region).parameters["amount"].default == 2.0


def test_face_reframe_preserves_canvas_and_moves_toward_face(
    tmp_path, monkeypatch,
):
    cv2 = pytest.importorskip("cv2")
    source = np.zeros((400, 300, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(300, dtype=np.uint8)[None, :]
    input_path = tmp_path / "wide.png"
    output_path = tmp_path / "reframed.png"
    cv2.imwrite(str(input_path), source)

    class FaceCascade:
        def detectMultiScale(self, *_args, **_kwargs):
            return np.array([[120, 70, 80, 80]])

    import server.repair.framing_repair as framing_module

    monkeypatch.setattr(
        framing_module.cv2, "CascadeClassifier", lambda *_args: FaceCascade()
    )
    reframe_small_face_region(input_path, output_path)
    reframed = cv2.imread(str(output_path))

    assert reframed.shape == source.shape
    assert not np.array_equal(reframed, source)
    assert reframed[200, 150, 0] > source[200, 150, 0]
