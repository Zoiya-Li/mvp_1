import sys
from pathlib import Path

from PIL import Image, ImageDraw

_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from server.evaluation.set_evaluator import (  # noqa: E402
    EXPECTED_SHOT_IDS,
    _parse_visual_set_review,
    build_set_contact_sheet,
    evaluate_portrait_set,
    judge_visual_portrait_set,
)


def _write_pattern(path: Path, index: int) -> None:
    image = Image.new("RGB", (96, 128), (24 + index * 17, 38, 70))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (6 + index * 7, 10 + index * 3, 38 + index * 7, 85 + index * 2),
        fill=(220, 180 - index * 13, 80 + index * 19),
    )
    draw.line((0, 20 + index * 15, 95, 110 - index * 9), fill="white", width=5)
    image.save(path)


def _set_images(tmp_path: Path, *, selfie_dominated: bool = False) -> list[dict]:
    images = []
    geometry = ["closeup", "medium", "small_face", "medium", "medium", "medium"]
    face_areas = (
        [0.22] * 6 if selfie_dominated else [0.24, 0.12, 0.035, 0.10, 0.14, 0.09]
    )
    center_offsets = [0.03] * 6 if selfie_dominated else [0.04, 0.08, 0.16, 0.07, 0.12, 0.10]
    for index, shot_id in enumerate(EXPECTED_SHOT_IDS):
        path = tmp_path / f"{index}-{shot_id}.png"
        _write_pattern(path, index)
        look = "Look A" if index < 3 else "Look B"
        images.append({
            "image_id": f"img_{index}",
            "storage_path": str(path),
            "prompt_id": shot_id,
            "resemblance": {
                "shot_spec": {
                    "shot_id": shot_id,
                    "wardrobe": f"{look}: continuous outfit family",
                    "narrative": f"story beat {index + 1}",
                },
                "selected_candidate": {
                    "deliverable": True,
                    "gate_status": {"hard_gates_pass": True},
                    "final_judgement": {
                        "scores": {"identity": 9 - (index % 2)},
                        "identity_quality": {
                            "cosine_similarity": 0.63 - index * 0.01,
                        },
                        "local_quality": {
                            "measurements": {
                                "face_area_ratio": face_areas[index],
                                "face_center_dx": center_offsets[index],
                                "geometry_profile": geometry[index],
                            },
                        },
                    },
                },
            },
        })
    return images


def test_complete_varied_six_frame_set_passes(tmp_path):
    report = evaluate_portrait_set(_set_images(tmp_path))

    assert report["pass"] is True
    assert report["hard_failures"] == []
    assert report["diagnostics"]["shot_ids"] == list(EXPECTED_SHOT_IDS)
    assert report["visual_review"]["required"] is True


def test_exact_duplicate_blocks_delivery(tmp_path):
    images = _set_images(tmp_path)
    Path(images[1]["storage_path"]).write_bytes(
        Path(images[0]["storage_path"]).read_bytes()
    )

    report = evaluate_portrait_set(images)

    assert report["pass"] is False
    assert "set_contains_exact_duplicate" in report["hard_failures"]


def test_six_centered_closeups_are_rejected_as_selfie_dominated(tmp_path):
    report = evaluate_portrait_set(_set_images(tmp_path, selfie_dominated=True))

    assert report["pass"] is False
    assert "set_is_selfie_dominated" in report["hard_failures"]


def test_missing_shot_and_look_assignment_block_delivery(tmp_path):
    images = _set_images(tmp_path)
    images[-1]["resemblance"]["shot_spec"]["shot_id"] = "closeup"
    images[-1]["resemblance"]["shot_spec"]["wardrobe"] = "unassigned outfit"

    report = evaluate_portrait_set(images)

    assert report["pass"] is False
    assert "set_shot_coverage_incomplete" in report["hard_failures"]
    assert "set_look_assignment_incomplete" in report["hard_failures"]


def test_contact_sheet_is_stable_two_by_three_review_surface(tmp_path):
    images = _set_images(tmp_path)

    output = build_set_contact_sheet(images, tmp_path / "review.jpg")

    with Image.open(output) as sheet:
        assert sheet.size == (768, 1536)
        assert sheet.mode == "RGB"


def test_visual_review_parser_clamps_scores_and_drops_unknown_values():
    review = _parse_visual_set_review("""
        prefix {
          "scores": {
            "identity_consistency": 12,
            "wardrobe_continuity": 8.4,
            "composition_variety": -2,
            "narrative_coherence": 7,
            "realism": 9,
            "commercial_readiness": 8
          },
          "hard_failures": ["identity_drift", "invented_failure"],
          "retry_shot_ids": ["profile", "unknown"],
          "notes": "Frame five changes identity."
        } suffix
    """)

    assert review["scores"]["identity_consistency"] == 10
    assert review["scores"]["composition_variety"] == 0
    assert review["hard_failures"] == ["identity_drift"]
    assert review["retry_shot_ids"] == ["profile"]
    assert review["blocking"] is False


def test_visual_set_judge_uses_one_contact_sheet_and_stays_diagnostic(tmp_path):
    images = _set_images(tmp_path)

    class Gateway:
        def judge(self, **kwargs):
            assert Path(kwargs["current_image_path"]).is_file()
            assert kwargs["reference_paths"] == []
            return """{
              "scores": {
                "identity_consistency": 9,
                "wardrobe_continuity": 8,
                "composition_variety": 9,
                "narrative_coherence": 8,
                "realism": 9,
                "commercial_readiness": 9
              },
              "hard_failures": [],
              "retry_shot_ids": [],
              "notes": "Coherent set."
            }"""

    review = judge_visual_portrait_set(
        Gateway(), images, contact_sheet_path=tmp_path / "judge-sheet.jpg"
    )

    assert review["status"] == "reviewed"
    assert review["scores"]["realism"] == 9
    assert review["hard_failures"] == []
    assert review["blocking"] is False
