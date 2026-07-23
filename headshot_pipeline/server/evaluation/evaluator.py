"""Evaluation service for portrait quality assessment.

Encapsulates VLM QA judging, deterministic local CV checks, identity similarity
scoring, and candidate gate evaluation. Extracted from GeminiWorker to keep
evaluation logic testable and reusable without a full worker instance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import settings
from ..learning import LearningLayer

# Module-level constants (mirrored from gemini_worker to avoid circular import)
QUALITY_ACCEPT_THRESHOLD = 8
IDENTITY_PASS_THRESHOLD = 8
IDENTITY_REPAIR_THRESHOLD = 7
IDENTITY_COSINE_ACCEPT_THRESHOLD = 0.45
MAX_IDENTITY_PACK_REFERENCES = 6

IDENTITY_ATTRIBUTE_PROMPT = """\
The attached images are private identity references of the same consenting
adult. Extract only stable visible portrait attributes needed to prevent an
image generator from changing the person's appearance. Do not infer race,
ethnicity, health, religion, sexuality, or personality. Return ONLY JSON:

{
  "eyewear": "none" | "clear_glasses" | "sunglasses" | "uncertain",
  "hair_length": "very_short" | "short" | "medium" | "long" | "uncertain",
  "hair_color": "one short visible color description" | "uncertain",
  "facial_hair": "none" | "present" | "uncertain",
  "distinctive_marks": ["short visible mark and location"],
  "apparent_age_band": "adult_18_24" | "adult_25_34" | "adult_35_49" | "adult_50_plus" | "uncertain"
}

Resolve temporary styling differences conservatively. If an attribute is not
clear across the references, return uncertain. Do not wrap JSON in markdown."""

QUALITY_JUDGE_PROMPT = """\
You are a strict production QA system for commercial AI portrait photography.

Image 1 is the generated candidate. Image 2, when present, is one identity
reference of the same consenting adult. Judge composition and realism from
Image 1 only; use Image 2 only to verify stable identity attributes such as
eyewear, hairline, moles, apparent age, and face shape. Return ONLY valid JSON
with this schema:

{
  "scores": {
    "identity": null,
    "face_quality": 0-10,
    "style_match": 0-10,
    "realism": 0-10,
    "artifact": 0-10,
    "commercial_readiness": 0-10
  },
  "composition": {
    "lowest_visible_landmark": "shoulders" | "chest" | "waist" | "hips" | "thighs" | "knees" | "feet",
    "matches_shot_spec": true | false
  },
  "hard_failures": ["unsafe_content" | "identity_too_low" | "identity_attribute_changed" | "identity_geometry_drift" | "skin_over_smoothed" | "face_distorted" | "no_face" | "wrong_style" | "wrong_composition" | "anti_selfie_composition" | "synthetic_appearance" | "bad_artifacts"],
  "recommended_action": "accept" | "face_swap" | "retry" | "discard",
  "notes": "one short sentence explaining the main issue"
}

Scoring rules:
- identity: always null. Identity is scored independently by a local face model.
- Add identity_attribute_changed when Image 1 invents or removes eyeglasses,
  materially changes the hairline, moles, apparent age, or another obvious
  identity attribute visible in Image 2. Do not use Image 2's crop, pose,
  clothing, background, or lighting to judge composition or style.
- face_quality: face clarity, natural skin, believable eyes/mouth/nose.
- style_match: whether it matches the requested style, framing, pose, lighting,
  and lens. Add wrong_composition when the visible crop or pose contradicts the
  supplied ShotSpec, even when the image is otherwise attractive.
- For a Hero Preview, add anti_selfie_composition when the face dominates the
  canvas like an arm's-length phone selfie, the camera appears too close or
  wide-angle distorted, or there is no readable photographic environment.
- realism: 10 means it is visually indistinguishable from a real camera photo
  on close inspection. Penalize plastic or uniformly smoothed skin, generic
  model-face normalization, enlarged eyes, narrowed jaw, perfect bilateral
  symmetry, empty gaze, incoherent hair detail, fake bokeh, halos, dreamy haze,
  and overly centered ID-photo framing. Add synthetic_appearance when realism
  is below 8 or any of these cues is severe.
- For an environmental or editorial target, a blank wall, smooth two-tone or
  charcoal-to-gray gradient, abstract blur field, and generic studio backdrop
  are NOT a readable environment. Add wrong_composition even when the face is
  attractive. Do not infer a location from lighting alone: require visible,
  physically coherent architecture, furniture, street detail, or landscape.
- Treat uniformly airbrushed skin, repeated pore-like noise, perfectly smooth
  clothing/background transitions, and generic corporate-stock-photo facial
  normalization as synthetic cues. A technically clean image can still fail
  realism and commercial readiness.
- artifact: 10 means no visible AI artifacts; lower means distortions.
- commercial_readiness: whether this is good enough to show to a paying user.
- Add unsafe_content to hard_failures for nudity, sexual content, minors, hate,
  violence, self-harm, or other content that should not be delivered.

Before judging composition, objectively identify the lowest visible body
landmark in the generated candidate. Use only what is visible in this one image;
do not infer a close-up from face size, portrait style, or any absent reference
photo. A half-body or three-quarter portrait can place the face in the upper
third while showing the torso, hands, hips, or thighs below it.

Be strict. Do not wrap the JSON in markdown."""


class EvaluationService:
    """Self-contained portrait evaluation with lazy-loaded identity scorer.

    When ``learning_layer`` is provided, thresholds are read from the calibrated
    policy store rather than hard-coded constants.
    """

    def __init__(self, learning_layer: LearningLayer | None = None) -> None:
        self._identity_app = None
        self._identity_app_load_failed: bool = False
        self._learning_layer = learning_layer

    def _get_identity_thresholds(self, shot_spec: dict | None = None) -> dict:
        """Return thresholds, preferring calibrated values when available."""
        from ..gemini_worker import identity_threshold_profile
        thresholds = identity_threshold_profile(shot_spec)
        if self._learning_layer is not None:
            cal = self._learning_layer.get_calibration()
            # Apply learned deltas to the geometry-specific profile. Replacing
            # a small-face threshold with the global close-up threshold would
            # make full-body shots impossible to pass.
            if cal.sample_count >= 10:
                thresholds["identity_pass_threshold"] = round(
                    float(thresholds["identity_pass_threshold"])
                    + (cal.identity_pass_threshold - 8.0),
                    3,
                )
                thresholds["identity_repair_threshold"] = round(
                    float(thresholds["identity_repair_threshold"])
                    + (cal.identity_repair_threshold - 7.0),
                    3,
                )
                thresholds["calibration_sample_count"] = cal.sample_count
                thresholds["calibrated"] = True
        return thresholds

    # ── Identity app lazy loader ──────────────────────────────

    def _get_identity_app(self):
        """Lazy-load InsightFace recognition for local identity scoring."""
        if self._identity_app is not None:
            return self._identity_app
        if self._identity_app_load_failed:
            return None
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._identity_app = app
            return self._identity_app
        except Exception as exc:
            print(f"⚠ Failed to load identity scorer: {exc}")
            self._identity_app_load_failed = True
            return None

    def release_identity_app(self) -> None:
        """Release ONNX face-analysis sessions before a high-memory repair."""
        if self._identity_app is None:
            return
        app = self._identity_app
        self._identity_app = None
        del app
        import gc

        gc.collect()

    def extract_identity_attributes(
        self,
        gateway,
        reference_photo_paths: list[str],
    ) -> dict:
        """Extract a small ephemeral identity contract from up to two references."""
        paths = [
            str(path) for path in reference_photo_paths
            if path and Path(path).is_file()
        ][:2]
        if not paths:
            return {}
        try:
            response = gateway.judge(
                current_image_path=paths[0],
                reference_paths=paths[1:2],
                judge_prompt=IDENTITY_ATTRIBUTE_PROMPT,
                timeout=180,
            )
            match = re.search(r"\{.*\}", response or "", flags=re.S)
            data = json.loads(match.group(0) if match else response)
        except Exception as exc:
            print(f"⚠ Identity attribute extraction skipped: {exc}")
            return {}
        if not isinstance(data, dict):
            return {}
        allowed = {
            "eyewear": {"none", "clear_glasses", "sunglasses", "uncertain"},
            "hair_length": {"very_short", "short", "medium", "long", "uncertain"},
            "facial_hair": {"none", "present", "uncertain"},
            "apparent_age_band": {
                "adult_18_24", "adult_25_34", "adult_35_49",
                "adult_50_plus", "uncertain",
            },
        }
        normalized = {}
        for key, values in allowed.items():
            value = str(data.get(key) or "uncertain").strip().lower()
            normalized[key] = value if value in values else "uncertain"
        color = str(data.get("hair_color") or "uncertain").strip()[:60]
        normalized["hair_color"] = color or "uncertain"
        marks = data.get("distinctive_marks")
        normalized["distinctive_marks"] = [
            str(item).strip()[:100]
            for item in marks[:5]
            if str(item).strip()
        ] if isinstance(marks, list) else []
        return normalized

    @staticmethod
    def identity_attribute_contract(attributes: dict | None) -> str:
        """Render the extracted attributes as hard generation constraints."""
        attributes = attributes or {}
        if not attributes:
            return ""
        eyewear = str(attributes.get("eyewear") or "uncertain")
        eyewear_rule = {
            "none": "The person wears no eyewear. Do not add glasses or sunglasses.",
            "clear_glasses": "Preserve the person's clear prescription glasses.",
            "sunglasses": "Preserve sunglasses only when consistently present in the references.",
        }.get(eyewear, "Do not invent or remove eyewear when the references are unclear.")
        marks = "; ".join(attributes.get("distinctive_marks") or []) or "none recorded"
        return (
            "Stable identity attribute contract (hard constraint):\n"
            f"- eyewear: {eyewear}. {eyewear_rule}\n"
            f"- hair length: {attributes.get('hair_length', 'uncertain')}\n"
            f"- hair color: {attributes.get('hair_color', 'uncertain')}\n"
            f"- facial hair: {attributes.get('facial_hair', 'uncertain')}\n"
            f"- apparent age band: {attributes.get('apparent_age_band', 'uncertain')}\n"
            f"- visible distinctive marks: {marks}\n"
            "Do not beautify away, invent, or reverse any confirmed attribute."
        )

    # ── VLM QA judge ──────────────────────────────────────────

    def judge_current_candidate(
        self,
        gateway,
        image_path: str | None = None,
        reference_photo_paths: list[str] | None = None,
        shot_spec: dict | None = None,
        identity_attributes: dict | None = None,
    ) -> dict:
        """Ask the model for structured QA and return a normalized dict."""
        judge_prompt = QUALITY_JUDGE_PROMPT
        if shot_spec:
            target = {
                key: shot_spec.get(key)
                for key in (
                    "shot_id", "framing", "pose", "environment", "lighting", "lens"
                )
                if shot_spec.get(key)
            }
            judge_prompt += (
                "\n\nComposition target (hard requirement):\n"
                + json.dumps(target, ensure_ascii=False, sort_keys=True)
                + "\nJudge the generated candidate against the central visual intent, "
                "not pixel-perfect wording. Add wrong_composition only when that intent "
                "is visibly absent: a seated shot is not visibly seated, an environmental "
                "shot has no readable location, a turned portrait remains frontal and "
                "symmetric, or a medium shot collapses to a tight head-and-shoulders crop."
            )
        attribute_contract = self.identity_attribute_contract(identity_attributes)
        if attribute_contract:
            judge_prompt += (
                "\n\nExpected stable identity attributes extracted before generation:\n"
                + attribute_contract
                + "\nIf Image 1 contradicts any confirmed attribute, you MUST add "
                "identity_attribute_changed to hard_failures."
            )
        try:
            response_text = gateway.judge(
                current_image_path=image_path or "",
                # The VLM judges the generated frame in isolation. Supplying
                # close-up identity references can make it attribute their crop
                # to the candidate. Identity is measured locally below.
                reference_paths=(reference_photo_paths or [])[:1],
                judge_prompt=judge_prompt,
                timeout=180,
            )
            judgement = self._parse_quality_judge_response(response_text)
            judgement["raw_response"] = response_text
        except Exception as exc:
            judgement = {
                "scores": {
                    "identity": None,
                    "face_quality": None,
                    "style_match": None,
                    "realism": None,
                    "artifact": None,
                    "commercial_readiness": None,
                },
                "hard_failures": ["judge_failed"],
                "recommended_action": "retry",
                "notes": f"Judge failed: {exc}",
                "raw_response": "",
                "error": str(exc),
            }
        if image_path and Path(image_path).exists():
            judgement = self._merge_local_quality(
                judgement, self._local_image_quality_check(image_path, shot_spec)
            )
            if reference_photo_paths:
                judgement = self._merge_identity_quality(
                    judgement,
                    self._local_identity_similarity_check(
                        image_path,
                        reference_photo_paths,
                        shot_spec=shot_spec,
                        cosine_accept_threshold=(
                            settings.hero_identity_cosine_accept_threshold
                            if (shot_spec or {}).get("hero_preview")
                            else None
                        ),
                        reference_calibration_floor=(
                            settings.hero_identity_cosine_reference_floor
                            if (shot_spec or {}).get("hero_preview")
                            else None
                        ),
                    ),
                )
        judgement["quality_evaluation"] = self._quality_evaluation_summary(judgement)
        return judgement

    @staticmethod
    def _parse_quality_judge_response(text: str) -> dict:
        """Parse the structured QA JSON, with a score-regex fallback."""

        def empty(notes: str = "") -> dict:
            return {
                "scores": {
                    "identity": None,
                    "face_quality": None,
                    "style_match": None,
                    "realism": None,
                    "artifact": None,
                    "commercial_readiness": None,
                },
                "hard_failures": [],
                "recommended_action": "retry",
                "notes": notes,
            }

        if not text:
            # An empty VLM response is a judge failure, not a pass: fail closed
            # exactly like the exception path in judge_current_candidate.
            out = empty("Empty judge response")
            out["hard_failures"] = ["judge_failed"]
            return out

        data = None
        stripped = text.strip()
        try:
            data = json.loads(stripped)
        except Exception:
            match = re.search(r"\{.*\}", stripped, flags=re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None

        if not isinstance(data, dict):
            # Local import to avoid circular dependency at module load time.
            from ..gemini_worker import GeminiWorker

            score, feedback = GeminiWorker._parse_judge_response(text)
            out = empty(feedback or "Could not parse structured QA JSON")
            if score is None:
                # Neither structured JSON nor a legacy score: the VLM returned
                # no usable verdict. Fail closed instead of letting an
                # unscored image through the delivery gate.
                out["hard_failures"] = ["judge_failed"]
                return out
            out["scores"]["identity"] = score
            out["scores"]["commercial_readiness"] = score
            out["recommended_action"] = (
                "accept" if score >= QUALITY_ACCEPT_THRESHOLD
                else "retry"
            )
            return out

        scores_in = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        scores = {}
        for key in (
            "identity",
            "face_quality",
            "style_match",
            "realism",
            "artifact",
            "commercial_readiness",
        ):
            value = scores_in.get(key)
            try:
                value = int(float(value) + 0.5)
                value = max(1, min(10, value))
            except Exception:
                value = None
            scores[key] = value

        realism_source = "vlm"
        if scores["realism"] is None:
            # The VLM did not return a realism score. Keep the local
            # compatibility value for diagnostics and variant ranking, but mark
            # its origin so the delivery gate can refuse to pass on a missing
            # VLM realism verdict (see _candidate_gate_status).
            compatibility_scores = [
                value for value in (
                    scores.get("face_quality"),
                    scores.get("artifact"),
                    scores.get("commercial_readiness"),
                )
                if isinstance(value, (int, float))
            ]
            if compatibility_scores:
                scores["realism"] = min(compatibility_scores)
                realism_source = "local_fallback"

        failures = data.get("hard_failures", [])
        if not isinstance(failures, list):
            failures = []
        failures = [str(x) for x in failures if str(x)]

        action = str(data.get("recommended_action", "retry"))
        if action not in {"accept", "face_swap", "retry", "discard"}:
            action = "retry"

        return {
            "scores": scores,
            "hard_failures": failures,
            "recommended_action": action,
            "notes": str(data.get("notes", ""))[:500],
            "realism_source": realism_source,
        }

    # ── Local CV quality checks ───────────────────────────────

    @staticmethod
    def _local_image_quality_check(
        image_path: str,
        shot_spec: dict | None = None,
    ) -> dict:
        """Deterministic local quality checks for generated portrait images.

        This deliberately avoids identity claims. It only checks things that are
        cheap and objective enough to gate production output: readable image,
        resolution, blur, and whether a plausible single face is detected.
        """
        result = {
            "scores": {
                "face_quality": None,
                "artifact": None,
                "commercial_readiness": None,
            },
            "hard_failures": [],
            "measurements": {},
            "notes": "",
        }
        try:
            import cv2
        except Exception as exc:
            result["notes"] = f"local_quality_unavailable: {exc}"
            return result

        img = cv2.imread(str(image_path))
        if img is None:
            result["scores"].update({
                "face_quality": 1,
                "artifact": 1,
                "commercial_readiness": 1,
            })
            result["hard_failures"].append("unreadable_image")
            result["notes"] = "Could not read generated image"
            return result

        height, width = img.shape[:2]
        min_dim = min(width, height)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        global_blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if min_dim >= 768:
            resolution_score = 10
        elif min_dim >= 512:
            resolution_score = 8
        elif min_dim >= 384:
            resolution_score = 6
        else:
            resolution_score = 3
            result["hard_failures"].append("bad_resolution")

        faces = []
        try:
            cascade_path = (
                Path(cv2.data.haarcascades)
                / "haarcascade_frontalface_default.xml"
            )
            cascade = cv2.CascadeClassifier(str(cascade_path))
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(80, 80),
            )
        except Exception:
            faces = []

        raw_face_count = len(faces)
        if raw_face_count > 1:
            # Haar occasionally reports an ear or eye cluster as a second,
            # much smaller face, or emits two nearby boxes for the same face
            # after a portrait is upscaled. Retain substantial, spatially
            # distinct detections; VLM and InsightFace still guard extra people.
            face_boxes = [tuple(int(value) for value in face) for face in faces]
            largest = max(
                face_boxes,
                key=lambda face: int(face[2]) * int(face[3]),
            )
            largest_area = int(largest[2]) * int(largest[3])
            largest_center = (
                int(largest[0]) + int(largest[2]) / 2,
                int(largest[1]) + int(largest[3]) / 2,
            )
            duplicate_radius = max(int(largest[2]), int(largest[3])) * 0.75
            filtered_faces = [largest]
            for face in face_boxes:
                if face == largest:
                    continue
                area = int(face[2]) * int(face[3])
                center = (
                    int(face[0]) + int(face[2]) / 2,
                    int(face[1]) + int(face[3]) / 2,
                )
                center_distance = (
                    (center[0] - largest_center[0]) ** 2
                    + (center[1] - largest_center[1]) ** 2
                ) ** 0.5
                if area >= largest_area * 0.55 and center_distance > duplicate_radius:
                    filtered_faces.append(face)
            faces = filtered_faces

        face_score = 10
        face_blur_var = None
        face_sharpness = None
        if len(faces) == 0:
            face_score = 3
            result["hard_failures"].append("no_face")
        elif len(faces) > 1:
            face_score = 6
            result["hard_failures"].append("multiple_faces")
        else:
            x, y, fw, fh = [int(v) for v in faces[0]]
            from ..input_quality import _measure_face_sharpness

            face_sharpness = _measure_face_sharpness(gray, (x, y, fw, fh))
            face_blur_var = face_sharpness.get("face_blur_variance")
            face_area_ratio = (fw * fh) / float(width * height)
            cx = x + fw / 2.0
            cy = y + fh / 2.0
            center_dx = abs(cx - width / 2.0) / width
            center_dy = abs(cy - height / 2.0) / height
            shot_text = " ".join(
                str((shot_spec or {}).get(key) or "").lower()
                for key in ("shot_id", "framing")
            )
            if bool((shot_spec or {}).get("hero_preview")):
                min_face_ratio, max_face_ratio = 0.045, 0.24
                max_center_dx, max_center_dy = 0.30, 0.32
                geometry_profile = "hero_editorial"
                result["measurements"]["anti_selfie_face_area_max"] = max_face_ratio
                if face_area_ratio > max_face_ratio:
                    result["hard_failures"].append("anti_selfie_composition")
            elif any(token in shot_text for token in ("environmental", "full_body", "full body")):
                min_face_ratio, max_face_ratio = 0.008, 0.24
                max_center_dx, max_center_dy = 0.34, 0.44
                geometry_profile = "small_face"
            elif any(token in shot_text for token in ("profile", "turned portrait", "side portrait")):
                min_face_ratio, max_face_ratio = 0.035, 0.40
                max_center_dx, max_center_dy = 0.30, 0.34
                geometry_profile = "profile_editorial"
            elif any(token in shot_text for token in ("half_body", "medium", "seated", "candid", "waist", "three-quarter")):
                min_face_ratio, max_face_ratio = 0.02, 0.38
                max_center_dx, max_center_dy = 0.28, 0.36
                geometry_profile = "medium"
            else:
                min_face_ratio, max_face_ratio = 0.05, 0.55
                max_center_dx, max_center_dy = 0.22, 0.22
                geometry_profile = "closeup"
            if face_area_ratio < min_face_ratio or face_area_ratio > max_face_ratio:
                face_score = min(face_score, 7)
                result["hard_failures"].append("face_scale_unusual")
            if center_dx > max_center_dx or center_dy > max_center_dy:
                face_score = min(face_score, 7)
                result["hard_failures"].append("face_off_center")
            result["measurements"].update({
                "face_area_ratio": round(face_area_ratio, 4),
                "face_center_dx": round(center_dx, 4),
                "face_center_dy": round(center_dy, 4),
                "geometry_profile": geometry_profile,
                "face_area_range": [min_face_ratio, max_face_ratio],
                "face_center_tolerance": [max_center_dx, max_center_dy],
            })

        # Portraits intentionally contain smooth skin and shallow-depth-of-field
        # backgrounds, so whole-frame Laplacian variance systematically labels
        # good studio work as blurry. When there is one usable face, measure the
        # face crop instead; retain the global value for diagnostics and fall
        # back to it when a face crop is unavailable.
        if face_sharpness is not None and face_blur_var is not None:
            sharpness_value = face_blur_var
            sharpness_source = "face_crop_256"
            failed_metrics = face_sharpness.get("failed_metrics") or []
            if not face_sharpness.get("pass"):
                sharpness_score = 2
                result["hard_failures"].append("too_blurry")
            elif (
                face_blur_var >= 100
                and float(face_sharpness.get("face_tenengrad") or 0) >= 3_000
                and float(face_sharpness.get("face_edge_density") or 0) >= 0.05
            ):
                sharpness_score = 10
            elif not failed_metrics:
                sharpness_score = 8
            else:
                # The normalized face-sharpness contract is 2-of-3 metrics.
                # A passing face must clear the delivery threshold even when
                # one noise-sensitive metric (usually Laplacian) disagrees.
                sharpness_score = 8
        else:
            sharpness_value = global_blur_var
            sharpness_source = "global"
            if sharpness_value >= 120:
                sharpness_score = 10
            elif sharpness_value >= 80:
                sharpness_score = 8
            elif sharpness_value >= 45:
                sharpness_score = 6
            elif sharpness_value >= 25:
                sharpness_score = 4
                result["hard_failures"].append("too_blurry")
            else:
                sharpness_score = 2
                result["hard_failures"].append("too_blurry")

        face_quality = min(resolution_score, sharpness_score, face_score)
        artifact = min(resolution_score, sharpness_score)
        commercial = min(face_quality, artifact)

        result["scores"].update({
            "face_quality": face_quality,
            "artifact": artifact,
            "commercial_readiness": commercial,
        })
        result["measurements"].update({
            "width": width,
            "height": height,
            "min_dim": min_dim,
            "blur_variance": round(global_blur_var, 2),
            "face_blur_variance": (
                round(face_blur_var, 2) if face_blur_var is not None else None
            ),
            "face_tenengrad": (
                face_sharpness.get("face_tenengrad") if face_sharpness else None
            ),
            "face_edge_density": (
                face_sharpness.get("face_edge_density") if face_sharpness else None
            ),
            "sharpness_failed_metrics": (
                face_sharpness.get("failed_metrics") if face_sharpness else []
            ),
            "sharpness_value": round(sharpness_value, 2),
            "sharpness_metric_source": sharpness_source,
            "face_count": int(len(faces)),
            "raw_face_count": int(raw_face_count),
        })
        if result["hard_failures"]:
            result["notes"] = ", ".join(result["hard_failures"])
        return result

    @staticmethod
    def _merge_local_quality(judgement: dict, local_quality: dict) -> dict:
        """Merge deterministic local quality gates into VLM QA output."""
        merged = dict(judgement)
        scores = dict(merged.get("scores") or {})
        local_scores = local_quality.get("scores") or {}
        for key in ("face_quality", "artifact", "commercial_readiness"):
            local_value = local_scores.get(key)
            if local_value is None:
                continue
            current = scores.get(key)
            scores[key] = local_value if current is None else min(current, local_value)
        merged["scores"] = scores

        failures = list(merged.get("hard_failures") or [])
        for failure in local_quality.get("hard_failures") or []:
            if failure not in failures:
                failures.append(failure)
        merged["hard_failures"] = failures
        merged["local_quality"] = local_quality

        severe = {
            "unreadable_image",
            "no_face",
            "multiple_faces",
            "bad_resolution",
            "too_blurry",
        }
        if severe.intersection(failures):
            merged["recommended_action"] = "discard"
            note = merged.get("notes", "")
            local_note = local_quality.get("notes", "")
            merged["notes"] = (
                f"{note} Local gate: {local_note}".strip()
                if note else f"Local gate: {local_note}"
            )[:500]
        merged["quality_evaluation"] = EvaluationService._quality_evaluation_summary(merged)
        return merged

    # ── Identity similarity ───────────────────────────────────

    def _local_identity_similarity_check(
        self,
        generated_path: str,
        reference_photo_paths: list[str],
        shot_spec: dict | None = None,
        cosine_accept_threshold: float | None = None,
        reference_calibration_floor: float | None = None,
    ) -> dict:
        """Score same-person similarity using local face embeddings."""
        result = {
            "score": None,
            "cosine_similarity": None,
            "reference_consistency": None,
            "hard_failures": [],
            "measurements": {},
            "notes": "",
        }
        app = self._get_identity_app()
        if app is None:
            result["notes"] = "identity_scorer_unavailable"
            return result

        try:
            import cv2
            import numpy as np
        except Exception as exc:
            result["notes"] = f"identity_dependencies_unavailable: {exc}"
            return result

        def best_embedding(path: str):
            img = cv2.imread(str(path))
            if img is None:
                return None, 0, None
            faces = app.get(img)
            if not faces:
                return None, 0, None
            face = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            appearance = None
            keypoints = getattr(face, "kps", None)
            if keypoints is not None and len(keypoints) >= 5:
                points = np.asarray(keypoints, dtype=np.float32)
                left_eye, right_eye, nose, left_mouth, right_mouth = points[:5]
                eye_mid = (left_eye + right_eye) / 2.0
                mouth_mid = (left_mouth + right_mouth) / 2.0
                eye_distance = float(np.linalg.norm(right_eye - left_eye))
                appearance = {
                    "inter_eye_to_face_width": eye_distance / width,
                    "eye_to_nose_to_face_height": float(
                        np.linalg.norm(nose - eye_mid)
                    ) / height,
                    "nose_to_mouth_to_face_height": float(
                        np.linalg.norm(mouth_mid - nose)
                    ) / height,
                    "mouth_to_face_width": float(
                        np.linalg.norm(right_mouth - left_mouth)
                    ) / width,
                    "face_width_to_height": width / height,
                    "yaw_proxy": abs(float(nose[0] - eye_mid[0])) / max(
                        eye_distance, 1.0
                    ),
                }

                ix1 = max(0, int(round(x1)))
                iy1 = max(0, int(round(y1)))
                ix2 = min(img.shape[1], int(round(x2)))
                iy2 = min(img.shape[0], int(round(y2)))
                crop = img[iy1:iy2, ix1:ix2]
                if crop.size:
                    gray_face = cv2.cvtColor(
                        cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY,
                    ).astype(np.float32)
                    smooth = cv2.GaussianBlur(gray_face, (0, 0), 1.25)
                    high_frequency = np.abs(gray_face - smooth)
                    cheeks = np.concatenate((
                        high_frequency[108:176, 32:96].reshape(-1),
                        high_frequency[108:176, 160:224].reshape(-1),
                    ))
                    appearance["skin_texture_p75"] = float(
                        np.percentile(cheeks, 75)
                    )
            return face.normed_embedding, len(faces), appearance

        ref_embeddings = []
        ref_appearance = []
        ref_face_counts = []
        for path in reference_photo_paths[:MAX_IDENTITY_PACK_REFERENCES]:
            emb, count, appearance = best_embedding(path)
            ref_face_counts.append(count)
            if emb is not None:
                ref_embeddings.append(emb)
            if appearance:
                ref_appearance.append(appearance)

        gen_embedding, gen_face_count, gen_appearance = best_embedding(generated_path)
        result["measurements"] = {
            "reference_count": len(reference_photo_paths[:MAX_IDENTITY_PACK_REFERENCES]),
            "reference_faces_detected": len(ref_embeddings),
            "reference_face_counts": ref_face_counts,
            "generated_face_count": gen_face_count,
            "appearance_reference_count": len(ref_appearance),
        }

        if gen_embedding is None:
            result["hard_failures"].append("identity_no_generated_face")
            result["notes"] = "No face embedding detected in generated image"
            return result
        if not ref_embeddings:
            result["hard_failures"].append("identity_no_reference_face")
            result["notes"] = "No face embedding detected in reference photos"
            return result

        ref_profile = np.mean(np.stack(ref_embeddings), axis=0)
        ref_profile = ref_profile / max(float(np.linalg.norm(ref_profile)), 1e-8)
        cosine = float(np.dot(gen_embedding, ref_profile))

        ref_consistency = None
        if len(ref_embeddings) >= 2:
            sims = []
            for i in range(len(ref_embeddings)):
                for j in range(i + 1, len(ref_embeddings)):
                    sims.append(float(np.dot(ref_embeddings[i], ref_embeddings[j])))
            ref_consistency = float(sum(sims) / len(sims)) if sims else None

        # Keep the configured threshold as the strict upper bound. Multi-view
        # Hero references may calibrate it only within a narrow explicit range,
        # so pose/lens disagreement cannot cause random rejection while a weak
        # reference pack can never lower the same-person floor indefinitely.
        base_cosine_threshold = (
            float(cosine_accept_threshold)
            if cosine_accept_threshold is not None
            else IDENTITY_COSINE_ACCEPT_THRESHOLD
        )
        if cosine_accept_threshold is None and self._learning_layer is not None:
            cal = self._learning_layer.get_calibration()
            if cal.sample_count >= 10:
                base_cosine_threshold = cal.identity_cosine_accept
        cosine_threshold = base_cosine_threshold
        calibration_floor = None
        if reference_calibration_floor is not None and ref_consistency is not None:
            calibration_floor = min(
                base_cosine_threshold,
                max(0.0, float(reference_calibration_floor)),
            )
            cosine_threshold = min(
                base_cosine_threshold,
                max(calibration_floor, ref_consistency),
            )

        score = EvaluationService._identity_cosine_to_score(cosine)
        if cosine < cosine_threshold:
            result["hard_failures"].append("identity_too_low")
            # Keep the discrete score consistent with the hard gate. Without
            # this cap, a cosine such as 0.74 maps to 10 on the legacy scale
            # while still failing a stricter Hero threshold, leaving the agent
            # with contradictory signals and no regeneration path.
            score = min(score, 7)

        result.update({
            "score": score,
            "cosine_similarity": round(cosine, 4),
            "reference_consistency": (
                round(ref_consistency, 4) if ref_consistency is not None else None
            ),
        })
        result["measurements"].update({
            "identity_accept_cosine": round(cosine_threshold, 4),
            "identity_accept_cosine_base": round(base_cosine_threshold, 4),
            "identity_threshold_reference_calibrated": (
                cosine_threshold < base_cosine_threshold
            ),
        })
        if calibration_floor is not None:
            result["measurements"]["identity_reference_calibration_floor"] = round(
                calibration_floor, 4
            )
        if gen_appearance and ref_appearance:
            geometry_keys = (
                "inter_eye_to_face_width",
                "eye_to_nose_to_face_height",
                "nose_to_mouth_to_face_height",
                "mouth_to_face_width",
                "face_width_to_height",
            )
            generated_yaw = float(gen_appearance.get("yaw_proxy") or 0)
            yaw_closest_reference = min(
                ref_appearance,
                key=lambda profile: abs(
                    float(profile.get("yaw_proxy") or 0)
                    - generated_yaw
                ),
            )
            minimum_yaw_delta = abs(
                float(yaw_closest_reference.get("yaw_proxy") or 0)
                - generated_yaw
            )
            pose_candidates = [
                profile for profile in ref_appearance
                if abs(float(profile.get("yaw_proxy") or 0) - generated_yaw)
                <= minimum_yaw_delta + 0.12
            ]

            def geometry_distance(profile: dict) -> float:
                return sum(
                    abs(
                        float(gen_appearance.get(key) or 0)
                        - float(profile.get(key) or 0)
                    ) / max(abs(float(profile.get(key) or 0)), 0.05)
                    for key in geometry_keys
                )

            closest_reference = min(
                pose_candidates,
                key=lambda profile: (
                    geometry_distance(profile),
                    abs(float(profile.get("yaw_proxy") or 0) - generated_yaw),
                ),
            )
            geometry_thresholds = {
                "inter_eye_to_face_width": 0.12,
                "eye_to_nose_to_face_height": 0.16,
                "nose_to_mouth_to_face_height": 0.16,
                "mouth_to_face_width": 0.18,
                "face_width_to_height": 0.15,
            }
            reference_yaw = float(closest_reference.get("yaw_proxy") or 0)
            yaw_delta = abs(generated_yaw - reference_yaw)
            shot_text = " ".join(
                str((shot_spec or {}).get(key) or "").lower()
                for key in ("shot_id", "framing", "pose")
            )
            geometry_threshold_profile = "frontal"
            if "profile" in shot_text and yaw_delta <= 0.15:
                # Horizontal landmark ratios contract non-linearly as the head
                # turns. Only relax them when a real angled reference has the
                # same measured pose; embedding similarity remains unchanged.
                geometry_thresholds.update({
                    "inter_eye_to_face_width": 0.18,
                    "eye_to_nose_to_face_height": 0.20,
                    "nose_to_mouth_to_face_height": 0.20,
                    "mouth_to_face_width": 0.30,
                    "face_width_to_height": 0.20,
                })
                geometry_threshold_profile = "pose_matched_profile"
            geometry_drift = {}
            for key, threshold in geometry_thresholds.items():
                generated_value = float(gen_appearance.get(key) or 0)
                reference_value = float(closest_reference.get(key) or 0)
                relative_drift = abs(generated_value - reference_value) / max(
                    abs(reference_value), 0.05
                )
                geometry_drift[key] = {
                    "generated": round(generated_value, 4),
                    "reference": round(reference_value, 4),
                    "relative_drift": round(relative_drift, 4),
                    "threshold": threshold,
                    "pass": relative_drift <= threshold,
                }
            failed_geometry = [
                key for key, measurement in geometry_drift.items()
                if not measurement["pass"]
            ]
            result["measurements"].update({
                "appearance_geometry": geometry_drift,
                "appearance_geometry_failed": failed_geometry,
                "appearance_reference_selection": "pose_then_geometry",
                "appearance_reference_candidate_count": len(pose_candidates),
                "appearance_geometry_threshold_profile": geometry_threshold_profile,
                "appearance_yaw_delta": round(yaw_delta, 4),
                "generated_yaw_proxy": round(
                    generated_yaw, 4
                ),
                "matched_reference_yaw_proxy": round(
                    reference_yaw, 4
                ),
            })
            if len(failed_geometry) >= 2:
                result["hard_failures"].append("identity_geometry_drift")

            generated_texture = float(
                gen_appearance.get("skin_texture_p75") or 0
            )
            reference_textures = sorted(
                float(profile.get("skin_texture_p75") or 0)
                for profile in ref_appearance
                if profile.get("skin_texture_p75") is not None
            )
            if reference_textures:
                reference_texture = reference_textures[len(reference_textures) // 2]
                texture_ratio = generated_texture / max(reference_texture, 0.1)
                result["measurements"].update({
                    "generated_skin_texture_p75": round(generated_texture, 4),
                    "reference_skin_texture_p75_median": round(reference_texture, 4),
                    "skin_texture_ratio": round(texture_ratio, 4),
                    "skin_texture_ratio_min": 0.88,
                })
                if reference_texture >= 1.5 and texture_ratio < 0.88:
                    result["hard_failures"].append("skin_over_smoothed")
        if result["hard_failures"]:
            result["notes"] = ", ".join(result["hard_failures"])
        return result

    @staticmethod
    def _identity_cosine_to_score(cosine: float) -> int:
        """Map InsightFace cosine similarity to the existing 1-10 score scale."""
        if cosine >= 0.58:
            return 10
        if cosine >= 0.52:
            return 9
        if cosine >= IDENTITY_COSINE_ACCEPT_THRESHOLD:
            return 8
        if cosine >= 0.40:
            return 7
        if cosine >= 0.34:
            return 6
        if cosine >= 0.28:
            return 5
        return 3

    @staticmethod
    def _merge_identity_quality(judgement: dict, identity_quality: dict) -> dict:
        """Merge local face-embedding identity score into VLM QA output."""
        merged = dict(judgement)
        scores = dict(merged.get("scores") or {})
        local_score = identity_quality.get("score")
        if local_score is not None:
            current = scores.get("identity")
            scores["identity"] = local_score if current is None else min(current, local_score)
        merged["scores"] = scores

        failures = list(merged.get("hard_failures") or [])
        for failure in identity_quality.get("hard_failures") or []:
            if failure not in failures:
                failures.append(failure)
        merged["hard_failures"] = failures
        merged["identity_quality"] = identity_quality

        appearance_failures = {
            "identity_geometry_drift",
            "skin_over_smoothed",
        }.intersection(failures)
        if appearance_failures:
            for key in ("realism", "commercial_readiness"):
                current = scores.get(key)
                scores[key] = 7 if current is None else min(current, 7)
            merged["scores"] = scores
            merged["recommended_action"] = "retry"

        if "identity_too_low" in failures:
            merged["recommended_action"] = "face_swap"
            note = merged.get("notes", "")
            local_note = identity_quality.get("notes", "")
            merged["notes"] = (
                f"{note} Local identity: {local_note}".strip()
                if note else f"Local identity: {local_note}"
            )[:500]
        elif any(f in failures for f in ("identity_no_generated_face", "identity_no_reference_face")):
            merged["recommended_action"] = "retry"
        merged["quality_evaluation"] = EvaluationService._quality_evaluation_summary(merged)
        return merged

    # ── Scoring helpers ───────────────────────────────────────

    @staticmethod
    def _score_status(score, threshold: float = QUALITY_ACCEPT_THRESHOLD) -> str:
        if score is None:
            return "unchecked"
        try:
            return "pass" if float(score) >= threshold else "fail"
        except Exception:
            return "unchecked"

    @staticmethod
    def _quality_evaluation_summary(judgement: dict) -> dict:
        """Return the product QA schema from VLM + local detector results."""
        scores = judgement.get("scores") or {}
        failures = list(judgement.get("hard_failures") or [])
        local = judgement.get("local_quality") or {}
        identity = judgement.get("identity_quality") or {}

        identity_score = scores.get("identity")
        face_quality_score = scores.get("face_quality")
        artifact_score = scores.get("artifact")
        realism_score = scores.get("realism")
        style_score = scores.get("style_match")
        readiness_score = scores.get("commercial_readiness")
        prompt_score = (
            round(style_score / 10.0, 4)
            if isinstance(style_score, (int, float)) else None
        )
        aesthetic_source_scores = [
            value for value in (style_score, readiness_score)
            if isinstance(value, (int, float))
        ]
        aesthetic_score = (
            round(sum(aesthetic_source_scores) / (10.0 * len(aesthetic_source_scores)), 4)
            if aesthetic_source_scores else None
        )

        return {
            "identity": {
                "score": identity_score,
                "status": EvaluationService._score_status(identity_score),
                "cosine_similarity": identity.get("cosine_similarity"),
                "reference_consistency": identity.get("reference_consistency"),
                "issues": [
                    item for item in failures
                    if str(item).startswith("identity_") or item == "identity_too_low"
                ],
                "measurements": identity.get("measurements") or {},
            },
            "face_quality": {
                "score": face_quality_score,
                "status": EvaluationService._score_status(face_quality_score),
                "issues": [
                    item for item in (local.get("hard_failures") or failures)
                    if item in {
                        "unreadable_image",
                        "no_face",
                        "multiple_faces",
                        "too_blurry",
                        "bad_resolution",
                        "skin_over_smoothed",
                        "face_scale_unusual",
                        "face_off_center",
                    }
                ],
                "measurements": local.get("measurements") or {},
            },
            "artifacts": {
                "score": artifact_score,
                "status": EvaluationService._score_status(artifact_score),
                "issues": [
                    item for item in failures
                    if item in {
                        "bad_artifacts",
                        "face_distorted",
                        "too_blurry",
                        "bad_resolution",
                        "skin_over_smoothed",
                        "identity_geometry_drift",
                    }
                ],
            },
            "realism": {
                "score": realism_score,
                "status": EvaluationService._score_status(realism_score),
                "issues": [
                    item for item in failures
                    if item == "synthetic_appearance"
                ],
            },
            "composition": {
                "score": style_score,
                "status": EvaluationService._score_status(style_score),
                "issues": [
                    item for item in failures
                    if item in {"wrong_style", "wrong_composition", "anti_selfie_composition", "global_composition_failed"}
                ],
            },
            "prompt_adherence": {
                "score": prompt_score,
                "status": EvaluationService._score_status(style_score),
            },
            "aesthetic": {
                "score": aesthetic_score,
                "status": EvaluationService._score_status(readiness_score),
            },
            "safety": {
                "status": "fail" if "unsafe_content" in failures else "pass",
                "issues": [
                    item for item in failures
                    if item == "unsafe_content"
                ],
            },
        }

    @staticmethod
    def _candidate_gate_status(
        judgement: dict,
        identity_thresholds: dict | None = None,
    ) -> dict:
        """Evaluate product hard gates before any aesthetic ranking."""
        from ..gemini_worker import identity_threshold_profile

        scores = judgement.get("scores", {}) or {}
        failures = set(judgement.get("hard_failures") or [])
        identity = scores.get("identity")
        face_quality = scores.get("face_quality")
        artifact = scores.get("artifact")
        realism = scores.get("realism")
        commercial = scores.get("commercial_readiness")
        # Realism must come from the VLM verdict itself. The local
        # compatibility value stays available for diagnostics/ranking, but a
        # judgement without a VLM realism score must not pass the quality gate.
        realism_missing = judgement.get("realism_source") == "local_fallback"
        if realism is None:
            compatibility_scores = [
                value for value in (face_quality, artifact, commercial)
                if isinstance(value, (int, float))
            ]
            if compatibility_scores:
                realism = min(compatibility_scores)
            realism_missing = True
        thresholds = identity_thresholds or identity_threshold_profile()
        identity_pass_threshold = float(
            thresholds.get("identity_pass_threshold", IDENTITY_PASS_THRESHOLD)
        )

        safety_pass = "unsafe_content" not in failures
        face_detected = not any(
            failure in failures
            for failure in ("no_face", "identity_no_generated_face")
        )
        identity_pass = (
            identity is not None
            and identity >= identity_pass_threshold
            and "identity_too_low" not in failures
        )
        severe_quality_fail = any(
            failure in failures
            for failure in (
                "face_distorted",
                "bad_artifacts",
                "unreadable_image",
                "bad_resolution",
                "too_blurry",
                "multiple_faces",
                "wrong_composition",
                "anti_selfie_composition",
                "synthetic_appearance",
                "judge_failed",
                "identity_attribute_changed",
                "identity_geometry_drift",
                "skin_over_smoothed",
            )
        )
        quality_accept_threshold = float(
            thresholds.get("quality_accept_threshold", QUALITY_ACCEPT_THRESHOLD)
        )
        realism_accept_threshold = float(
            thresholds.get("realism_accept_threshold", quality_accept_threshold)
        )
        commercial_accept_threshold = float(
            thresholds.get("commercial_accept_threshold", quality_accept_threshold)
        )
        quality_pass = (
            face_quality is None or face_quality >= QUALITY_ACCEPT_THRESHOLD
        ) and (
            artifact is None or artifact >= QUALITY_ACCEPT_THRESHOLD
        ) and (
            not realism_missing
            and realism is not None
            and realism >= realism_accept_threshold
        ) and (
            commercial is None or commercial >= commercial_accept_threshold
        )
        hard_gates_pass = (
            safety_pass
            and face_detected
            and identity_pass
            and quality_pass
            and not severe_quality_fail
        )
        hard_gate_failures = []
        if not safety_pass:
            hard_gate_failures.append("unsafe_content")
        if not face_detected:
            hard_gate_failures.append("no_usable_face_detected")
        if not identity_pass:
            hard_gate_failures.append("identity_fail")
        if not quality_pass:
            hard_gate_failures.append("quality_below_threshold")
        if severe_quality_fail:
            hard_gate_failures.append("severe_quality_failure")
        return {
            "safety_pass": safety_pass,
            "face_detected": face_detected,
            "identity_pass": identity_pass,
            "identity_pass_threshold": identity_pass_threshold,
            "identity_repair_threshold": float(
                thresholds.get(
                    "identity_repair_threshold",
                    IDENTITY_REPAIR_THRESHOLD,
                )
            ),
            "identity_threshold_profile": thresholds.get("profile", "closeup"),
            "quality_pass": quality_pass,
            "quality_accept_threshold": quality_accept_threshold,
            "realism_accept_threshold": realism_accept_threshold,
            "realism_score_missing": realism_missing,
            "commercial_accept_threshold": commercial_accept_threshold,
            "severe_quality_fail": severe_quality_fail,
            "hard_gates_pass": hard_gates_pass,
            "hard_gate_failures": hard_gate_failures,
        }

    @staticmethod
    def _aggregate_quality_score(judgement: dict) -> float:
        scores = judgement.get("scores", {})
        weights = {
            "identity": 0.40,
            "face_quality": 0.15,
            "style_match": 0.15,
            "realism": 0.15,
            "artifact": 0.10,
            "commercial_readiness": 0.05,
        }
        total = 0.0
        used = 0.0
        for key, weight in weights.items():
            value = scores.get(key)
            if value is None:
                continue
            total += float(value) * weight
            used += weight
        if used == 0:
            return 0.0
        penalty = 0.0
        if judgement.get("recommended_action") == "discard":
            penalty -= 1.0
        elif judgement.get("hard_failures"):
            penalty -= 1.5
        return min(10.0, max(0.0, round(total / used + penalty, 2)))
