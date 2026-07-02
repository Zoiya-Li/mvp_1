"""Evaluation service for portrait quality assessment.

Encapsulates VLM QA judging, deterministic local CV checks, identity similarity
scoring, and candidate gate evaluation. Extracted from GeminiWorker to keep
evaluation logic testable and reusable without a full worker instance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Module-level constants (mirrored from gemini_worker to avoid circular import)
QUALITY_ACCEPT_THRESHOLD = 8
IDENTITY_PASS_THRESHOLD = 8
IDENTITY_REPAIR_THRESHOLD = 7
IDENTITY_COSINE_ACCEPT_THRESHOLD = 0.45
MAX_IDENTITY_PACK_REFERENCES = 6

QUALITY_JUDGE_PROMPT = """\
You are a strict production QA system for AI professional headshots.

Compare image 1 (generated candidate) against image 2 (the user's real
reference photo). Return ONLY valid JSON with this schema:

{
  "scores": {
    "identity": 0-10,
    "face_quality": 0-10,
    "style_match": 0-10,
    "artifact": 0-10,
    "commercial_readiness": 0-10
  },
  "hard_failures": ["unsafe_content" | "identity_too_low" | "face_distorted" | "no_face" | "wrong_style" | "bad_artifacts"],
  "recommended_action": "accept" | "face_swap" | "retry" | "discard",
  "notes": "one short sentence explaining the main issue"
}

Scoring rules:
- identity: strict same-person judgement. A familiar person should recognize
  them immediately for 8+.
- face_quality: face clarity, natural skin, believable eyes/mouth/nose.
- style_match: whether it matches the requested/template headshot style.
- artifact: 10 means no visible AI artifacts; lower means distortions.
- commercial_readiness: whether this is good enough to show to a paying user.
- Add unsafe_content to hard_failures for nudity, sexual content, minors, hate,
  violence, self-harm, or other content that should not be delivered.

Be strict. Do not wrap the JSON in markdown."""


class EvaluationService:
    """Self-contained portrait evaluation with lazy-loaded identity scorer."""

    def __init__(self) -> None:
        self._identity_app = None
        self._identity_app_load_failed: bool = False

    # ── Identity app lazy loader ──────────────────────────────

    def _get_identity_app(self):
        """Lazy-load InsightFace recognition for local identity scoring."""
        if self._identity_app is not None:
            return self._identity_app
        if self._identity_app_load_failed:
            return None
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._identity_app = app
            return self._identity_app
        except Exception as exc:
            print(f"⚠ Failed to load identity scorer: {exc}")
            self._identity_app_load_failed = True
            return None

    # ── VLM QA judge ──────────────────────────────────────────

    def judge_current_candidate(
        self,
        gateway,
        image_path: str | None = None,
        reference_photo_paths: list[str] | None = None,
    ) -> dict:
        """Ask the model for structured QA and return a normalized dict."""
        try:
            response_text = gateway.judge(
                current_image_path=image_path or "",
                reference_paths=reference_photo_paths or [],
                judge_prompt=QUALITY_JUDGE_PROMPT,
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
                judgement, self._local_image_quality_check(image_path)
            )
            if reference_photo_paths:
                judgement = self._merge_identity_quality(
                    judgement,
                    self._local_identity_similarity_check(
                        image_path, reference_photo_paths
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
                    "artifact": None,
                    "commercial_readiness": None,
                },
                "hard_failures": [],
                "recommended_action": "retry",
                "notes": notes,
            }

        if not text:
            return empty("Empty judge response")

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
            out["scores"]["identity"] = score
            out["scores"]["commercial_readiness"] = score
            out["recommended_action"] = (
                "accept" if score is not None and score >= QUALITY_ACCEPT_THRESHOLD
                else "retry"
            )
            return out

        scores_in = data.get("scores") if isinstance(data.get("scores"), dict) else {}
        scores = {}
        for key in (
            "identity",
            "face_quality",
            "style_match",
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
        }

    # ── Local CV quality checks ───────────────────────────────

    @staticmethod
    def _local_image_quality_check(image_path: str) -> dict:
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
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if min_dim >= 768:
            resolution_score = 10
        elif min_dim >= 512:
            resolution_score = 8
        elif min_dim >= 384:
            resolution_score = 6
        else:
            resolution_score = 3
            result["hard_failures"].append("bad_resolution")

        if blur_var >= 120:
            sharpness_score = 10
        elif blur_var >= 80:
            sharpness_score = 8
        elif blur_var >= 45:
            sharpness_score = 6
        elif blur_var >= 25:
            sharpness_score = 4
            result["hard_failures"].append("too_blurry")
        else:
            sharpness_score = 2
            result["hard_failures"].append("too_blurry")

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

        face_score = 10
        if len(faces) == 0:
            face_score = 3
            result["hard_failures"].append("no_face")
        elif len(faces) > 1:
            face_score = 6
            result["hard_failures"].append("multiple_faces")
        else:
            x, y, fw, fh = [int(v) for v in faces[0]]
            face_area_ratio = (fw * fh) / float(width * height)
            cx = x + fw / 2.0
            cy = y + fh / 2.0
            center_dx = abs(cx - width / 2.0) / width
            center_dy = abs(cy - height / 2.0) / height
            if face_area_ratio < 0.05 or face_area_ratio > 0.55:
                face_score = min(face_score, 7)
                result["hard_failures"].append("face_scale_unusual")
            if center_dx > 0.22 or center_dy > 0.22:
                face_score = min(face_score, 7)
                result["hard_failures"].append("face_off_center")
            result["measurements"].update({
                "face_area_ratio": round(face_area_ratio, 4),
                "face_center_dx": round(center_dx, 4),
                "face_center_dy": round(center_dy, 4),
            })

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
            "blur_variance": round(blur_var, 2),
            "face_count": int(len(faces)),
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
                return None, 0
            faces = app.get(img)
            if not faces:
                return None, 0
            face = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            return face.normed_embedding, len(faces)

        ref_embeddings = []
        ref_face_counts = []
        for path in reference_photo_paths[:MAX_IDENTITY_PACK_REFERENCES]:
            emb, count = best_embedding(path)
            ref_face_counts.append(count)
            if emb is not None:
                ref_embeddings.append(emb)

        gen_embedding, gen_face_count = best_embedding(generated_path)
        result["measurements"] = {
            "reference_count": len(reference_photo_paths[:MAX_IDENTITY_PACK_REFERENCES]),
            "reference_faces_detected": len(ref_embeddings),
            "reference_face_counts": ref_face_counts,
            "generated_face_count": gen_face_count,
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

        score = EvaluationService._identity_cosine_to_score(cosine)
        if score < IDENTITY_REPAIR_THRESHOLD:
            result["hard_failures"].append("identity_too_low")

        result.update({
            "score": score,
            "cosine_similarity": round(cosine, 4),
            "reference_consistency": (
                round(ref_consistency, 4) if ref_consistency is not None else None
            ),
        })
        result["measurements"]["identity_accept_cosine"] = IDENTITY_COSINE_ACCEPT_THRESHOLD
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
                    }
                ],
            },
            "composition": {
                "score": style_score,
                "status": EvaluationService._score_status(style_score),
                "issues": [
                    item for item in failures
                    if item in {"wrong_style", "global_composition_failed"}
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
        commercial = scores.get("commercial_readiness")
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
            identity is not None and identity >= identity_pass_threshold
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
            )
        )
        quality_pass = (
            face_quality is None or face_quality >= QUALITY_ACCEPT_THRESHOLD
        ) and (
            artifact is None or artifact >= QUALITY_ACCEPT_THRESHOLD
        ) and (
            commercial is None or commercial >= QUALITY_ACCEPT_THRESHOLD
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
            "severe_quality_fail": severe_quality_fail,
            "hard_gates_pass": hard_gates_pass,
            "hard_gate_failures": hard_gate_failures,
        }

    @staticmethod
    def _aggregate_quality_score(judgement: dict) -> float:
        scores = judgement.get("scores", {})
        weights = {
            "identity": 0.45,
            "face_quality": 0.20,
            "style_match": 0.20,
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
