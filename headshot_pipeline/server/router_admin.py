"""Token-gated operational support endpoints; never consumed by clients."""

from __future__ import annotations

import hmac
import json

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from . import storage
from .config import settings
from .job_queue import queue
from .portrait_storage import (
    get_theme,
    grant_support_entitlement,
    operational_metrics,
    record_operational_event,
    support_project_snapshot,
)


router = APIRouter(prefix="/api/internal/support", tags=["internal-support"])


def require_support_admin(
    x_flashshot_admin: str = Header(default="", alias="X-FlashShot-Admin"),
) -> None:
    expected = settings.support_admin_token
    if len(expected) < 32 or not hmac.compare_digest(x_flashshot_admin, expected):
        raise HTTPException(404, "Not found")


class SupportReason(BaseModel):
    reason: str = Field(min_length=8, max_length=500)


def _support_catalog_direction(project: dict) -> dict:
    theme = get_theme(project.get("theme_id")) if project.get("theme_id") else None
    blueprint = (theme or {}).get("blueprint") or {}
    direction: dict = {}
    if blueprint.get("template_id"):
        direction["template_id"] = str(blueprint["template_id"])
    if isinstance(blueprint.get("shots"), list) and blueprint["shots"]:
        direction["shot_overrides"] = blueprint["shots"]
    return direction


def _compact_generation_evaluation(metadata: dict) -> dict | None:
    """Expose the useful QA decision without returning prompts or image paths."""
    if not isinstance(metadata, dict) or not metadata:
        return None
    selected = metadata.get("selected_candidate") or {}
    final_evaluate = metadata.get("final_evaluate") or {}
    budget = metadata.get("budget") or {}
    result = {
        "selected_candidate": {
            "candidate_id": selected.get("candidate_id"),
            "aggregate_score": selected.get("aggregate_score"),
            "identity_score": selected.get("identity_score"),
            "deliverable": selected.get("deliverable"),
            "gate_status": selected.get("gate_status"),
            "final_judgement": selected.get("final_judgement"),
            "variants": selected.get("variants") or [],
        },
        "shortlist": metadata.get("shortlist") or [],
        "strategy": metadata.get("strategy") or {},
        "candidates": [
            {
                "index": candidate.get("index"),
                "candidate_id": candidate.get("candidate_id"),
                "filename": candidate.get("filename"),
                "aggregate_score": candidate.get("aggregate_score"),
                "gate_status": candidate.get("gate_status"),
                "judgement": candidate.get("judgement"),
                "variants": candidate.get("variants") or [],
                "repair": candidate.get("repair"),
                "local_edit": candidate.get("local_edit"),
            }
            for candidate in (metadata.get("candidates") or [])
            if isinstance(candidate, dict)
        ],
        "provider_invocations": metadata.get("provider_invocations") or [],
        "delivery_gate": final_evaluate.get("delivery_gate"),
        "budget": {
            "initial_candidates_generated": budget.get("initial_candidates_generated"),
            "regenerations_used": budget.get("regenerations_used"),
            "local_edits_used": budget.get("local_edits_used"),
            "identity_repairs_used": budget.get("identity_repairs_used"),
        },
    }
    return result


@router.get("/metrics", dependencies=[Depends(require_support_admin)])
async def support_metrics():
    return {
        **operational_metrics(),
        "queue_length": queue.queue_length(),
        "generation_busy": queue.is_busy,
        "generation_ready": queue.generation_ready,
    }


@router.get("/projects/{project_id}", dependencies=[Depends(require_support_admin)])
async def support_project(project_id: str):
    snapshot = support_project_snapshot(project_id)
    if not snapshot:
        raise HTTPException(404, "Project not found")
    session_id = snapshot["project"].get("legacy_session_id")
    if session_id:
        state = queue.get_session(session_id)
        events = []
        for row in storage.load_generation_events(session_id):
            shot_spec = (
                json.loads(row["shot_spec_json"])
                if row["shot_spec_json"]
                else {}
            )
            metadata = (
                json.loads(row["metadata_json"])
                if row["metadata_json"]
                else {}
            )
            automatic_retry = metadata.get("automatic_retry") or {}
            evaluation = _compact_generation_evaluation(metadata)
            events.append({
                "job_id": row["job_id"],
                "prompt_id": row["prompt_id"],
                "shot_id": shot_spec.get("shot_id"),
                "status": row["status"],
                "failure_reason": row["failure_reason"],
                "automatic_retry_count": automatic_retry.get("count", 0),
                "automatic_retry_reason": automatic_retry.get("reason"),
                "evaluation": evaluation,
                "error": row["error"][:1_000] if row["error"] else None,
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
            })
        snapshot["generation"] = {
            "session_status": state.status.value if state else None,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "status": job.status.value,
                    "prompt_id": job.prompt_id,
                    "shot_id": (job.shot_spec or {}).get("shot_id"),
                    "automatic_retry_count": job.automatic_retry_count,
                    "automatic_retry_reason": job.automatic_retry_reason,
                    "error": job.error[:1_000] if job.error else None,
                }
                for job in queue.get_jobs(session_id)
            ],
            "events": events[-20:],
            "pipeline_metrics": (
                state.to_response().pipeline_metrics if state else {}
            ),
        }
    return snapshot


@router.post("/projects/{project_id}/retry", dependencies=[Depends(require_support_admin)])
async def retry_project(project_id: str, req: SupportReason):
    snapshot = support_project_snapshot(project_id)
    if not snapshot:
        raise HTTPException(404, "Project not found")
    project = snapshot["project"]
    session_id = project.get("legacy_session_id")
    if not session_id:
        raise HTTPException(409, "Project has no generation session")
    if not queue.generation_ready:
        raise HTTPException(503, "Generation worker is unavailable")
    try:
        jobs = await queue.retry_failed_jobs(session_id)
    except ValueError as exc:
        if project.get("source") not in {"catalog", "official_theme"}:
            raise HTTPException(409, str(exc)) from exc
        try:
            jobs = await queue.submit_unlock(
                session_id,
                **_support_catalog_direction(project),
            )
        except (KeyError, ValueError) as resume_exc:
            raise HTTPException(409, str(resume_exc)) from resume_exc
    except KeyError as exc:
        raise HTTPException(409, str(exc)) from exc
    record_operational_event(
        "support_retry",
        project_id=project_id,
        metadata={"reason": req.reason, "job_count": len(jobs)},
        now=storage.utcnow(),
    )
    return {"project_id": project_id, "queued_job_ids": [job.job_id for job in jobs]}


@router.post(
    "/projects/{project_id}/replacement-entitlement",
    dependencies=[Depends(require_support_admin)],
)
async def replacement_entitlement(project_id: str, req: SupportReason):
    snapshot = support_project_snapshot(project_id)
    if not snapshot:
        raise HTTPException(404, "Project not found")
    project = snapshot["project"]
    order = grant_support_entitlement(
        user_id=project["user_id"], project_id=project_id,
        reason=req.reason, now=storage.utcnow(),
    )
    if project.get("legacy_session_id"):
        queue.grant_verified_project_purchase(
            project["legacy_session_id"], order["order_id"],
        )
    return {
        "project_id": project_id,
        "order_id": order["order_id"],
        "status": order["status"],
    }
