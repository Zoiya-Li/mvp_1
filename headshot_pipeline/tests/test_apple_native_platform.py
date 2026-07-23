from __future__ import annotations

import base64
import hashlib
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from server import storage
from server.apple_identity import AppleIdentityError, AppleIdentityVerifier
from server.apple_iap import VerifiedAppleTransaction, apple_iap_verifier
from server.config import settings
from server.job_queue import queue
from server.models import (
    Job,
    JobStatus,
    JobType,
    PricingTier,
    SessionState,
    SessionStatus,
    StyleKey,
)
from server.portrait_domain import AppleNotificationRequest, ApplePurchaseClaimRequest
from server.portrait_storage import (
    check_rate_limit,
    claim_apple_transaction,
    create_guest_user,
    create_project,
    credit_balance,
    get_project,
    get_portrait_order,
    grant_support_entitlement,
    has_paid_project_entitlement,
    link_apple_identity,
    list_projects,
    operational_metrics,
    support_project_snapshot,
    user_for_token,
)
from server.router_admin import (
    SupportReason,
    require_support_admin,
    retry_project,
    support_project,
)
from server.router_portrait_v2 import (
    apple_server_notification,
    claim_project_apple_purchase,
)
from fastapi import HTTPException


@pytest.fixture()
def portrait_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB_PATH", tmp_path / "apple-native-test.db")
    storage.init_db()
    yield tmp_path
    storage._DB_PATH = None


def _project(user_id: str, now: datetime):
    return create_project(
        user_id,
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )


def test_device_fingerprint_only_receives_one_welcome_preview(portrait_db):
    now = storage.utcnow()
    first, _ = create_guest_user(now, preview_fingerprint="device-1")
    second, _ = create_guest_user(now, preview_fingerprint="device-1")

    assert credit_balance(first["user_id"]) == 1
    assert credit_balance(second["user_id"]) == 0


def test_rate_limit_is_persistent_and_reports_retry_after(portrait_db):
    now = storage.utcnow()
    assert check_rate_limit(
        scope="test", subject="203.0.113.4", max_calls=2,
        window_seconds=60, now=now,
    ) == (True, 0)
    assert check_rate_limit(
        scope="test", subject="203.0.113.4", max_calls=2,
        window_seconds=60, now=now,
    ) == (True, 0)
    allowed, retry_after = check_rate_limit(
        scope="test", subject="203.0.113.4", max_calls=2,
        window_seconds=60, now=now,
    )
    assert allowed is False
    assert 1 <= retry_after <= 60


def test_apple_identity_merges_guest_workspace_without_bonus_credit(portrait_db):
    now = storage.utcnow()
    account, first_token = create_guest_user(now)
    linked, linked_token, merged = link_apple_identity(
        current_user_id=account["user_id"], subject="apple-subject",
        email="person@example.com", display_name="Person", now=now,
    )
    assert merged is False
    assert linked["account_type"] == "apple"
    assert user_for_token(first_token) is None
    assert user_for_token(linked_token)["user_id"] == account["user_id"]

    guest, _ = create_guest_user(now)
    project = _project(guest["user_id"], now)
    restored, restored_token, merged = link_apple_identity(
        current_user_id=guest["user_id"], subject="apple-subject",
        email=None, display_name=None, now=now,
    )

    assert merged is True
    assert restored["user_id"] == account["user_id"]
    assert user_for_token(linked_token) is None
    assert user_for_token(restored_token)["user_id"] == account["user_id"]
    assert get_project(project["project_id"], account["user_id"]) is not None
    assert list_projects(guest["user_id"]) == []
    assert credit_balance(account["user_id"]) == 1


def test_verified_apple_transaction_is_idempotent_and_project_bound(portrait_db):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    first = _project(user["user_id"], now)
    second = _project(user["user_id"], now)
    values = dict(
        user_id=user["user_id"],
        project_id=first["project_id"],
        transaction_id="2000000123456789",
        original_transaction_id="2000000123456789",
        product_id="portrait_set_6",
        environment="sandbox",
        bundle_id="com.flashshot.app",
        signed_payload="header.payload.signature",
        purchased_at=now,
        now=now,
    )

    order, newly_claimed = claim_apple_transaction(**values)
    repeated, repeated_new = claim_apple_transaction(**values)

    assert newly_claimed is True
    assert repeated_new is False
    assert repeated["order_id"] == order["order_id"]
    with pytest.raises(ValueError, match="already been claimed"):
        claim_apple_transaction(**{**values, "project_id": second["project_id"]})


def _b64url_int(value: int) -> str:
    size = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(size, "big")).rstrip(b"=").decode()


def test_apple_identity_verifier_checks_signature_audience_and_nonce(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = private_key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "test-key",
        "use": "sig",
        "alg": "RS256",
        "n": _b64url_int(public.n),
        "e": _b64url_int(public.e),
    }
    raw_nonce = "a-secure-random-nonce"
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": "https://appleid.apple.com",
            "aud": "com.flashshot.app",
            "sub": "apple-user-1",
            "iat": now,
            "exp": now + 300,
            "nonce": hashlib.sha256(raw_nonce.encode()).hexdigest(),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )
    verifier = AppleIdentityVerifier()
    monkeypatch.setattr(verifier, "_load_jwks", lambda: {"keys": [jwk]})
    monkeypatch.setattr(settings, "apple_client_id", "com.flashshot.app")

    assert verifier.verify(token, raw_nonce)["sub"] == "apple-user-1"
    with pytest.raises(AppleIdentityError, match="nonce mismatch"):
        verifier.verify(token, "wrong-nonce")


@pytest.mark.asyncio
async def test_storekit_claim_promotes_only_after_server_verification(
    portrait_db, monkeypatch,
):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    project = _project(user["user_id"], now)
    session_id = "s_storekit_claim"
    from server.portrait_storage import attach_legacy_session

    attach_legacy_session(
        project_id=project["project_id"], user_id=user["user_id"],
        session_id=session_id, gender="female", status="preview_ready", now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "owner-token")
    state.hero_preview_generated = True
    state.hero_preview_image_id = "img_hero"
    queue._sessions[session_id] = state
    monkeypatch.setattr(settings, "apple_iap_product_id", "portrait_set_6")
    monkeypatch.setattr(settings, "apple_bundle_id", "com.flashshot.app")
    monkeypatch.setattr(settings, "apple_iap_environment", "sandbox")
    monkeypatch.setattr(
        apple_iap_verifier,
        "verify_transaction",
        lambda _signed: VerifiedAppleTransaction(
            transaction_id="tx-route",
            original_transaction_id="tx-route",
            product_id="portrait_set_6",
            bundle_id="com.flashshot.app",
            environment="sandbox",
            purchased_at=datetime.now(timezone.utc),
            revoked_at=None,
        ),
    )
    try:
        response = await claim_project_apple_purchase(
            project["project_id"],
            ApplePurchaseClaimRequest(signed_transaction="header.payload.signature"),
            user=user,
        )
    finally:
        queue._sessions.pop(session_id, None)

    assert response.status == "paid"
    assert response.newly_claimed is True
    assert state.tier == PricingTier.standard


def test_support_admin_is_hidden_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "support_admin_token", "")
    with pytest.raises(HTTPException) as exc:
        require_support_admin("anything")
    assert exc.value.status_code == 404


def test_support_replacement_entitlement_is_idempotent_and_audited(portrait_db):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    project = _project(user["user_id"], now)

    first = grant_support_entitlement(
        user_id=user["user_id"], project_id=project["project_id"],
        reason="Generation failed after a verified purchase", now=now,
    )
    repeated = grant_support_entitlement(
        user_id=user["user_id"], project_id=project["project_id"],
        reason="Repeated support action", now=now,
    )

    assert first["order_id"] == repeated["order_id"]
    assert has_paid_project_entitlement(user["user_id"], project["project_id"])
    assert operational_metrics()["events_24h"]["support_entitlement"] == 1
    snapshot = support_project_snapshot(project["project_id"])
    assert snapshot["orders"][0]["provider"] == "support"
    assert "inspiration_spec_json" not in snapshot["project"]


@pytest.mark.asyncio
async def test_support_retry_resumes_missing_official_theme_shots(
    portrait_db,
    monkeypatch,
):
    resumed = Job(
        "s_interrupted",
        JobType.full_set,
        "resume portrait",
        shot_spec={"shot_id": "seated"},
    )
    monkeypatch.setattr(
        "server.router_admin.support_project_snapshot",
        lambda _project_id: {
            "project": {
                "project_id": "prj_interrupted",
                "source": "official_theme",
                "legacy_session_id": "s_interrupted",
            }
        },
    )
    monkeypatch.setattr(
        queue,
        "retry_failed_jobs",
        AsyncMock(side_effect=ValueError("no in-memory jobs")),
    )
    monkeypatch.setattr(
        queue,
        "_worker",
        SimpleNamespace(
            active_session_id=None,
            provider_readiness={"pass": True},
        ),
    )
    resume = AsyncMock(return_value=[resumed])
    monkeypatch.setattr(queue, "submit_unlock", resume)

    response = await retry_project(
        "prj_interrupted",
        SupportReason(reason="Resume after worker interruption"),
    )

    resume.assert_awaited_once_with("s_interrupted")
    assert response["queued_job_ids"] == [resumed.job_id]


@pytest.mark.asyncio
async def test_support_retry_restores_exact_catalog_direction(
    portrait_db,
    monkeypatch,
):
    resumed = Job(
        "s_catalog_interrupted",
        JobType.full_set,
        "resume exact portrait",
        shot_spec={"shot_id": "seated"},
    )
    shots = [{"shot_id": "closeup"}, {"shot_id": "seated"}]
    monkeypatch.setattr(
        "server.router_admin.support_project_snapshot",
        lambda _project_id: {
            "project": {
                "project_id": "prj_catalog_interrupted",
                "source": "official_theme",
                "theme_id": "thm_concrete",
                "legacy_session_id": "s_catalog_interrupted",
            }
        },
    )
    monkeypatch.setattr(
        "server.router_admin.get_theme",
        lambda _theme_id: {
            "blueprint": {
                "template_id": "jp_f_fresh",
                "shots": shots,
            }
        },
    )
    monkeypatch.setattr(
        queue,
        "retry_failed_jobs",
        AsyncMock(side_effect=ValueError("no in-memory jobs")),
    )
    monkeypatch.setattr(
        queue,
        "_worker",
        SimpleNamespace(
            active_session_id=None,
            provider_readiness={"pass": True},
        ),
    )
    resume = AsyncMock(return_value=[resumed])
    monkeypatch.setattr(queue, "submit_unlock", resume)

    response = await retry_project(
        "prj_catalog_interrupted",
        SupportReason(reason="Resume exact catalog direction"),
    )

    resume.assert_awaited_once_with(
        "s_catalog_interrupted",
        template_id="jp_f_fresh",
        shot_overrides=shots,
    )
    assert response["queued_job_ids"] == [resumed.job_id]


@pytest.mark.asyncio
async def test_support_project_includes_generation_diagnostics(portrait_db):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    project = _project(user["user_id"], now)
    session_id = "s_support_diagnostics"
    from server.portrait_storage import attach_legacy_session

    attach_legacy_session(
        project_id=project["project_id"], user_id=user["user_id"],
        session_id=session_id, gender="female", status="failed", now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "owner")
    state.status = SessionStatus.failed
    state.pipeline_metrics = {
        "generation_attempts": 1,
        "generation_failures": 1,
        "failed_generation_reasons": {"delivery_gate_failed": 1},
    }
    job = Job(
        session_id,
        JobType.hero_preview,
        "portrait prompt",
        prompt_id="hero_closeup",
        shot_spec={"shot_id": "closeup"},
    )
    job.status = JobStatus.failed
    job.error = "No deliverable portrait passed final QA"
    queue._sessions[session_id] = state
    queue._jobs[job.job_id] = job
    storage.save_generation_event(
        event_id=job.job_id,
        session_id=session_id,
        job_id=job.job_id,
        prompt_id=job.prompt_id,
        shot_spec=job.shot_spec,
        status="failed",
        failure_reason="delivery_gate_failed",
        error=job.error,
        result_image_id=None,
        metadata={
            "selected_candidate": {
                "candidate_id": "cand_2",
                "aggregate_score": 7.4,
                "identity_score": 8.6,
                "deliverable": False,
                "gate_status": {
                    "hard_gates_pass": False,
                    "hard_gate_failures": ["severe_quality_failure"],
                },
            },
            "shortlist": [{"candidate_id": "cand_2", "rank": 1}],
            "budget": {
                "initial_candidates_generated": 3,
                "regenerations_used": 1,
                "local_edits_used": 0,
                "identity_repairs_used": 0,
            },
            "final_evaluate": {
                "delivery_gate": {
                    "pass": False,
                    "issues": ["severe_quality_failure", "not_deliverable"],
                }
            },
        },
        created_at=now,
        completed_at=now,
    )
    try:
        snapshot = await support_project(project["project_id"])
    finally:
        queue._sessions.pop(session_id, None)
        queue._jobs.pop(job.job_id, None)

    generation = snapshot["generation"]
    assert generation["session_status"] == "failed"
    assert generation["jobs"][0]["status"] == "failed"
    assert generation["events"][0]["failure_reason"] == "delivery_gate_failed"
    evaluation = generation["events"][0]["evaluation"]
    assert evaluation["selected_candidate"]["candidate_id"] == "cand_2"
    assert evaluation["selected_candidate"]["gate_status"]["hard_gate_failures"] == [
        "severe_quality_failure"
    ]
    assert evaluation["budget"]["regenerations_used"] == 1
    assert evaluation["delivery_gate"]["pass"] is False
    assert generation["pipeline_metrics"]["generation_failures"] == 1


@pytest.mark.asyncio
async def test_retry_failed_jobs_replaces_only_failed_work(portrait_db):
    session_id = "s_retry"
    state = SessionState(session_id, StyleKey.cinematic, "female", "owner")
    failed = Job(session_id, JobType.full_set, "retry prompt", prompt_id="shot-2")
    failed.status = JobStatus.failed
    completed = Job(session_id, JobType.full_set, "done prompt", prompt_id="shot-3")
    completed.status = JobStatus.completed
    queue._sessions[session_id] = state
    queue._jobs[failed.job_id] = failed
    queue._jobs[completed.job_id] = completed
    try:
        replacements = await queue.retry_failed_jobs(session_id)
        queued = queue._queue.get_nowait()
        queue._queue.task_done()
        assert len(replacements) == 1
        assert queued.job_id == replacements[0].job_id
        assert replacements[0].prompt_id == "shot-2"
        assert failed.job_id not in queue._jobs
        assert completed.job_id in queue._jobs
    finally:
        queue._sessions.pop(session_id, None)
        queue._jobs.pop(failed.job_id, None)
        queue._jobs.pop(completed.job_id, None)
        for replacement in locals().get("replacements", []):
            queue._jobs.pop(replacement.job_id, None)


@pytest.mark.asyncio
async def test_refund_notification_revokes_apple_project_entitlement(
    portrait_db, monkeypatch,
):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    project = _project(user["user_id"], now)
    order, _ = claim_apple_transaction(
        user_id=user["user_id"], project_id=project["project_id"],
        transaction_id="tx-refund", original_transaction_id="tx-refund",
        product_id="portrait_set_6", environment="sandbox",
        bundle_id="com.flashshot.app", signed_payload="signed.transaction",
        purchased_at=now, now=now,
    )
    transaction = VerifiedAppleTransaction(
        transaction_id="tx-refund", original_transaction_id="tx-refund",
        product_id="portrait_set_6", bundle_id="com.flashshot.app",
        environment="sandbox", purchased_at=now, revoked_at=now,
    )
    monkeypatch.setattr(
        apple_iap_verifier, "verify_notification",
        lambda _payload: ("refund", transaction),
    )

    response = await apple_server_notification(
        AppleNotificationRequest(signedPayload="header.payload.signature")
    )

    assert response == {"accepted": True}
    assert get_portrait_order(order["order_id"], user["user_id"])["status"] == "refunded"
    assert not has_paid_project_entitlement(user["user_id"], project["project_id"])
    assert operational_metrics()["events_24h"]["iap_refund"] == 1
