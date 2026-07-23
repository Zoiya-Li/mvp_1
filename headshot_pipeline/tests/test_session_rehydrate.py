"""Deterministic regression test for session rehydration after a restart (#55).

Proves ``JobQueue.get_session()`` rebuilds a ``SessionState`` from SQLite + disk
on a cache miss, recovering: owner_token (auth), paid tier (revenue), and the
generated-image gallery only when delivery metadata survives — plus orphan
quarantine, intermediate-file filtering, idempotent re-rehydrate, and DB cleanup
on delete.

A "restart" is simulated by clearing the in-memory cache; no network / no Chrome.
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_PIPELINE = Path(__file__).resolve().parents[1]
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

import pytest  # noqa: E402

from server import storage  # noqa: E402
from server.config import settings  # noqa: E402
from server.job_queue import JobQueue, reference_slot_filename  # noqa: E402
from server.models import PricingTier, SessionStatus, StyleKey  # noqa: E402

SID = "s_testdead"
TOK = "TOK_REHYDRATE_TEST_abc"


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    """Point storage + settings at a throwaway data dir."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "output")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    # Reset the cached DB path so storage re-initializes against tmp.
    monkeypatch.setattr(storage, "_DB_PATH", None)
    storage.init_db()

    now = storage.utcnow()
    storage.save_session(SID, TOK, StyleKey.business.value, "male", now)
    storage.update_session_tier(SID, PricingTier.premium.value, 3, "pay_12345")
    storage.update_session_consent(SID, {
        "face_processing_consent": True,
        "adult_subject_confirmed": True,
        "no_training_by_default": True,
        "cross_user_search_prohibited": True,
        "long_term_face_library_prohibited": True,
        "consented_at": now.isoformat(),
        "policy_version": "face_processing_consent_v1",
    })

    # On-disk artifacts (what survives a real restart).
    out_dir = settings.output_dir / SID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "img_aabbccdd.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 50)
    (out_dir / "pp_11223344.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"y" * 50)
    (out_dir / "pp_00000000.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"z" * 50)
    # An intermediate post-process file that must NOT be re-listed as an image.
    (out_dir / "img_aabbccdd_crop_red.png").write_bytes(b"garbage")

    up_dir = settings.upload_dir / SID
    up_dir.mkdir(parents=True, exist_ok=True)
    (up_dir / "selfie1.jpg").write_bytes(b"jpg")
    (up_dir / "selfie2.jpg").write_bytes(b"jpg")

    # Metadata row for the raw generated image (premium resemblance history).
    storage.save_generated_image(
        image_id="img_aabbccdd", session_id=SID, prompt_id="biz_male_01", turn=1,
        revised_image_id=None, parent_image_id=None, operation=None,
        resemblance={
            "iterations": 2,
            "final_score": 8,
            "history": [6, 8],
            "selected_candidate": {
                "candidate_id": "cand_1",
                "deliverable": True,
                "gate_status": {
                    "hard_gates_pass": True,
                    "hard_gate_failures": [],
                },
            },
        },
        created_at=now - timedelta(seconds=2),
    )
    storage.save_generated_image(
        image_id="pp_11223344", session_id=SID, prompt_id="upscale_x2", turn=1,
        revised_image_id=None, parent_image_id="img_aabbccdd",
        operation="upscale_x2",
        resemblance={
            "final_asset": {
                "image_id": "pp_11223344",
                "operation": "upscale_x2",
                "metadata_ai_label": True,
            }
        },
        created_at=now - timedelta(seconds=1),
    )
    storage.save_generated_image(
        image_id="pp_00000000", session_id=SID, prompt_id="upscale_x2", turn=1,
        revised_image_id=None, parent_image_id="img_aabbccdd",
        operation="upscale_x2",
        resemblance={
            "final_asset": {
                "image_id": "pp_00000000",
                "operation": "upscale_x2",
                "metadata_ai_label": True,
            }
        },
        created_at=now,
    )
    storage.save_generation_event(
        event_id="j_completed",
        session_id=SID,
        job_id="j_completed",
        prompt_id="biz_male_01",
        shot_spec={"shot_id": "closeup"},
        status="completed",
        failure_reason=None,
        error=None,
        result_image_id="img_aabbccdd",
        created_at=now,
        completed_at=now,
        metadata={
            "selected_candidate": {"deliverable": True},
            "candidates": [{}, {}, {}],
            "provider_invocations": [
                {
                    "operation": "CREATE_FROM_REFERENCES",
                    "estimated_cost": 0.12,
                    "latency_ms": 1000,
                }
            ],
        },
    )
    storage.save_generation_event(
        event_id="j_failed",
        session_id=SID,
        job_id="j_failed",
        prompt_id="biz_male_02",
        shot_spec={"shot_id": "half_body"},
        status="failed",
        failure_reason="delivery_gate_failed",
        error="identity_fail",
        result_image_id=None,
        created_at=now,
        completed_at=now,
        metadata={
            "candidates": [{}, {}],
            "budget": {"regenerations_used": 1},
            "provider_invocations": [
                {
                    "operation": "CREATE_FROM_REFERENCES",
                    "estimated_cost": 0.12,
                    "latency_ms": 1000,
                }
            ],
        },
    )
    return tmp_path


def _fresh_queue():
    q = JobQueue()
    assert SID not in q._sessions, "precondition: cache must be empty"
    return q


def test_rehydrate_returns_session(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state is not None


def test_owner_token_survives_restart(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state.owner_token == TOK


def test_paid_tier_survives_restart(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state.tier == PricingTier.premium


def test_max_revisions_and_payment_recovered(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state.max_revisions == 3
    assert state.payment_id == "pay_12345"


def test_session_consents_survive_restart(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state.session_consents.face_processing_consent is True
    assert state.session_consents.adult_subject_confirmed is True
    assert state.session_consents.no_training_by_default is True
    assert state.session_consents.cross_user_search_prohibited is True
    assert state.session_consents.long_term_face_library_prohibited is True
    assert state.to_response().session_consents.consented_at is not None


def test_style_gender_recovered(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state.style == StyleKey.business
    assert state.gender == "male"


def test_uploads_relisted_from_disk(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert len(state.uploaded_photos) == 2


def test_reference_slot_filename_is_stable_and_idempotent():
    assert reference_slot_filename("z-last.png", 1) == "ref01_z-last.png"
    assert reference_slot_filename("ref02_a-first.png", 2) == "ref02_a-first.png"


def test_rehydrate_preserves_identity_pack_slot_order(tmp_env):
    sid = "s_slots"
    storage.save_session(
        sid,
        "TOK_SLOTS",
        StyleKey.business.value,
        "female",
        storage.utcnow(),
    )
    up_dir = settings.upload_dir / sid
    up_dir.mkdir(parents=True, exist_ok=True)
    # Alphabetically, a-smile would sort before z-front. The refNN prefix is
    # the durable Identity Pack slot order.
    (up_dir / "ref01_z-front.png").write_bytes(b"png")
    (up_dir / "ref02_a-smile.png").write_bytes(b"png")
    (up_dir / "ref03_m-left.png").write_bytes(b"png")

    state = _fresh_queue().get_session(sid)

    assert state is not None
    assert [p.name for p in state.uploaded_photos] == [
        "ref01_z-front.png",
        "ref02_a-smile.png",
        "ref03_m-left.png",
    ]


def test_gallery_has_raw_and_pp_intermediate_excluded(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert len(state.generated_images) == 3
    ids = {im.image_id for im in state.generated_images}
    assert ids == {"img_aabbccdd", "pp_11223344", "pp_00000000"}


def test_gallery_rehydrates_in_persisted_creation_order(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert [image.image_id for image in state.generated_images] == [
        "img_aabbccdd",
        "pp_11223344",
        "pp_00000000",
    ]


def test_raw_image_metadata_and_resemblance_recovered(tmp_env):
    state = _fresh_queue().get_session(SID)
    raw = next(im for im in state.generated_images if im.image_id == "img_aabbccdd")
    assert raw.prompt_id == "biz_male_01"
    assert raw.resemblance is not None
    assert raw.resemblance.get("final_score") == 8
    assert raw.resemblance.get("history") == [6, 8]


def test_pp_variant_recovered_from_disk_with_marker(tmp_env):
    state = _fresh_queue().get_session(SID)
    pp = next(im for im in state.generated_images if im.image_id == "pp_11223344")
    assert pp.operation == "upscale_x2"
    assert pp.prompt_id == "upscale_x2"
    assert pp.parent_image_id == "img_aabbccdd"


def test_status_promoted_off_created_when_images_exist(tmp_env):
    state = _fresh_queue().get_session(SID)
    assert state.status != SessionStatus.created


def test_generation_event_metrics_survive_restart(tmp_env):
    state = _fresh_queue().get_session(SID)
    metrics = state.to_response().pipeline_metrics

    assert metrics["generation_attempts"] == 2
    assert metrics["generation_failures"] == 1
    assert metrics["failed_generation_reasons"] == {"delivery_gate_failed": 1}
    assert metrics["generation_failure_rate"] == 0.5
    assert metrics["candidates_generated"] == 2
    assert metrics["total_provider_invocations"] == 1
    assert metrics["create_from_reference_invocations"] == 1
    assert metrics["operation_counts"]["CREATE_FROM_REFERENCES"] == 1
    assert metrics["estimated_total_cost"] == 0.12
    assert metrics["estimated_cost_per_image"] == 0.06
    assert metrics["regenerations"] == 1
    assert metrics["regeneration_rate"] == 0.5
    assert metrics["shot_metrics"]["closeup"]["attempts"] == 1
    assert metrics["shot_metrics"]["closeup"]["completed"] == 1
    assert metrics["shot_metrics"]["closeup"]["failed"] == 0
    assert metrics["shot_metrics"]["closeup"]["deliverable_count"] == 1
    assert metrics["shot_metrics"]["closeup"]["first_pass_rate"] == 1
    assert metrics["shot_metrics"]["closeup"]["estimated_cost"] == 0.12
    assert metrics["shot_metrics"]["half_body"]["attempts"] == 1
    assert metrics["shot_metrics"]["half_body"]["completed"] == 0
    assert metrics["shot_metrics"]["half_body"]["failed"] == 1
    assert metrics["shot_metrics"]["half_body"]["failure_reasons"] == {
        "delivery_gate_failed": 1
    }
    assert metrics["shot_metrics"]["half_body"]["failure_rate"] == 1


def test_restart_recovers_hero_unlock_and_closes_interrupted_job(tmp_env):
    now = storage.utcnow()
    storage.update_session_hero_preview(SID, "img_aabbccdd", unlocked=True)
    storage.update_session_status(SID, SessionStatus.generating.value)
    storage.save_generation_event(
        event_id="j_interrupted",
        session_id=SID,
        job_id="j_interrupted",
        prompt_id="biz_male_03",
        shot_spec={"shot_id": "seated"},
        status="processing",
        failure_reason=None,
        error=None,
        result_image_id=None,
        created_at=now,
        completed_at=None,
    )

    state = _fresh_queue().get_session(SID)

    assert state.hero_preview_generated is True
    assert state.hero_preview_image_id == "img_aabbccdd"
    assert state.unlocked is True
    assert state.status == SessionStatus.failed
    interrupted = next(
        row for row in storage.load_generation_events(SID)
        if row["job_id"] == "j_interrupted"
    )
    assert interrupted["status"] == "failed"
    assert interrupted["failure_reason"] == "worker_interrupted"
    assert interrupted["completed_at"] is not None


@pytest.mark.asyncio
async def test_unlock_after_restart_queues_only_missing_shots(tmp_env):
    now = storage.utcnow()
    storage.update_session_hero_preview(SID, "img_aabbccdd", unlocked=True)
    storage.save_generation_event(
        event_id="j_half_body_done",
        session_id=SID,
        job_id="j_half_body_done",
        prompt_id="biz_male_half_body",
        shot_spec={"shot_id": "half_body"},
        status="completed",
        failure_reason=None,
        error=None,
        result_image_id="img_halfbody",
        created_at=now,
        completed_at=now,
    )
    queue = _fresh_queue()
    queue._prompts_data = json.loads(
        (_PIPELINE / "prompts.json").read_text(encoding="utf-8")
    )
    state = queue.get_session(SID)
    assert state is not None

    jobs = await queue.submit_unlock(SID)

    shot_ids = {(job.shot_spec or {}).get("shot_id") for job in jobs}
    assert "half_body" not in shot_ids
    assert shot_ids == {"environmental", "seated", "profile", "candid"}


def test_idempotent_rerehydrate(tmp_env):
    q = _fresh_queue()
    q._sessions.clear()
    state2 = q.get_session(SID)
    assert state2 is not None
    assert state2.owner_token == TOK
    assert state2.tier == PricingTier.premium


def test_orphan_disk_only_session_does_not_rehydrate_gallery(tmp_env):
    SID2 = "s_orphan"
    storage.save_session(SID2, "TOK_ORPHAN", StyleKey.social.value, "female",
                         storage.utcnow())
    out2 = settings.output_dir / SID2
    out2.mkdir(parents=True, exist_ok=True)
    (out2 / "img_deadbeef.png").write_bytes(b"\x89PNG" + b"z" * 40)
    # Deliberately NO save_generated_image call.
    q = _fresh_queue()
    orphan = q.get_session(SID2)
    assert orphan is not None
    assert orphan.generated_images == []


def test_unknown_session_returns_none(tmp_env):
    assert _fresh_queue().get_session("s_does_not_exist") is None


def test_init_db_migrates_generation_event_metadata_column(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "_DB_PATH", None)
    with storage.get_conn() as conn:
        conn.execute(
            """CREATE TABLE generation_events (
                event_id        TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                job_id          TEXT NOT NULL,
                prompt_id       TEXT,
                shot_spec_json  TEXT,
                status          TEXT NOT NULL,
                failure_reason  TEXT,
                error           TEXT,
                result_image_id TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            )"""
        )

    storage.init_db()

    with storage.get_conn() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(generation_events)")}
    assert "metadata_json" in columns


def test_delete_then_rehydrate_returns_none(tmp_env):
    q = _fresh_queue()
    storage.delete_session_images(SID)
    storage.delete_session_generation_events(SID)
    storage.delete_session_row(SID)
    q._sessions.pop(SID, None)
    assert q.get_session(SID) is None


@pytest.mark.asyncio
async def test_delete_session_removes_disk_dirs_even_when_not_hydrated(tmp_env):
    sid = "s_staleprivacy"
    storage.save_session(
        sid,
        "TOK_STALE",
        StyleKey.business.value,
        "female",
        storage.utcnow(),
    )
    upload_dir = settings.upload_dir / sid
    output_dir = settings.output_dir / sid
    upload_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (upload_dir / "ref01_front.png").write_bytes(b"face")
    (output_dir / "img_deadbeef.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    root_candidate = settings.output_dir / f"{sid}_hero_cand1_deadbeef.png"
    root_candidate.write_bytes(b"private intermediate pixels")
    other_candidate = settings.output_dir / "s_other_hero_cand1_keep.png"
    other_candidate.write_bytes(b"other session")
    storage.save_generation_event(
        event_id="j_stale",
        session_id=sid,
        job_id="j_stale",
        prompt_id="business_closeup",
        shot_spec={"shot_id": "closeup"},
        status="failed",
        failure_reason="delivery_gate_failed",
        error="identity_fail",
        result_image_id=None,
        created_at=storage.utcnow(),
        completed_at=storage.utcnow(),
    )

    q = _fresh_queue()
    assert sid not in q._sessions

    await q.delete_session(sid)

    assert not upload_dir.exists()
    assert not output_dir.exists()
    assert not root_candidate.exists()
    assert other_candidate.exists()
    assert storage.load_generation_events(sid) == []
    assert q.get_session(sid) is None


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest",
                             str(__file__), "-q"]))
