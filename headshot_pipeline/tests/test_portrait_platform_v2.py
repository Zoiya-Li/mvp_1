from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

from server import storage
from server.config import settings
from server.job_queue import JobQueue, queue
from server.models import (
    FeedbackEvent,
    GeneratedImage,
    Job,
    JobStatus,
    JobType,
    PaymentResponse,
    PaymentStatus,
    PricingTier,
    SessionState,
    SessionStatus,
    StyleKey,
)
from server.payment import PaymentService
from server.portrait_catalog import seed_theme_catalog
from server.delivery_label import (  # noqa: E402
    CLEAN_EXPORT_RETENTION_DAYS,
    CLEAN_EXPORT_TERMS_VERSION,
)
from server.portrait_domain import (  # noqa: E402
    CleanExportRequest,
    CreatePortraitOrderRequest,
    PreviewRetryRequest,
)
from server.router_portrait_v2 import (
    _sync_project_generation,
    catalog_image,
    confirm_project_preview,
    create_project_order,
    project_clean_asset,
    retry_project_preview,
    theme_detail,
    themes,
)
from server.portrait_storage import (
    attach_legacy_session,
    create_guest_user,
    create_project,
    create_share_recipe,
    credit_balance,
    delete_project_data,
    deliver_photo_set,
    expire_project_sources,
    get_asset,
    get_photo_set,
    get_project,
    get_portrait_order,
    get_theme,
    get_share_recipe,
    list_themes,
    list_projects,
    save_inspiration,
    save_portrait_order,
    set_project_preview_ready,
    spend_credit_once,
    user_for_token,
    upsert_theme,
)


@pytest.fixture()
def portrait_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB_PATH", tmp_path / "portrait-test.db")
    storage.init_db()
    yield tmp_path
    storage._DB_PATH = None


def _theme_for(engine_style_key: str, presentation: str | None = None) -> dict:
    for summary in list_themes():
        detail = get_theme(summary["theme_id"])
        blueprint = detail["blueprint"]
        if blueprint.get("engine_style_key") != engine_style_key:
            continue
        if presentation and blueprint.get("presentation") != presentation:
            continue
        return detail
    raise AssertionError(
        f"No catalog shoot for {engine_style_key=} {presentation=}"
    )


def test_catalog_seeds_concrete_six_frame_shoots(portrait_db):
    catalog = list_themes()

    assert len(catalog) == 10
    assert any(theme["featured"] for theme in catalog)
    assert all(theme["source_style_key"].startswith("shoot_") for theme in catalog)
    assert all(
        theme["cover_image"].startswith("/api/v2/catalog-images/")
        for theme in catalog
    )
    for summary in catalog:
        theme = get_theme(summary["theme_id"])
        blueprint = theme["blueprint"]
        assert theme["theme_version_id"].startswith("thv_")
        assert blueprint["engine_style_key"]
        assert blueprint["presentation"] in {"female", "male"}
        assert len(blueprint["templates"]) == 1
        assert blueprint["templates"][0]["template_id"] == blueprint["template_id"]
        assert blueprint["templates"][0]["gender"] == blueprint["presentation"]
        assert len(blueprint["shots"]) == 6
        assert len({shot["shot_id"] for shot in blueprint["shots"]}) == 6
        assert all(shot["style_prompt"] for shot in blueprint["shots"])


def test_catalog_retires_broad_themes_without_breaking_old_project_lookup(
    portrait_db,
):
    now = storage.utcnow()
    upsert_theme({
        "theme_id": "thm_legacy_broad",
        "slug": "legacy-broad-style",
        "title": "旧宽泛分类",
        "title_en": "Legacy Broad Style",
        "tagline": "Legacy",
        "category": "Legacy",
        "cover_image": "/legacy.jpg",
        "preview_images": ["/legacy.jpg"],
        "use_cases": [],
        "source_style_key": "legacy_broad_style",
        "featured": False,
        "sort_order": 99,
        "blueprint": {"source_style_key": "cinematic", "templates": []},
    }, now)

    seed_theme_catalog()

    assert all(
        item["source_style_key"] != "legacy_broad_style"
        for item in list_themes()
    )
    assert get_theme("thm_legacy_broad")["source_style_key"] == "legacy_broad_style"


@pytest.mark.asyncio
async def test_catalog_image_endpoint_builds_small_cacheable_jpeg(
    portrait_db, monkeypatch,
):
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(settings, "data_dir", Path(portrait_db) / "data")

    response = await catalog_image("jp_m_fresh.jpg")

    rendered = Path(response.path)
    assert rendered.is_file()
    assert response.media_type == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    with Image.open(rendered) as image:
        assert image.format == "JPEG"
        assert image.width <= 720
        assert image.height <= 960


@pytest.mark.asyncio
async def test_portrait_portal_hides_single_shot_id_photo_contract(portrait_db):
    catalog = await themes()
    assert catalog.themes
    assert all(theme.source_style_key.startswith("shoot_") for theme in catalog.themes)
    with pytest.raises(HTTPException) as exc:
        await theme_detail("id-photo")
    assert exc.value.status_code == 404


def test_guest_token_and_credit_ledger_are_user_scoped(portrait_db):
    now = storage.utcnow()
    first, first_token = create_guest_user(now)
    second, second_token = create_guest_user(now)

    assert user_for_token(first_token)["user_id"] == first["user_id"]
    assert user_for_token(second_token)["user_id"] == second["user_id"]
    assert user_for_token("wrong-token") is None
    assert credit_balance(first["user_id"]) == 1
    assert credit_balance(second["user_id"]) == 1


def test_project_ownership_and_private_inspiration(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    stranger, _ = create_guest_user(now)
    theme = _theme_for("jk_portrait", "female")
    project = create_project(
        owner["user_id"],
        {
            "theme_id": theme["theme_id"],
            "source": "official_theme",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )

    assert project["status"] == "awaiting_references"
    assert get_project(project["project_id"], stranger["user_id"]) is None

    inspiration = Path(portrait_db) / "inspiration.jpg"
    inspiration.write_bytes(b"private image placeholder")
    spec = {
        "scene": "rainy city street",
        "wardrobe": "black coat",
        "lighting": "neon rim light",
        "composition": "vertical half body",
        "pose": "three-quarter turn",
        "mood": "cinematic",
        "forbidden_transfer": ["source_person_identity"],
    }
    asset_id = save_inspiration(
        user_id=owner["user_id"],
        project_id=project["project_id"],
        storage_path=str(inspiration),
        mime_type="image/jpeg",
        spec=spec,
        now=now,
    )
    restored = get_project(project["project_id"], owner["user_id"])

    assert asset_id.startswith("ast_")
    assert restored["source"] == "private_inspiration"
    assert restored["inspiration_spec"]["scene"] == "rainy city street"


def test_missing_theme_cannot_create_project(portrait_db):
    user, _ = create_guest_user(storage.utcnow())
    with pytest.raises(ValueError, match="Theme not found"):
        create_project(
            user["user_id"],
            {
                "theme_id": "thm_missing",
                "source": "official_theme",
                "gender": "unspecified",
                "shared_recipe_id": None,
            },
            storage.utcnow(),
        )


def test_project_bridge_and_idempotent_preview_debit(portrait_db):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    theme = _theme_for("cinematic", "female")
    project = create_project(
        user["user_id"],
        {
            "theme_id": theme["theme_id"],
            "source": "official_theme",
            "gender": "unspecified",
            "shared_recipe_id": None,
        },
        now,
    )

    attach_legacy_session(
        project_id=project["project_id"], user_id=user["user_id"],
        session_id="s_bridge", gender="female", status="ready", now=now,
    )
    restored = list_projects(user["user_id"])[0]
    assert restored["legacy_session_id"] == "s_bridge"
    assert restored["status"] == "ready"
    assert spend_credit_once(
        user_id=user["user_id"], reason="hero_preview",
        reference_id=project["project_id"], now=now,
    ) is True
    assert spend_credit_once(
        user_id=user["user_id"], reason="hero_preview",
        reference_id=project["project_id"], now=now,
    ) is False
    assert credit_balance(user["user_id"]) == 0


@pytest.mark.asyncio
async def test_preview_not_like_me_queues_one_feedback_conditioned_free_retry(
    portrait_db,
    monkeypatch,
):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    theme = _theme_for("cinematic", "female")
    project = create_project(
        user["user_id"],
        {
            "theme_id": theme["theme_id"],
            "source": "official_theme",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    session_id = "s_preview_retry"
    attach_legacy_session(
        project_id=project["project_id"],
        user_id=user["user_id"],
        session_id=session_id,
        gender="female",
        status="preview_ready",
        now=now,
    )
    set_project_preview_ready(
        project_id=project["project_id"],
        user_id=user["user_id"],
        image_id="img_first",
        now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.hero_preview_generated = True
    state.hero_preview_image_id = "img_first"
    state.status = SessionStatus.hero_preview_ready
    state.generated_images.append(GeneratedImage(
        image_id="img_first",
        url="/hero",
        prompt_id="closeup",
        turn=1,
        created_at=now,
        resemblance={"selected_candidate": {"deliverable": True}},
    ))
    queue._sessions[session_id] = state
    feedback_events: list[tuple[str, str]] = []

    class ReadyWorker:
        provider_readiness = {"pass": True}

        def connect(self):
            self.provider_readiness = {"pass": True}

    async def fake_record_feedback(
        _session_id, _image_id, event, reason=None, score=None,
    ):
        feedback_events.append((event.value, reason))
        state.user_feedback.append({
            "image_id": _image_id,
            "event": event.value,
            "reason": reason,
            "score": score,
        })
        return {"feedback_id": "fb_retry"}

    async def fake_submit(_session_id, **_kwargs):
        assert state.user_feedback[-1]["event"] == "not_like_me"
        assert _kwargs["template_id"] == theme["blueprint"]["template_id"]
        assert _kwargs["shot_overrides"] == theme["blueprint"]["shots"]
        state.status = SessionStatus.generating
        return [Job(
            session_id,
            JobType.hero_preview,
            "retry",
            replaces_image_id=_kwargs["replaces_image_id"],
        )]

    previous_worker = queue._worker
    queue._worker = ReadyWorker()
    monkeypatch.setattr(queue, "record_user_feedback", fake_record_feedback)
    monkeypatch.setattr(queue, "submit_hero_preview", fake_submit)
    try:
        response = await retry_project_preview(
            project["project_id"],
            PreviewRetryRequest(reason="identity"),
            user=user,
        )
        updated = get_project(project["project_id"], user["user_id"])
        synced_during_retry = _sync_project_generation(
            updated, user["user_id"]
        )
    finally:
        queue._worker = previous_worker
        queue._sessions.pop(session_id, None)

    assert response.status == "preview_generating"
    assert response.retries_remaining == 0
    assert len(response.jobs) == 1
    assert feedback_events == [("not_like_me", "preview_retry:identity")]
    assert updated["preview_retries_used"] == 1
    assert updated["preview_retries_remaining"] == 0
    assert updated["status"] == "preview_generating"
    assert state.hero_preview_generated is True
    assert state.hero_preview_image_id == "img_first"
    assert state.generated_images[0].operation is None
    assert response.jobs[0].job_id
    assert synced_during_retry["status"] == "preview_generating"


@pytest.mark.asyncio
async def test_preview_confirmation_records_current_hero_identity_feedback(
    portrait_db,
    monkeypatch,
):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    theme = _theme_for("cinematic", "female")
    project = create_project(
        user["user_id"],
        {
            "theme_id": theme["theme_id"],
            "source": "official_theme",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    session_id = "s_preview_confirm"
    attach_legacy_session(
        project_id=project["project_id"],
        user_id=user["user_id"],
        session_id=session_id,
        gender="female",
        status="preview_ready",
        now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.hero_preview_generated = True
    state.hero_preview_image_id = "img_confirmed"
    state.status = SessionStatus.hero_preview_ready
    queue._sessions[session_id] = state
    recorded: list[tuple[str, str, str, int]] = []

    async def fake_record_feedback(
        actual_session_id, image_id, event, reason=None, score=None,
    ):
        recorded.append((actual_session_id, image_id, event.value, score))
        payload = {
            "feedback_id": "fb_confirmed",
            "session_id": actual_session_id,
            "image_id": image_id,
            "event": event.value,
            "reason": reason,
            "score": score,
            "created_at": now.isoformat(),
        }
        state.user_feedback.append(payload)
        return payload

    monkeypatch.setattr(queue, "record_user_feedback", fake_record_feedback)
    try:
        response = await confirm_project_preview(project["project_id"], user=user)
        synced = _sync_project_generation(
            get_project(project["project_id"], user["user_id"]),
            user["user_id"],
        )
    finally:
        queue._sessions.pop(session_id, None)

    assert response.feedback_id == "fb_confirmed"
    assert response.image_id == "img_confirmed"
    assert response.event == FeedbackEvent.looks_like_me
    assert synced["preview_confirmed"] is True
    assert recorded == [(
        session_id,
        "img_confirmed",
        "looks_like_me",
        2,
    )]


@pytest.mark.asyncio
async def test_failed_preview_replacement_restores_original_hero(portrait_db):
    now = storage.utcnow()
    session_id = "s_preview_retry_failure"
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.hero_preview_generated = True
    state.hero_preview_image_id = "img_original"
    state.status = SessionStatus.generating
    original = GeneratedImage(
        image_id="img_original",
        url="/hero",
        prompt_id="closeup",
        turn=1,
        created_at=now,
        resemblance={"selected_candidate": {"deliverable": True}},
    )
    state.generated_images.append(original)
    storage.save_session(
        state.session_id, state.owner_token, state.style.value,
        state.gender, state.created_at,
    )
    storage.save_generated_image(
        image_id=original.image_id,
        session_id=state.session_id,
        prompt_id=original.prompt_id,
        turn=original.turn,
        revised_image_id=None,
        parent_image_id=None,
        operation=None,
        resemblance=original.resemblance,
        created_at=original.created_at,
    )
    replacement = Job(
        session_id,
        JobType.hero_preview,
        "retry",
        replaces_image_id=original.image_id,
    )
    queue._sessions[session_id] = state

    class UnavailableWorker:
        provider_readiness = {"pass": False}

    previous_worker = queue._worker
    queue._worker = UnavailableWorker()
    try:
        await queue._execute_job(replacement)
    finally:
        queue._worker = previous_worker
        queue._sessions.pop(session_id, None)

    assert replacement.status.value == "failed"
    assert state.status == SessionStatus.hero_preview_ready
    assert state.hero_preview_generated is True
    assert state.hero_preview_image_id == original.image_id
    assert storage.load_session_row(session_id)["status"] == "hero_preview_ready"
    assert storage.load_generated_images(session_id)[0]["operation"] is None


def test_successful_preview_replacement_supersedes_original(portrait_db):
    now = storage.utcnow()
    state = SessionState("s_preview_retry_success", StyleKey.cinematic, "female", "tok")
    original = GeneratedImage(
        image_id="img_original",
        url="/hero",
        prompt_id="closeup",
        turn=1,
        created_at=now,
        resemblance={"selected_candidate": {"deliverable": True}},
    )
    state.generated_images.append(original)
    storage.save_session(
        state.session_id, state.owner_token, state.style.value,
        state.gender, state.created_at,
    )
    storage.save_generated_image(
        image_id=original.image_id,
        session_id=state.session_id,
        prompt_id=original.prompt_id,
        turn=original.turn,
        revised_image_id=None,
        parent_image_id=None,
        operation=None,
        resemblance=original.resemblance,
        created_at=original.created_at,
    )
    replacement = Job(
        state.session_id,
        JobType.hero_preview,
        "retry",
        replaces_image_id=original.image_id,
    )

    assert queue._supersede_replaced_hero(state, replacement) is True
    assert original.operation == "superseded_preview"
    assert storage.load_generated_images(state.session_id)[0]["operation"] == "superseded_preview"


def test_portrait_order_is_owned_and_persisted(portrait_db):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    stranger, _ = create_guest_user(now)
    project = create_project(
        user["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    save_portrait_order(
        order_id="pay_portrait", user_id=user["user_id"],
        project_id=project["project_id"], product_code="portrait_set",
        amount_cents=500, status="pending", now=now,
    )

    assert get_portrait_order("pay_portrait", user["user_id"])["amount_cents"] == 500
    assert get_portrait_order("pay_portrait", stranger["user_id"]) is None


@pytest.mark.asyncio
async def test_project_order_schedules_mock_confirmation_on_route_loop(
    portrait_db, monkeypatch,
):
    now = storage.utcnow()
    user, _ = create_guest_user(now)
    project = create_project(
        user["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    session_id = "s_mock_checkout"
    attach_legacy_session(
        project_id=project["project_id"], user_id=user["user_id"],
        session_id=session_id, gender="female", status="preview_ready", now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.hero_preview_generated = True
    state.hero_preview_image_id = "img_hero"
    queue._sessions[session_id] = state
    scheduled: list[str] = []

    def fake_create_order(*_args, **_kwargs):
        return PaymentResponse(
            payment_id="pay_mock_route",
            session_id=session_id,
            tier=PricingTier.standard,
            status=PaymentStatus.pending,
            amount_cents=500,
            created_at=now,
        )

    monkeypatch.setattr(PaymentService, "create_order", fake_create_order)
    monkeypatch.setattr(
        PaymentService,
        "schedule_mock_confirmation",
        lambda payment_id: scheduled.append(payment_id) or True,
    )
    try:
        response = await create_project_order(
            project["project_id"],
            CreatePortraitOrderRequest(product_code="portrait_set"),
            user=user,
        )
    finally:
        queue._sessions.pop(session_id, None)

    assert response.status == "pending"
    assert scheduled == ["pay_mock_route"]
    assert get_portrait_order("pay_mock_route", user["user_id"])["status"] == "pending"


def test_shared_recipe_recreates_direction_without_source_pixels(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    recipient, _ = create_guest_user(now)
    theme = _theme_for("cinematic", "female")
    project = create_project(
        owner["user_id"],
        {
            "theme_id": theme["theme_id"],
            "source": "official_theme",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    shared = create_share_recipe(
        user_id=owner["user_id"], project=project,
        title="Rainy cinema", recipe={
            "source": "private_inspiration",
            "theme_id": theme["theme_id"],
            "inspiration_spec": {"scene": "rainy street", "mood": "cinematic"},
        },
        include_portrait=False, hero_image_id=None, now=now,
    )
    imported = create_project(
        recipient["user_id"],
        {
            "theme_id": None,
            "source": "shared_recipe",
            "gender": "female",
            "shared_recipe_id": shared["share_token"],
        },
        now,
    )

    assert get_share_recipe(shared["share_token"])["include_portrait"] is False
    assert imported["theme_id"] == theme["theme_id"]
    assert imported["inspiration_spec"]["scene"] == "rainy street"
    assert imported["inspiration_asset_id"] is None


def test_delivered_set_is_exactly_six_owned_immutable_assets(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    stranger, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    images = []
    for index in range(6):
        path = Path(portrait_db) / f"finished-{index}.png"
        path.write_bytes(b"finished portrait")
        images.append({"image_id": f"img_{index}", "storage_path": str(path)})

    photo_set_id = deliver_photo_set(
        user_id=owner["user_id"], project_id=project["project_id"],
        title="Six-frame story", images=images, now=now,
    )
    delivered = get_photo_set(photo_set_id, project["project_id"], owner["user_id"])

    assert delivered["status"] == "delivered"
    assert len(delivered["assets"]) == 6
    assert [item["position"] for item in delivered["assets"]] == list(range(6))
    assert get_photo_set(photo_set_id, project["project_id"], stranger["user_id"]) is None
    assert get_project(project["project_id"], owner["user_id"])["status"] == "delivered"
    assert deliver_photo_set(
        user_id=owner["user_id"], project_id=project["project_id"],
        title="Changed title", images=list(reversed(images)), now=now,
    ) == photo_set_id

    with pytest.raises(ValueError, match="exactly six"):
        deliver_photo_set(
            user_id=owner["user_id"], project_id="missing",
            title="Incomplete", images=images[:5], now=now,
        )


@pytest.mark.asyncio
async def test_clean_export_requires_consent_and_keeps_six_month_audit(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    images = []
    for index in range(6):
        labeled = Path(portrait_db) / f"labeled-{index}.png"
        clean = Path(portrait_db) / f"clean-{index}.png"
        labeled.write_bytes(b"labeled")
        clean.write_bytes(b"clean")
        images.append({
            "image_id": f"img_{index}",
            "storage_path": str(labeled),
            "clean_storage_path": str(clean),
        })
    set_id = deliver_photo_set(
        user_id=owner["user_id"],
        project_id=project["project_id"],
        title="Clean export set",
        images=images,
        now=now,
    )
    delivered = get_photo_set(set_id, project["project_id"], owner["user_id"])
    asset_id = delivered["assets"][0]["asset_id"]

    response = await project_clean_asset(
        project["project_id"],
        asset_id,
        CleanExportRequest(
            terms_version=CLEAN_EXPORT_TERMS_VERSION,
            ai_generated_acknowledged=True,
            redistribution_responsibility_accepted=True,
        ),
        user=owner,
    )

    assert Path(response.path) == Path(images[0]["clean_storage_path"])
    with storage.get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM portrait_clean_export_requests WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
    assert row is not None
    assert row["terms_version"] == CLEAN_EXPORT_TERMS_VERSION
    retained = datetime.fromisoformat(row["retain_until"])
    requested = datetime.fromisoformat(row["requested_at"])
    assert (retained - requested).days == CLEAN_EXPORT_RETENTION_DAYS

    deleted = delete_project_data(project["project_id"], owner["user_id"])
    assert set(deleted["paths"]) == {
        path
        for image in images
        for path in (image["storage_path"], image["clean_storage_path"])
    }
    with storage.get_conn() as conn:
        retained_row = conn.execute(
            "SELECT 1 FROM portrait_clean_export_requests WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
    assert retained_row is not None


def test_project_deletion_revokes_shares_and_removes_media_metadata(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    inspiration = Path(portrait_db) / "private-source.jpg"
    inspiration.write_bytes(b"source")
    asset_id = save_inspiration(
        user_id=owner["user_id"], project_id=project["project_id"],
        storage_path=str(inspiration), mime_type="image/jpeg",
        spec={"scene": "studio"}, now=now,
    )
    shared = create_share_recipe(
        user_id=owner["user_id"], project=get_project(project["project_id"], owner["user_id"]),
        title="Private direction", recipe={"source": "private_inspiration"},
        include_portrait=False, hero_image_id=None, now=now,
    )

    payload = delete_project_data(project["project_id"], owner["user_id"])

    assert payload["paths"] == [str(inspiration)]
    assert get_project(project["project_id"], owner["user_id"]) is None
    assert get_asset(asset_id, owner["user_id"]) is None
    assert get_share_recipe(shared["share_token"]) is None


def test_source_expiration_removes_private_style_and_requires_fresh_upload(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    path = Path(portrait_db) / "expiring-source.jpg"
    path.write_bytes(b"source")
    asset_id = save_inspiration(
        user_id=owner["user_id"], project_id=project["project_id"],
        storage_path=str(path), mime_type="image/jpeg",
        spec={"scene": "window light"}, now=now,
    )

    paths = expire_project_sources(project["project_id"], owner["user_id"])
    restored = get_project(project["project_id"], owner["user_id"])

    assert paths == [str(path)]
    assert get_asset(asset_id, owner["user_id"]) is None
    assert restored["inspiration_asset_id"] is None
    assert restored["inspiration_spec"] is None
    assert restored["status"] == "draft"


def test_completed_engine_batch_syncs_to_six_asset_delivery(portrait_db):
    from PIL import Image, ImageDraw

    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    session_id = "s_portrait_delivery"
    attach_legacy_session(
        project_id=project["project_id"], user_id=owner["user_id"],
        session_id=session_id, gender="female", status="set_generating", now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.output_dir = Path(portrait_db) / "delivery-output"
    state.output_dir.mkdir()
    state.unlocked = True
    state.status = SessionStatus.done
    state.hero_preview_image_id = "img_hero"
    shot_ids = [
        "closeup", "half_body", "environmental",
        "seated", "profile", "candid",
    ]
    geometry_profiles = [
        "closeup", "medium", "small_face", "medium", "medium", "medium",
    ]
    face_areas = [0.24, 0.12, 0.035, 0.10, 0.14, 0.09]
    center_offsets = [0.04, 0.08, 0.16, 0.07, 0.12, 0.10]
    for index, shot_id in enumerate(shot_ids):
        image_id = "img_hero" if index == 0 else f"img_set_{index - 1}"
        path = state.output_dir / f"{image_id}.png"
        image = Image.new("RGB", (96, 128), (25 + index * 18, 38, 70))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (6 + index * 7, 10 + index * 3, 38 + index * 7, 85 + index * 2),
            fill=(220, 180 - index * 13, 80 + index * 19),
        )
        draw.line(
            (0, 20 + index * 15, 95, 110 - index * 9),
            fill="white", width=5,
        )
        image.save(path)
        look = "Look A" if index < 3 else "Look B"
        state.generated_images.append(GeneratedImage(
            image_id=image_id,
            url=f"/images/{image_id}",
            prompt_id=shot_id,
            turn=1,
            created_at=now,
            resemblance={
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
                                "geometry_profile": geometry_profiles[index],
                            },
                        },
                    },
                },
            },
        ))
    queue._sessions[session_id] = state
    try:
        synced = _sync_project_generation(
            get_project(project["project_id"], owner["user_id"]),
            owner["user_id"],
        )
    finally:
        queue._sessions.pop(session_id, None)

    assert synced["status"] == "delivered"
    assert synced["photo_set_id"].startswith("set_")
    delivered = get_photo_set(
        synced["photo_set_id"], project["project_id"], owner["user_id"],
    )
    assert len(delivered["assets"]) == 6
    cover = get_asset(delivered["cover_asset_id"], owner["user_id"])
    assert json.loads(cover["metadata_json"])["legacy_image_id"] == "img_hero"


def test_set_quality_failure_blocks_delivery_and_preserves_paid_retry(
    portrait_db, monkeypatch,
):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    session_id = "s_set_quality_failed"
    attach_legacy_session(
        project_id=project["project_id"], user_id=owner["user_id"],
        session_id=session_id, gender="female", status="set_generating", now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.output_dir = Path(portrait_db) / "set-quality-failed-output"
    state.output_dir.mkdir()
    state.unlocked = True
    state.status = SessionStatus.done
    state.hero_preview_image_id = "img_0"
    for index in range(6):
        image_id = f"img_{index}"
        (state.output_dir / f"{image_id}.png").write_bytes(
            f"portrait-{index}".encode()
        )
        state.generated_images.append(GeneratedImage(
            image_id=image_id,
            url=f"/images/{image_id}",
            prompt_id=f"shot_{index}",
            turn=1,
            created_at=now,
            resemblance={
                "selected_candidate": {
                    "deliverable": True,
                    "gate_status": {"hard_gates_pass": True},
                },
            },
        ))
    monkeypatch.setattr(
        "server.router_portrait_v2.evaluate_portrait_set",
        lambda images: {
            "pass": False,
            "hard_failures": ["set_is_selfie_dominated"],
            "diagnostics": {},
            "visual_review": {"required": True},
            "policy_version": "portrait_set_delivery_v1",
        },
    )
    queue._sessions[session_id] = state
    try:
        synced = _sync_project_generation(
            get_project(project["project_id"], owner["user_id"]),
            owner["user_id"],
        )
    finally:
        queue._sessions.pop(session_id, None)

    assert synced["status"] == "failed"
    assert state.status == SessionStatus.failed
    assert synced["failure_message"] == (
        "有一张或多张写真没有通过最终检查。你的购买权益已保留，可联系客服协助重试。"
    )
    assert synced.get("photo_set_id") is None
    with storage.get_conn() as conn:
        event = conn.execute(
            """SELECT metadata_json FROM portrait_operational_events
               WHERE project_id=? AND event_type='portrait_set_quality_failed'""",
            (project["project_id"],),
        ).fetchone()
    assert json.loads(event["metadata_json"])["hard_failures"] == [
        "set_is_selfie_dominated"
    ]


@pytest.mark.asyncio
async def test_set_quality_retry_requeues_five_shots_without_old_assets(portrait_db):
    q = JobQueue()
    session_id = "s_set_quality_retry"
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.hero_preview_image_id = "img_hero"
    state.status = SessionStatus.done
    state.unlocked = True
    state.generated_images.append(GeneratedImage(
        image_id="img_hero", url="/images/img_hero", prompt_id="closeup",
        turn=1, created_at=storage.utcnow(),
    ))
    retry_shots = ["half_body", "environmental", "seated", "profile", "candid"]
    for index, shot_id in enumerate(retry_shots):
        image_id = f"img_paid_{index}"
        state.generated_images.append(GeneratedImage(
            image_id=image_id, url=f"/images/{image_id}", prompt_id=shot_id,
            turn=1, created_at=storage.utcnow(),
        ))
        job = Job(
            session_id=session_id,
            job_type=JobType.full_set,
            prompt=f"prompt {shot_id}",
            prompt_id=shot_id,
            shot_spec={"shot_id": shot_id},
        )
        job.status = JobStatus.completed
        q._jobs[job.job_id] = job
    q._sessions[session_id] = state

    prepared = q.prepare_set_quality_retry(session_id)
    restarted = JobQueue()
    restarted._sessions[session_id] = state
    replacements = await restarted.retry_failed_jobs(session_id)

    assert prepared == retry_shots
    assert state.generated_images[0].operation is None
    assert all(
        image.operation == "set_quality_retry_pending"
        for image in state.generated_images[1:]
    )
    assert [job.shot_spec["shot_id"] for job in replacements] == retry_shots
    assert all(job.status == JobStatus.queued for job in replacements)
    assert restarted._queue.qsize() == 5
    assert state.status == SessionStatus.generating


def test_failed_project_sync_exposes_safe_actionable_reason(portrait_db):
    now = storage.utcnow()
    owner, _ = create_guest_user(now)
    project = create_project(
        owner["user_id"],
        {
            "theme_id": None,
            "source": "private_inspiration",
            "gender": "female",
            "shared_recipe_id": None,
        },
        now,
    )
    session_id = "s_failed_portrait"
    attach_legacy_session(
        project_id=project["project_id"], user_id=owner["user_id"],
        session_id=session_id, gender="female", status="preview_generating", now=now,
    )
    state = SessionState(session_id, StyleKey.cinematic, "female", "tok")
    state.status = SessionStatus.failed
    queue._sessions[session_id] = state
    storage.save_generation_event(
        event_id="j_failed_portrait",
        session_id=session_id,
        job_id="j_failed_portrait",
        prompt_id="hero_closeup",
        shot_spec={"shot_id": "closeup"},
        status="failed",
        failure_reason="delivery_gate_failed",
        error="provider-secret-detail-must-not-be-public",
        result_image_id=None,
        created_at=now,
        completed_at=now,
    )
    assert spend_credit_once(
        user_id=owner["user_id"], reason="hero_preview",
        reference_id=project["project_id"], now=now,
    ) is True
    assert credit_balance(owner["user_id"]) == 0
    try:
        synced = _sync_project_generation(
            get_project(project["project_id"], owner["user_id"]),
            owner["user_id"],
        )
        repeated = _sync_project_generation(
            get_project(project["project_id"], owner["user_id"]),
            owner["user_id"],
        )
    finally:
        queue._sessions.pop(session_id, None)

    assert synced["status"] == "failed"
    assert synced["failure_code"] == "delivery_gate_failed"
    assert "更加清晰的照片" in synced["failure_message"]
    assert "已经恢复" in synced["failure_message"]
    assert "provider-secret-detail" not in synced["failure_message"]
    assert credit_balance(owner["user_id"]) == 1
    assert repeated["status"] == "failed"
    assert credit_balance(owner["user_id"]) == 1
