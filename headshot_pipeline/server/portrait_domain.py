"""Public contracts for FlashShot's portrait-platform API v2.

The v1 API treats a generation session as user, order, entitlement, and
project at once.  These contracts separate those concepts while the existing
generation engine remains available behind a compatibility bridge.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import JobResponse


ProjectSource = Literal["official_theme", "private_inspiration", "shared_recipe"]
ProjectStatus = Literal[
    "draft",
    "awaiting_references",
    "ready",
    "preview_generating",
    "preview_ready",
    "set_generating",
    "delivered",
    "failed",
]


class GuestUserResponse(BaseModel):
    user_id: str
    access_token: str
    created_at: datetime


class AppleSignInRequest(BaseModel):
    identity_token: str = Field(min_length=20, max_length=16_384)
    raw_nonce: str = Field(min_length=16, max_length=512)
    display_name: str | None = Field(default=None, max_length=120)


class AuthenticatedUserResponse(BaseModel):
    user_id: str
    access_token: str
    account_type: Literal["apple"] = "apple"
    merged_guest_workspace: bool = False
    created_at: datetime


class ThemeSummary(BaseModel):
    theme_id: str
    slug: str
    title: str
    title_en: str
    tagline: str
    category: str
    cover_image: str
    preview_images: list[str] = Field(default_factory=list)
    featured: bool = False
    source_style_key: str
    active_version: int
    presentation: Literal["male", "female", "unspecified"] = "unspecified"
    preview_integrity: Literal[
        "single_direction_study", "coherent_series"
    ] = "single_direction_study"
    shot_labels: list[str] = Field(default_factory=list)


class ThemeDetail(ThemeSummary):
    use_cases: list[str] = Field(default_factory=list)
    shot_count: int = 6
    reference_min: int = 4
    reference_max: int = 6
    blueprint: dict[str, Any] = Field(default_factory=dict)


class ThemeListResponse(BaseModel):
    themes: list[ThemeSummary]


class CreateProjectRequest(BaseModel):
    theme_id: str | None = None
    source: ProjectSource = "official_theme"
    gender: Literal["male", "female", "unspecified"] = "unspecified"
    shared_recipe_id: str | None = None


class PortraitProjectResponse(BaseModel):
    project_id: str
    user_id: str
    theme_id: str | None = None
    theme_version_id: str | None = None
    source: ProjectSource
    status: ProjectStatus
    gender: str
    inspiration_asset_id: str | None = None
    inspiration_spec: dict[str, Any] | None = None
    hero_asset_id: str | None = None
    photo_set_id: str | None = None
    legacy_session_id: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    preview_retries_used: int = 0
    preview_retries_remaining: int = 1
    preview_confirmed: bool = False
    created_at: datetime
    updated_at: datetime


class InspirationUploadResponse(BaseModel):
    asset_id: str
    project_id: str
    analysis_status: Literal["analyzed", "pending", "rejected"]
    inspiration_spec: dict[str, Any] | None = None
    message: str


class ReferenceUploadResponse(BaseModel):
    project_id: str
    legacy_session_id: str
    reference_count: int
    status: ProjectStatus
    reference_quality: dict[str, Any]


class PreviewRetryRequest(BaseModel):
    reason: Literal["identity", "expression", "overall"] = "identity"


class PreviewRetryResponse(BaseModel):
    project_id: str
    status: Literal["preview_generating"] = "preview_generating"
    retries_remaining: int
    jobs: list[JobResponse]


class ProjectListResponse(BaseModel):
    projects: list[PortraitProjectResponse]


class PortraitAssetResponse(BaseModel):
    asset_id: str
    mime_type: str
    position: int


class PhotoSetResponse(BaseModel):
    photo_set_id: str
    project_id: str
    title: str
    status: Literal["delivered"]
    cover_asset_id: str | None = None
    assets: list[PortraitAssetResponse]
    created_at: datetime
    delivered_at: datetime


class CleanExportRequest(BaseModel):
    terms_version: str = Field(min_length=1, max_length=80)
    ai_generated_acknowledged: bool
    redistribution_responsibility_accepted: bool


class EntitlementBalanceResponse(BaseModel):
    user_id: str
    credit_balance: int


class CreatePortraitOrderRequest(BaseModel):
    product_code: Literal["portrait_set", "portrait_set_hd"] = "portrait_set"


class PortraitOrderResponse(BaseModel):
    order_id: str
    project_id: str
    product_code: str
    status: Literal["pending", "paid", "expired", "refunded"]
    amount_cents: int
    currency: str = "USD"
    checkout_url: str | None = None


class ApplePurchaseClaimRequest(BaseModel):
    signed_transaction: str = Field(min_length=20, max_length=65_536)


class ApplePurchaseClaimResponse(BaseModel):
    order_id: str
    project_id: str
    product_id: str
    transaction_id: str
    status: Literal["paid", "refunded"]
    newly_claimed: bool


class AppleNotificationRequest(BaseModel):
    signedPayload: str = Field(min_length=20, max_length=131_072)


class CreateShareRecipeRequest(BaseModel):
    include_portrait: bool = False


class SharedRecipeResponse(BaseModel):
    share_token: str
    title: str
    theme_id: str | None = None
    theme_slug: str | None = None
    source: ProjectSource
    recipe: dict[str, Any] = Field(default_factory=dict)
    portrait_available: bool = False
