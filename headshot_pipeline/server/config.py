"""Server configuration via pydantic-settings."""

import secrets
from pathlib import Path
from typing import Literal
from pydantic import PrivateAttr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_environment: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Generation backend switch ───────────────────────────────
    # "openrouter"  -> Gemini image models through OpenRouter (default; production)
    # "siliconflow" -> Qwen Image Edit + Qwen VLM judge through SiliconFlow
    # "chrome"      -> drive gemini.google.com via an already-running Chrome CDP
    gemini_backend: Literal["openrouter", "siliconflow", "chrome"] = "openrouter"

    # Chrome backend settings (only used when gemini_backend == "chrome").
    # The chrome_cdp_port is also kept for the legacy persistent_client.py CLI.
    chrome_cdp_port: int = 9222
    chrome_user_data_dir: Path | None = None
    chrome_headless: bool = False
    chrome_wait_timeout: int = 120

    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    upload_dir: Path | None = None  # derived in model_post_init if unset
    output_dir: Path | None = None  # derived in model_post_init if unset
    gemini_wait_timeout: int = 180

    # Legacy Gemini image settings retained for the optional chat-completions
    # image client. Production OpenRouter generation uses the dedicated Image
    # API settings below.
    gemini_image_aspect_ratio: str = "3:4"
    gemini_image_size: str = "1K"

    # ── OpenRouter dedicated Image API (production generation path) ──
    # ``gemini_model`` is a legacy env name kept for deployment compatibility;
    # it now identifies the active OpenRouter image model.
    openrouter_api_key: str = ""
    gemini_model: str = "bytedance-seed/seedream-4.5"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_judge_model: str = "qwen/qwen3-vl-32b-instruct"
    openrouter_image_provider: str = "seed"
    openrouter_image_size: str = "1728x2304"
    openrouter_estimated_image_cost: float = 0.04
    # FLUX.2 Pro is the identity-critical base generator for Hero and the paid
    # set. Seedream remains available for bounded local edits; production E2E
    # showed its raw portraits repeatedly failed the skin-texture realism gate.
    openrouter_hero_model: str = "black-forest-labs/flux.2-pro"
    openrouter_hero_image_provider: str = "black-forest-labs"
    openrouter_hero_estimated_image_cost: float = 0.12
    # Optional benchmark-approved route used only after the primary route has
    # failed the same quality class. Empty by default: a model is never promoted
    # to recovery duty merely because its API is available.
    openrouter_recovery_model: str = ""
    openrouter_recovery_image_provider: str = ""
    openrouter_recovery_estimated_image_cost: float = 0.15
    hero_identity_cosine_accept_threshold: float = 0.78
    # Multi-view references naturally disagree because of pose, expression,
    # and phone-lens distortion. Hero may calibrate down to the pack's own
    # consistency, but never below this conservative same-person floor.
    hero_identity_cosine_reference_floor: float = 0.76
    openrouter_min_credit_balance: float = 3.0
    openrouter_max_reference_images: int = 5
    # SiliconFlow image/edit + VLM judge backend. The same account key can
    # access both /images/generations and OpenAI-compatible /chat/completions.
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_image_model: str = "Qwen/Qwen-Image-Edit-2509"
    siliconflow_text_to_image_model: str = "Qwen/Qwen-Image"
    siliconflow_judge_model: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    siliconflow_estimated_image_cost: float = 0.05
    max_file_size_mb: int = 10
    max_photos: int = 6
    # Identity-preserving portraits need multiple references. Four gives the
    # pipeline enough signal for front/smile/angle coverage without
    # making onboarding too heavy.
    min_photos: int = 4
    # Overseas build: the marketing site is flashshot.top. Localhost stays for
    # dev. (The legacy shanxiang.ai CN origins are dropped — the product pivoted
    # overseas to avoid ICP filing / generative-AI registration / PIPL scope.)
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3010",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3010",
        "https://flashshot.top",
        "https://www.flashshot.top",
    ]

    # ── Security ───────────────────────────────────────
    # Secret key for signing session tokens. Generated once per-process if not
    # set; for production set SESSION_SECRET_KEY in env so tokens survive
    # restarts.
    session_secret_key: str = ""

    # Payment: mock auto-confirm is DANGEROUS (gives away premium for free).
    # It is OFF by default. Set PAYMENT_MOCK_ENABLED=1 ONLY for local dev.
    payment_mock_enabled: bool = False

    # Paddle (Merchant of Record) — the overseas payment provider. Leave unset
    # → real payment returns a clear error instead of silently mocking. Fill
    # from the Paddle dashboard: sandbox first, then flip PADDLE_ENVIRONMENT
    # to production at launch. See server/payment.py for the full flow.
    paddle_api_key: str = ""  # server API key
    paddle_client_token: str = ""  # public Paddle.js token (live_... / test_...)
    paddle_webhook_secret: str = ""  # webhook signing key (verifies callbacks)
    paddle_environment: str = "sandbox"  # "sandbox" | "production"
    paddle_price_standard_id: str = ""  # pri_... for the $5 Standard tier
    paddle_price_premium_id: str = ""  # pri_... for the $10 Pro tier
    paddle_return_url: str = "https://flashshot.top/checkout"  # approved Paddle.js page

    # Apple is the primary payment and account provider for the native iOS app.
    # The server validates Sign in with Apple identity tokens and StoreKit 2
    # signed transactions; the client can never grant its own entitlement.
    apple_client_id: str = "com.flashshot.app"
    apple_bundle_id: str = "com.flashshot.app"
    apple_app_id: int | None = None
    apple_iap_environment: Literal["sandbox", "production"] = "sandbox"
    apple_iap_product_id: str = "portrait_set_6"
    apple_root_cert_dir: Path = Path(__file__).resolve().parent.parent / "config" / "apple-root-certs"
    apple_jwks_url: str = "https://appleid.apple.com/auth/keys"
    apple_identity_issuer: str = "https://appleid.apple.com"
    trusted_proxy_ips: list[str] = ["127.0.0.1", "::1"]
    guest_create_limit_10m: int = 10
    guest_create_limit_day: int = 5
    preview_limit_ip_day: int = 5
    support_admin_token: str = ""

    # Layered retention. Identity/inspiration sources expire first; generated
    # portraits remain available longer; operational metadata is last.
    source_retention_days: int = 7
    generated_retention_days: int = 30
    metadata_retention_days: int = 90
    retention_days: int = 7  # legacy env compatibility

    # Face-swap post-processing (InsightFace inswapper_128). When enabled the
    # worker swaps the user's detected face onto the Gemini-generated portrait
    # as a final identity-preservation step.
    face_swap_enabled: bool = True
    face_swap_model_path: Path = (
        Path(__file__).resolve().parent.parent / "models" / "inswapper_128.onnx"
    )

    # ICP filing number (legally required for China-facing sites). Set in env
    # for production; empty hides the footer line.
    icp_beian: str = ""

    _session_secret_generated: bool = PrivateAttr(default=False)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def model_post_init(self, __context):
        # Only derive sub-dirs if they weren't explicitly set via env.
        # NOTE: a `Path("")` default is a broken sentinel — it str-ifies to "."
        # (truthy), so `if not str(self.output_dir)` would NEVER fire and both
        # dirs would silently resolve to the CWD, scattering user face photos
        # at the project root. We use None instead, which is unambiguous.
        if self.upload_dir is None:
            self.upload_dir = self.data_dir / "uploads"
        if self.output_dir is None:
            self.output_dir = self.data_dir / "output"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Guard against the sentinel regression: if either dir resolves to the
        # CWD, user face photos would be scattered at the project root (outside
        # the gitignored data/ tree) — fail loudly at startup instead.
        _cwd = Path.cwd().resolve()
        for _d in (self.upload_dir, self.output_dir):
            if _d.resolve() == _cwd:
                raise RuntimeError(
                    f"{_d} resolved to the working directory; refusing to store "
                    f"user data at the project root. Set DATA_DIR/OUTPUT_DIR "
                    f"explicitly or fix the sentinel logic."
                )
        # Generate a per-process secret if none provided.
        if not self.session_secret_key:
            self.session_secret_key = secrets.token_urlsafe(48)
            self._session_secret_generated = True

    def production_readiness_errors(self) -> list[str]:
        """Return release-blocking configuration errors for production."""
        if self.app_environment != "production":
            return []
        return self.launch_readiness_errors()

    def launch_readiness_errors(self) -> list[str]:
        """Return blockers for a real paid production launch in any environment."""
        errors = []
        if self.app_environment != "production":
            errors.append("APP_ENVIRONMENT must be production")
        if self._session_secret_generated:
            errors.append("SESSION_SECRET_KEY must be persistent in production")
        if self.gemini_backend == "openrouter" and not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is missing")
        if self.gemini_backend == "siliconflow" and not self.siliconflow_api_key:
            errors.append("SILICONFLOW_API_KEY is missing")
        if self.payment_mock_enabled:
            errors.append("PAYMENT_MOCK_ENABLED must be off in production")
        if not self.apple_client_id:
            errors.append("APPLE_CLIENT_ID is missing")
        if not self.apple_bundle_id:
            errors.append("APPLE_BUNDLE_ID is missing")
        if not self.apple_iap_product_id:
            errors.append("APPLE_IAP_PRODUCT_ID is missing")
        if self.apple_iap_environment != "production":
            errors.append("APPLE_IAP_ENVIRONMENT must be production")
        if self.apple_app_id is None:
            errors.append("APPLE_APP_ID is missing")
        root_certs = list(self.apple_root_cert_dir.glob("*.cer")) if self.apple_root_cert_dir.exists() else []
        if not root_certs:
            errors.append("Apple root certificates are missing")
        if self.face_swap_enabled and not self.face_swap_model_path.exists():
            errors.append(f"face-swap model is missing: {self.face_swap_model_path}")
        face_swap_emap = self.face_swap_model_path.with_suffix(".emap.npy")
        if self.face_swap_enabled and not face_swap_emap.exists():
            errors.append(f"face-swap embedding map is missing: {face_swap_emap}")
        if len(self.support_admin_token) < 32:
            errors.append("SUPPORT_ADMIN_TOKEN must contain at least 32 characters")
        return errors

    def web_payment_readiness_errors(self) -> list[str]:
        """Optional Paddle readiness for the companion website."""
        errors = []
        if self.paddle_environment != "production":
            errors.append("PADDLE_ENVIRONMENT must be production")
        if not self.paddle_api_key:
            errors.append("PADDLE_API_KEY is missing")
        if not self.paddle_client_token:
            errors.append("PADDLE_CLIENT_TOKEN is missing")
        if not self.paddle_webhook_secret:
            errors.append("PADDLE_WEBHOOK_SECRET is missing")
        if not self.paddle_price_standard_id:
            errors.append("PADDLE_PRICE_STANDARD_ID is missing")
        if not self.paddle_price_premium_id:
            errors.append("PADDLE_PRICE_PREMIUM_ID is missing")
        return errors


settings = Settings()
