# FlashShot — Overseas VPS Deploy Runbook (66.175.213.242)

> **Hard constraint (verbatim, non-negotiable):**
> *"我们的服务器是 ssh root@66.175.213.242，上面还有别的项目，不要动其他项目"*
>
> This box is **shared**. Syntropy (trading firm), Caddy (terminates TLS for
> other sites), Tailscale, and fwxt-relay run here. **Every step below is
> additive** — we add a `flashshot` user, two systemd units on non-conflicting
> loopback ports, and one appended Caddy site block. We never edit, restart,
> reorder, or remove anything belonging to another project. The rollback (§11)
> removes *only* FlashShot and leaves every co-tenant untouched.
>
> **Generation backend:** the pipeline drives Qwen Image Edit and Qwen3.6 VLM
> via the **SiliconFlow REST API** - stateless HTTPS calls per turn. There is
> **no Chrome / no browser / no VNC login** anymore (that was retired with the
> Chrome→API pivot; it was the single biggest source of "needs constant
> debugging"). The production generation backend is SiliconFlow; its only
> generation secret is `SILICONFLOW_API_KEY`.

This runbook is a set of commands **you run yourself** on the VPS. Claude does
not SSH into the shared box. Read every block before pasting it.

---

## 0. What runs where

```
                ┌─────────────────────────────────────────────────────┐
                │            66.175.213.242  (shared VPS)              │
                │                                                     │
   :443 ──────▶ │  Caddy  (CO-TENANT — we only APPEND one site block) │
                │   flashshot.top /api/* ──┐                          │
                │   flashshot.top /ws/*  ──┼─▶ 127.0.0.1:8001  FastAPI │
                │   flashshot.top *      ──┴─▶ 127.0.0.1:3001  Next.js │
                │                                                     │
                │  flashshot-api.service   127.0.0.1:8001  FastAPI    │
                │   └─ generation + judge turn call out over HTTPS ─┐ │
                │                                                   ▼ │
                │  flashshot-web.service   127.0.0.1:3001  Next.js    │
                │                                                     │
                │  CO-TENANTS (untouched): Syntropy / Tailscale /     │
                │  fwxt-relay / other Caddy sites                     │
                └──────────────────────┬──────────────────────────────┘
                                       │ HTTPS egress only (no ingress)
                                       ▼
                       api.siliconflow.cn/v1 → Qwen Image Edit + Qwen3.6 VLM
```

| Unit | Port | Binds to | Purpose |
|------|------|----------|---------|
| `flashshot-api.service` | 8001 | **127.0.0.1** | FastAPI: pipeline + Paddle payment + WS; calls SiliconFlow over HTTPS |
| `flashshot-web.service` | 3001 | **127.0.0.1** | Next.js marketing + upload/review UI |

Both bind to **loopback only** — they are never directly reachable from
the internet. Caddy is the sole ingress and multiplexes by hostname, so a
misconfiguration in the flashshot block physically cannot affect other sites.

**RAM budget (honest):** image decoding, InsightFace, ONNX and post-processing
can spike well above steady-state memory. On this shared 3.8 GB host the units
are capped at API 1.5 GB + Web 0.4 GB ≈ **1.9 GB ceiling**. The in-process queue
serializes generation to one job at a time. `preflight.sh` returns NO-GO below
2.2 GB available; do not raise these caps without re-auditing every co-tenant.

---

## 1. Preflight (read-only, run FIRST)

```bash
# From your laptop — push the deploy dir up, then audit. NOTHING is modified.
scp -r deploy/overseas-vps root@66.175.213.242:/root/flashshot-deploy

ssh root@66.175.213.242
cd /root/flashshot-deploy
sudo bash preflight.sh
```

Read the report. Proceed only when **RESULT: GO** (zero `[FAIL]`). Clear every
`[WARN]` you can. In particular confirm:
- ports **8001 / 3001** are free,
- `node` (built under `/opt/flashshot/.nvm` in §4), `python3`+`venv`, and
  `caddy` are installed,
- the **Caddyfile path** (usually `/etc/caddy/Caddyfile`) is found.
- No `google-chrome` / `xvfb` / `x11vnc` is needed — generation is an API call.

If anything is missing, install it (§2) — these installs are standard apt /
NodeSource / Google-Chrome steps; they do not touch co-tenant services.

---

## 2. Install system prerequisites (only what's missing)

```bash
# Python venv module (Ubuntu 24.04 splits it out). No xvfb/x11vnc, no Chrome —
# generation drives the SiliconFlow API, so there is nothing graphical to install
# or log into.
sudo apt-get update
sudo apt-get install -y python3.12-venv

# Node 20 LTS — installed nvm-ISOLATED under the flashshot user in §4, NEVER
# globally to /usr/bin/node. This box is SHARED (Syntropy/Caddy/Tailscale/
# fwxt-relay); a global NodeSource setup_20.x was forbidden as a co-tenant risk
# (it replaces the system Node). So do NOT run NodeSource here. §4 builds Node
# under /opt/flashshot/.nvm and symlinks it to /opt/flashshot/bin/node, which is
# what flashshot-web.service ExecStart points to. preflight.sh will WARN about
# /usr/bin/node missing — that WARN is expected and does not block GO.
```

Re-run `preflight.sh` — every prerequisite line should now be `✓`.

---

## 3. Create the isolated `flashshot` user + paths

```bash
# System user, no login shell, no sudo, no home clutter. -r = system account.
sudo useradd --system --create-home --shell /usr/sbin/nologin \
  --home-dir /opt/flashshot flashshot

# App code lives under /opt/flashshot (its home). State lives under /var/lib/flashshot.
# (No chrome-profile dir anymore — the retired Chrome stack was its only consumer.)
sudo mkdir -p /opt/flashshot /var/lib/flashshot/data
sudo chown -R flashshot:flashshot /opt/flashshot /var/lib/flashshot
```

---

## 4. Lay down the code, venv, and build the web app

From your **laptop**, push the two repos:

```bash
# Backend pipeline (includes server/ + prompts.json + deploy/).
#
# CRITICAL: '.env' MUST be excluded. The laptop headshot_pipeline/.env holds DEV
# secrets, while /opt/flashshot/.env on the server holds PROD secrets (prod
# SiliconFlow key, SESSION_SECRET_KEY, Paddle keys). Without this exclude, the
# rsync + `cp -a` below silently overwrites prod secrets with dev ones —
# breaking production on the next restart.
rsync -avz --exclude '__pycache__' --exclude '.venv' --exclude 'output' \
  --exclude '.env' --exclude '.git' --exclude '.pytest_cache' \
  /Users/lizeyan/Desktop/mvp_1/headshot_pipeline/ \
  root@66.175.213.242:/tmp/flashshot-src-pipeline/

# Frontend source (we will build ON the VPS for the correct arch/runtime).
rsync -avz --exclude 'node_modules' --exclude '.next' \
  /Users/lizeyan/Desktop/mvp_1/headshot-landing/ \
  root@66.175.213.242:/tmp/flashshot-src-web/
```

On the **VPS** as root, move them into place and build:

```bash
# ── Backend ──
sudo cp -a /tmp/flashshot-src-pipeline/. /opt/flashshot/
sudo chown -R flashshot:flashshot /opt/flashshot

sudo -u flashshot bash -lc '
  set -e
  cd /opt/flashshot
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  # Test deps. pytest-asyncio is mandatory — without it async tests are
  # silently skipped (PytestUnknownMarkWarning) and the gate looks green
  # while exercising none of the async pipeline.
  .venv/bin/pip install -r requirements-dev.txt
  # Sanity gate. Run against an ISOLATED DATA_DIR so the suite never touches
  # the live /var/lib/flashshot DBs. (Running pytest as root against the real
  # data_dir once left root-owned learning_layer.db files that the flashshot
  # service then could not write — "attempt to write a readonly database".)
  DATA_DIR="$(mktemp -d)" .venv/bin/python -m pytest tests/ -q
'

# ── Frontend ──
sudo mkdir -p /opt/flashshot/web
sudo cp -a /tmp/flashshot-src-web/. /opt/flashshot/web/
sudo chown -R flashshot:flashshot /opt/flashshot/web
sudo -u flashshot bash -lc '
  set -e
  cd /opt/flashshot/web
  # No build-time public env needed: the app calls the API via the relative
  # "/api" path, which Caddy routes to the FastAPI process on the same origin.
  npm ci
  npm run build
'
```

> If `npm ci` complains about a lockfile, run `npm install` instead and commit
> the regenerated lockfile back on your laptop afterward.

---

## 5. Write the production `.env`

```bash
sudo -u flashshot install -m600 /dev/stdin /opt/flashshot/.env <<'EOF'
HOST=127.0.0.1
PORT=8001
APP_ENVIRONMENT=staging
CORS_ORIGINS=["http://localhost:3000","https://flashshot.top","https://www.flashshot.top"]

# Generate on the box: python3 -c "import secrets;print(secrets.token_urlsafe(48))"
SESSION_SECRET_KEY=REPLACE_WITH_OUTPUT_OF_THE_ABOVE_COMMAND

# ── Generation backend (SiliconFlow API - NO browser, NO login) ──
# Paste your SiliconFlow key. The worker refuses to start if this
# is empty: there is no logged-in Chrome session to fall back to anymore.
GEMINI_BACKEND=siliconflow
SILICONFLOW_API_KEY=PASTE_YOUR_SILICONFLOW_KEY_HERE
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_IMAGE_MODEL=Qwen/Qwen-Image-Edit-2509
SILICONFLOW_TEXT_TO_IMAGE_MODEL=Qwen/Qwen-Image
SILICONFLOW_JUDGE_MODEL=Qwen/Qwen3.6-35B-A3B
GEMINI_WAIT_TIMEOUT=180

PAYMENT_MOCK_ENABLED=0
PADDLE_ENVIRONMENT=sandbox
PADDLE_API_KEY=
PADDLE_CLIENT_TOKEN=
PADDLE_WEBHOOK_SECRET=
PADDLE_PRICE_STANDARD_ID=
PADDLE_PRICE_PREMIUM_ID=
PADDLE_RETURN_URL=https://flashshot.top/checkout

DATA_DIR=/var/lib/flashshot/data
RETENTION_DAYS=7
MAX_FILE_SIZE_MB=10
MAX_PHOTOS=6
MIN_PHOTOS=4

# Overseas build: leave EMPTY so the footer hides the ICP block.
ICP_BEIAN=
EOF
```

Fill the **SiliconFlow key** after `SILICONFLOW_API_KEY=`
and the **Paddle values** (from §9). Generate the session secret:

```bash
sudo -u flashshot bash -lc 'python3 -c "import secrets;print(secrets.token_urlsafe(48))"'
# paste the output after SESSION_SECRET_KEY= in /opt/flashshot/.env
```

---

## 6. Generation backend — SiliconFlow API (no login step)

> **This replaces the old "one-time Chrome / VNC login" step entirely.** There
> is no browser, no Google login, no VNC, no profile to expire. Generation is a
> stateless HTTPS calls to SiliconFlow: Qwen Image Edit generates and repairs
> portraits, while Qwen3.6 VLM judges candidate quality.

The only thing the backend needs is the **SiliconFlow API key** you already put
in `/opt/flashshot/.env` (§5: `SILICONFLOW_API_KEY=`). Verify it is set and the
worker constructs cleanly:

```bash
sudo -u flashshot bash -lc '
  set -e
  cd /opt/flashshot
  .venv/bin/python -c "
from server.config import settings
from server.gemini_worker import GeminiWorker
assert settings.siliconflow_api_key, \"SILICONFLOW_API_KEY is empty in .env\"
GeminiWorker().connect()   # validates auth and both configured model IDs
print(\"OK - SiliconFlow image and judge models ready\")
"
'
```

If that prints `OK - SiliconFlow image and judge models ready`, §6 is done.
If it raises `SILICONFLOW_API_KEY is not set`, go back and fill §5.

### Retiring a previously-installed Chrome stack (only if this box ever ran the old units)

If `flashshot-chrome.service` was installed here during an earlier Chrome-era
deploy, retire it now — it is dead weight (the worker no longer attaches to
CDP) and its 1.6 GB RAM cap is wasted headroom:

```bash
# Stop + disable the retired chrome unit. Does NOT touch api/web/co-tenants.
sudo systemctl disable --now flashshot-chrome 2>/dev/null || true
sudo rm -f /etc/systemd/system/flashshot-chrome.service
sudo systemctl daemon-reload

# Drop the now-unused Chrome profile (held stale Google session cookies).
sudo rm -rf /var/lib/flashshot/chrome-profile

# (Optional) remove the VNC + Chrome packages the login step once needed.
sudo apt-get remove -y x11vnc xvfb google-chrome-stable 2>/dev/null || true
sudo apt-get autoremove -y
```

None of these touches `flashshot-api`, `flashshot-web`, Caddy's other site
blocks, Syntropy, Tailscale, or fwxt-relay.

---

## 7. Install + enable the two systemd units

```bash
sudo install -m644 /root/flashshot-deploy/flashshot-api.service /etc/systemd/system/
sudo install -m644 /root/flashshot-deploy/flashshot-web.service /etc/systemd/system/
sudo systemctl daemon-reload

# api first (it serves /api/health the web app polls), then web.
sudo systemctl enable --now flashshot-api
sleep 2
sudo systemctl enable --now flashshot-web

# Both should be active (running):
sudo systemctl --no-pager --type=service | grep flashshot
```

If `flashshot-api` enters `activating` then `failed`, check
`journalctl -u flashshot-api -n 50 --no-pager`. The most common cause is a
malformed `/opt/flashshot/.env` (pydantic-settings is strict) or an empty
`SILICONFLOW_API_KEY`. Startup lists the account's available models without
generating a paid image; an invalid model ID, outbound-network failure, or
regional provider restriction keeps `/api/ready` at 503. Confirm:

```bash
curl -sf http://127.0.0.1:8001/api/health                    # process liveness
curl -sf http://127.0.0.1:8001/api/ready                     # generation readiness
curl -sf http://127.0.0.1:3001/ -o /dev/null -w '%{http_code}\n'   # 200
```

---

## 8. Append the Caddy site block + DNS

**DNS first.** Add A records for **both** `flashshot.top` and
`www.flashshot.top` pointing at `66.175.213.242`. Wait for propagation
(`dig +short flashshot.top` must return the IP) before reloading Caddy, or the
ACME challenge for this block fails. A failure here does not touch other sites'
certs.

**Append the block** to the existing Caddyfile — never overwrite it:

```bash
# Make a timestamped backup of the EXISTING Caddyfile (co-tenant safety).
sudo cp -a /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak.$(date -u +%Y%m%dT%H%M%S)

# Append the flashshot block verbatim.
sudo tee -a /etc/caddy/Caddyfile >/dev/null < /root/flashshot-deploy/Caddyfile.flashshot

# Validate BEFORE reload. A bad config is caught here; reload only runs on PASS.
sudo caddy validate --config /etc/caddy/Caddyfile \
  && sudo systemctl reload caddy \
  || echo "validate FAILED — Caddy NOT reloaded, other sites unaffected"
```

Confirm TLS came up:

```bash
curl -sfI https://flashshot.top/api/health | head -5    # HTTP/2 200, valid cert
curl -sfI https://flashshot.top/api/ready | head -5     # HTTP/2 200, ready to accept work
curl -sfI https://flashshot.top/             | head -5    # served by Next.js
```

If TLS isn't ready within ~30s, watch `journalctl -u caddy -n 50 --no-pager`
for ACME errors (usually DNS not propagated yet). Caddy will keep retrying.

---

## 9. Configure Paddle (sandbox → production)

In the Paddle dashboard (sandbox first):

1. **Developer Tools → Authentication:** create an API key → put it in
   `PADDLE_API_KEY=`.
2. **Notifications:** create a webhook destination
   `https://flashshot.top/api/payments/paddle/webhook`, subscribe to
   `transaction.completed` + `transaction.paid`. Copy the **webhook secret** →
   `PADDLE_WEBHOOK_SECRET=`.
3. **Products + Prices:** create two products with one-time prices **$5.00
   (Standard)** and **$10.00 (Pro)**. Copy each `pri_…` →
   `PADDLE_PRICE_STANDARD_ID=` / `PADDLE_PRICE_PREMIUM_ID=`.

Keep `APP_ENVIRONMENT=staging` with `PADDLE_ENVIRONMENT=sandbox`, restart the API, and run a sandbox
checkout end-to-end (upload → pay with Paddle's sandbox card → webhook marks
the session paid → download). When green, flip `PADDLE_ENVIRONMENT=production`
with the **production** API key + price IDs, set `APP_ENVIRONMENT=production`,
and restart the API again. Production readiness remains 503 until all required
secrets, price IDs, the face-swap model and generation worker are present.

```bash
# After editing .env:
sudo systemctl restart flashshot-api
# Confirm Paddle is wired:
curl -s https://flashshot.top/api/config/public   # smoke
journalctl -u flashshot-api -n 5 --no-pager | grep Paddle   # "Paddle: configured"
```

Webhook signature verification is the **only** tier-upgrade gate (see
`server/router_payment.py::verify_paddle_signature`); client polling is
read-only. The 14 regression tests in `tests/test_paddle_payment.py` cover
every forgery class.

---

## 10. End-to-end smoke (the real gate)

1. Open `https://flashshot.top`, upload 2–8 photos, pick a style, generate.
   Watch `journalctl -u flashshot-api -f` for the resemblance-judge loop
   (score ≥ 8/10, up to 3 iterations).
2. Upgrade via Paddle (sandbox card `4242 4242 4242 4242`); confirm the session
   flips to the paid tier after the webhook.
3. Test post-processing (ID crop, background swap) and HD download.
4. Wait > 1 hour, confirm the retention sweep is scheduled (it runs hourly;
   logs `🗑 retention sweep`).

---

## 11. Rollback — remove FlashShot WITHOUT touching co-tenants

```bash
# 1. Stop + disable the two FlashShot units.
sudo systemctl disable --now flashshot-web flashshot-api

# 2. Remove ONLY the flashshot lines from Caddy. Restore the backup you made:
sudo cp -a /etc/caddy/Caddyfile.bak.* /etc/caddy/Caddyfile   # pick the right timestamp
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy

# 3. (Optional) remove the units + user + data.
sudo rm -f /etc/systemd/system/flashshot-*.service && sudo systemctl daemon-reload
sudo rm -rf /opt/flashshot /var/lib/flashshot
sudo userdel flashshot
```

None of these steps restart Caddy's *other* site blocks, Syntropy, Tailscale,
or fwxt-relay. Step 2 reloads Caddy only after `validate` passes.

---

## 12. Known limitations / ops notes

- **Serialized generation.** The in-process job queue runs one generation at a
  time (keeps SiliconFlow spend + latency predictable, and fits the shared-box
  RAM budget). The `--workers` count in `flashshot-api.service` is a capacity
  dial, not a correctness constraint — raise it only with confirmed free RAM.
- **SiliconFlow is the single external dependency for generation.** No Google
  login, no profile, no expiry to babysit. If generation jobs start failing,
  check (a) `SILICONFLOW_API_KEY` still set in `/opt/flashshot/.env`, (b) the
  key's credit balance / rate limit on the SiliconFlow dashboard, (c)
  `journalctl -u flashshot-api` for `SiliconFlowError` / 429 / 5xx. The client
  retries twice on transient 429/5xx; persistent failures surface as a job
  failure to the user.
- **Generated URLs are downloaded immediately.** SiliconFlow response URLs are
  temporary, so the adapter validates and stores every result as a local PNG
  before the job proceeds.
- **No amount verification on the webhook.** Paddle is the Merchant of Record
  and the price is pinned server-side by `price_id`; the HMAC signature +
  `custom_data.payment_id` binding is the security guarantee. This is
  documented in `server/payment.py`.
- **Resemblance detail strings** are emitted in English by `gemini_worker.py`
  (#68); the judge prompts/parser remain Chinese (battle-tested internal
  contract) and never reach the user.
- **Real-ESRGAN true-HD upscaling** (#56) is still pending; current HD download
  uses the best available resolution from the pipeline.
