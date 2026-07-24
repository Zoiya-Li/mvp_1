#!/usr/bin/env bash
# FlashShot deploy — READ-ONLY preflight audit of the shared VPS.
#
# Prints a go/no-go report. TOUCHES NOTHING. Safe to run on a box shared with
# Syntropy / Caddy / Tailscale / fwxt-relay - every command below is read-only
# (systemctl status / list-unit-files, ss -ltn, free, df, whoami, test -e).
# No file is written, no service is started/stopped/reloaded.
#
# Usage:   sudo bash preflight.sh
#   (sudo only so we can read other services' status + port owners; nothing is
#    modified regardless of privilege.)
#
# Exit code: 0 = GO (no FAIL), 1 = NO-GO (≥1 FAIL).

set -u

echo "================================================================"
echo " FlashShot deploy — preflight (READ-ONLY)   $(date -u +%FT%TZ)"
echo "================================================================"

pass=0; warn=0; fail=0
ok()  { echo "  ✓ $1";                 pass=$((pass+1)); }
wa()  { echo "  ! $1  [WARN]";          warn=$((warn+1)); }
no()  { echo "  ✗ $1  [FAIL]";          fail=$((fail+1)); }
hdr() { echo; echo "── $1 ──"; }

# ── OS / arch ──
hdr "Operating system"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  echo "  ${PRETTY_NAME:-unknown}"
  case "${ID:-}/${VERSION_ID:-}" in
    ubuntu/20*|ubuntu/24*) ok "Ubuntu ${VERSION_ID:-?} (tested with the uv-managed Python runtime)" ;;
    ubuntu/2*)  wa "Ubuntu ${VERSION_ID:-?} - verify the uv-managed Python runtime" ;;
    debian/*)   wa "Debian — close to Ubuntu, verify node paths" ;;
    *)          wa "${ID:-unknown} — untested; adapt package names" ;;
  esac
fi
[ "$(uname -m)" = "x86_64" ] && ok "x86_64 arch" || wa "arch $(uname -m) — units assume amd64"

# ── RAM budget ──
hdr "Memory budget (shared 3.8 GB; FlashShot units cap at ~1.9 GB)"
memtotal_m=$(awk '/MemTotal/     {print int($2/1024)}' /proc/meminfo)
memavail_m=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
echo "  MemTotal     ${memtotal_m} MB"
echo "  MemAvailable ${memavail_m} MB"
[ "${memavail_m:-0}" -ge 2200 ] && ok "≥2.2 GB available now" \
  || no "only ${memavail_m} MB available - below the API 1.5 GB + Web 0.4 GB safety budget"

# ── Disk ──
hdr "Disk"
df -h / | awk 'NR==1 || NR==2 {print "  "$0}'
free_gb=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${free_gb:-0}" -ge 8 ]; then ok "≥8 GB free on /"; else wa "only ${free_gb:-?} GB free — uploads + generated output grow"; fi

# ── Co-tenant services we must NOT touch (confirm they're here) ──
hdr "Co-tenant services (expected — will be left untouched)"
found_cotenant=0
for s in caddy tailscaled fwxt-relay; do
  if systemctl list-unit-files "${s}.service" >/dev/null 2>&1; then
    st=$(systemctl is-active "${s}.service" 2>/dev/null)
    [ -n "${st}" ] || st=unknown
    ok "${s}.service present (${st})"
    found_cotenant=1
  fi
done
[ "${found_cotenant}" -eq 1 ] || wa "no known co-tenants detected — still treat the box as shared"

# ── Ports: ours must be FREE, and we must not collide ──
hdr "Port checks (8001 api / 3001 web — loopback only)"
for p in 8001 3001; do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${p}\$"; then
    unit="flashshot-api"
    [ "${p}" = "3001" ] && unit="flashshot-web"
    if systemctl is-active --quiet "${unit}.service"; then
      ok "port ${p} is owned by active ${unit}.service (upgrade-safe)"
    else
      no "port ${p} already in use by a non-FlashShot process"
      ss -ltnp 2>/dev/null | awk -v p=":${p}" '$4 ~ p {print "      "$0}'
    fi
  else
    ok "port ${p} free"
  fi
done

# ── Software prerequisites ──
# NOTE: no google-chrome / xvfb / x11vnc anymore — generation drives the
# SiliconFlow REST API, so there is no browser to install or log into.
hdr "Software"
node_bin=""
for candidate in /usr/bin/node /opt/flashshot/bin/node; do
  if [ -x "${candidate}" ]; then node_bin="${candidate}"; break; fi
done
if [ -n "${node_bin}" ]; then
  ok "node $("${node_bin}" -v 2>/dev/null) at ${node_bin}"
else
  no "node missing from /usr/bin and /opt/flashshot/bin"
fi
if command -v python3 >/dev/null 2>&1; then
  ok "python3 $(python3 --version 2>&1)"
else
  no "python3 missing"
fi
if python3 -c 'import venv' 2>/dev/null; then
  ok "python3-venv available"
else
  no "python3-venv missing (apt: python3.12-venv)"
fi
if command -v caddy >/dev/null 2>&1; then
  ok "caddy present (shared ingress - validate before reload)"
else
  no "caddy missing - nothing terminates TLS for flashshot.top"
fi

# ── Isolation user ──
hdr "Dedicated user"
if id flashshot >/dev/null 2>&1; then
  ok "flashshot user already exists"
else
  wa "flashshot user not yet created — runbook §1 will add it"
fi

# ── Deploy paths ──
hdr "Deploy paths"
for d in /opt/flashshot /var/lib/flashshot /var/lib/flashshot/data; do
  if [ -e "${d}" ]; then
    owner=$(stat -c '%U:%G' "${d}" 2>/dev/null || echo unknown)
    if [ "${owner}" = "flashshot:flashshot" ]; then
      ok "${d} exists with dedicated-user ownership (upgrade-safe)"
    else
      wa "${d} exists with ownership ${owner} - verify before reusing"
    fi
  else
    ok "${d} absent (will be created)"
  fi
done

# ── Caddyfile ──
hdr "Caddyfile"
caddyfile="/etc/caddy/Caddyfile"
if [ -e "${caddyfile}" ]; then
  ok "Caddyfile exists at ${caddyfile}"
  if caddy validate --config "${caddyfile}" >/dev/null 2>&1; then
    ok "Caddy configuration validates"
  else
    no "Caddy configuration validation failed"
  fi
  if grep -q 'flashshot\.top' "${caddyfile}" 2>/dev/null; then
    ok "flashshot.top site block already present (upgrade-safe)"
  else
    wa "flashshot.top site block absent - append Caddyfile.flashshot once"
  fi
else
  no "${caddyfile} is missing"
fi

# ── Result ──
echo
echo "================================================================"
echo " preflight: ${pass} ok / ${warn} warn / ${fail} fail"
if [ "${fail}" -eq 0 ]; then
  echo " RESULT: GO  (clear any [WARN] first; treat [FAIL]=0 as the gate)"
else
  echo " RESULT: NO-GO — fix every [FAIL] line before deploying"
fi
echo "================================================================"
[ "${fail}" -eq 0 ]
