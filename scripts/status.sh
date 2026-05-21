#!/usr/bin/env bash
# Unified status check for the ARCHGPU OLLAMA Bridge stack.
#
# To start the bridge, Open WebUI, and re-run these checks: ./scripts/stack.sh up
# (or: make stack)
#
# Verifies that:
#   1. The bridge is up and serving its OpenAI + Ollama endpoints.
#   2. Open WebUI is installed and running as a Docker container.
#   3. Open WebUI can reach the bridge from inside its container.
#   4. Open WebUI is configured to use the bridge (best-effort).
#
# Override defaults via env vars:
#   BRIDGE_URL       (default: http://127.0.0.1:8000)
#   OPENWEBUI_NAME   (default: openwebui)
#   OPENWEBUI_URL    (default: http://127.0.0.1:3000)

set -uo pipefail

BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8000}"
OPENWEBUI_NAME="${OPENWEBUI_NAME:-openwebui}"
OPENWEBUI_URL="${OPENWEBUI_URL:-http://127.0.0.1:3000}"

if [ -t 1 ]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
  BLU=$'\033[0;34m'; DIM=$'\033[2m';   RST=$'\033[0m'
else
  RED=""; GRN=""; YEL=""; BLU=""; DIM=""; RST=""
fi

ok()   { printf "%s[ ok ]%s %s\n"   "$GRN" "$RST" "$*"; }
warn() { printf "%s[warn]%s %s\n"   "$YEL" "$RST" "$*"; }
fail() { printf "%s[fail]%s %s\n"   "$RED" "$RST" "$*"; }
info() { printf "%s[info]%s %s\n"   "$BLU" "$RST" "$*"; }
hdr()  { printf "\n%s== %s ==%s\n"  "$DIM" "$*" "$RST"; }

errors=0
container_api_probes_ok=0
note() { fail "$@"; errors=$((errors + 1)); }

# ---------------------------------------------------------------------------
hdr "Tooling"
# ---------------------------------------------------------------------------
for cmd in docker curl; do
  if command -v "$cmd" >/dev/null 2>&1; then
    ok "$cmd"
  else
    note "$cmd not found"
  fi
done
if command -v jq >/dev/null 2>&1; then ok "jq"; else warn "jq not found (output will be less detailed)"; fi
HAVE_JQ=$(command -v jq >/dev/null 2>&1 && echo 1 || echo 0)

# ---------------------------------------------------------------------------
hdr "Bridge ($BRIDGE_URL)"
# ---------------------------------------------------------------------------
if curl -fsS --max-time 2 "$BRIDGE_URL/health" >/dev/null 2>&1; then
  ok "/health responding"
else
  note "/health not responding"
  echo "       start the bridge with: ${DIM}make run${RST}  (or ${DIM}python3 -m archgpu_ollama_bridge${RST})"
fi

models_json=$(curl -fsS --max-time 5 "$BRIDGE_URL/v1/models" 2>/dev/null || true)
if [ -n "$models_json" ]; then
  if [ "$HAVE_JQ" = 1 ]; then
    ids=$(printf '%s' "$models_json" | jq -r '.data[].id' 2>/dev/null | paste -sd, -)
    ok "/v1/models: ${ids:-<empty>}"
  else
    ok "/v1/models OK"
  fi
else
  note "/v1/models did not respond"
fi

tags_json=$(curl -fsS --max-time 5 "$BRIDGE_URL/api/tags" 2>/dev/null || true)
if [ -n "$tags_json" ]; then
  if [ "$HAVE_JQ" = 1 ]; then
    names=$(printf '%s' "$tags_json" | jq -r '.models[].name' 2>/dev/null | paste -sd, -)
    total=$(printf '%s' "$tags_json" | jq -r '.models | length' 2>/dev/null)
    ok "/api/tags: ${names:-<empty>} (${total:-0} model(s))"
    if [ "${total:-0}" = "0" ]; then
      info "no models yet; pull one with:"
      echo "       ${DIM}curl -N -X POST $BRIDGE_URL/api/pull \\
         -H 'Content-Type: application/json' \\
         -d '{\"name\": \"Qwen/Qwen2.5-Coder-3B-Instruct-GGUF:qwen2.5-coder-3b-instruct-q4_k_m.gguf\"}'${RST}"
    fi
  else
    ok "/api/tags OK"
  fi
else
  note "/api/tags did not respond"
fi

ps_json=$(curl -fsS --max-time 2 "$BRIDGE_URL/api/ps" 2>/dev/null || true)
if [ -n "$ps_json" ] && [ "$HAVE_JQ" = 1 ]; then
  loaded=$(printf '%s' "$ps_json" | jq -r '.models[]?.name' 2>/dev/null | paste -sd, -)
  if [ -z "$loaded" ]; then
    info "loaded models: none (lazy-load on first request)"
  else
    info "loaded models: $loaded"
  fi
fi

# ---------------------------------------------------------------------------
hdr "Open WebUI"
# ---------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  note "docker not available; cannot inspect Open WebUI"
else
  if ! docker ps -a --format '{{.Names}}' | grep -qx "$OPENWEBUI_NAME"; then
    note "container '$OPENWEBUI_NAME' is not installed"
    echo "       try (host networking, simplest):"
    echo "         ${DIM}docker run -d --name $OPENWEBUI_NAME --network host \\
           -v openwebui-data:/app/backend/data \\
           ghcr.io/open-webui/open-webui:main${RST}"
  else
    ok "container '$OPENWEBUI_NAME' installed"

    state=$(docker inspect -f '{{.State.Status}}' "$OPENWEBUI_NAME" 2>/dev/null || echo unknown)
    if [ "$state" = "running" ]; then
      ok "container state: running"
    else
      note "container state: $state"
      echo "       start with: ${DIM}docker start $OPENWEBUI_NAME${RST}"
    fi

    if curl -fsS --max-time 5 "$OPENWEBUI_URL" >/dev/null 2>&1; then
      ok "$OPENWEBUI_URL reachable from host"
    else
      warn "$OPENWEBUI_URL not reachable from host (may still serve clients)"
    fi
  fi
fi

# ---------------------------------------------------------------------------
hdr "Open WebUI -> Bridge connectivity"
# ---------------------------------------------------------------------------
reachable=""
if command -v docker >/dev/null 2>&1 \
   && docker ps --format '{{.Names}}' | grep -qx "$OPENWEBUI_NAME"; then

  network_mode=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$OPENWEBUI_NAME" 2>/dev/null || echo "")
  info "openwebui network mode: ${network_mode:-unknown}"

  bridge_port="${BRIDGE_URL##*:}"
  bridge_port="${bridge_port%%/*}"

  candidates=()
  if [ "$network_mode" = "host" ]; then
    candidates+=("http://127.0.0.1:${bridge_port}")
  fi
  candidates+=("http://host.docker.internal:${bridge_port}")
  candidates+=("http://172.17.0.1:${bridge_port}")

  for url in "${candidates[@]}"; do
    if docker exec "$OPENWEBUI_NAME" curl -fsS --max-time 3 "$url/health" >/dev/null 2>&1 \
       || docker exec "$OPENWEBUI_NAME" wget -qO- --timeout=3 "$url/health" >/dev/null 2>&1; then
      ok "openwebui can reach the bridge at $url"
      reachable="$url"
      break
    fi
  done

  if [ -z "$reachable" ]; then
    note "openwebui cannot reach the bridge from inside its container"
    echo "       options (pick one):"
    echo "         - relaunch openwebui with --network host"
    echo "         - relaunch with --add-host=host.docker.internal:host-gateway"
    echo "         - find a host address the container can reach (docker network inspect bridge)"
  else
    hdr "Open WebUI -> Bridge API (from inside container)"
    v1_out=$(docker exec "$OPENWEBUI_NAME" curl -fsS --max-time 5 "$reachable/v1/models" 2>/dev/null || true)
    if [ -n "$v1_out" ]; then
      if [ "$HAVE_JQ" = 1 ]; then
        n_v1=$(printf '%s' "$v1_out" | jq -r '.data | length' 2>/dev/null || echo "?")
        ids=$(printf '%s' "$v1_out" | jq -r '.data[].id' 2>/dev/null | paste -sd, -)
        ok "GET $reachable/v1/models -> $n_v1 model(s)${ids:+ ($ids)}"
      else
        ok "GET $reachable/v1/models OK (install jq for ids)"
      fi
    else
      warn "could not GET $reachable/v1/models from inside the container"
    fi
    tags_out=$(docker exec "$OPENWEBUI_NAME" curl -fsS --max-time 5 "$reachable/api/tags" 2>/dev/null || true)
    if [ -n "$tags_out" ]; then
      if [ "$HAVE_JQ" = 1 ]; then
        n_tags=$(printf '%s' "$tags_out" | jq -r '.models | length' 2>/dev/null || echo "?")
        tnames=$(printf '%s' "$tags_out" | jq -r '.models[].name' 2>/dev/null | paste -sd, -)
        ok "GET $reachable/api/tags -> $n_tags model(s)${tnames:+ ($tnames)}"
      else
        ok "GET $reachable/api/tags OK"
      fi
    else
      warn "could not GET $reachable/api/tags from inside the container"
    fi
    if [ -n "$v1_out" ] && [ -n "$tags_out" ]; then
      container_api_probes_ok=1
    fi
    info "If the UI still shows no models, set Settings -> Connections to this base: ${DIM}${reachable}/v1${RST} (OpenAI) or ${DIM}${reachable}${RST} (Ollama)"
  fi
else
  info "skipping (openwebui not running)"
fi

# ---------------------------------------------------------------------------
hdr "Open WebUI configured backends (best-effort)"
# ---------------------------------------------------------------------------
config_json=$(curl -fsS --max-time 5 "$OPENWEBUI_URL/api/config" 2>/dev/null || true)
if [ -z "$config_json" ]; then
  if [ "$container_api_probes_ok" = "1" ]; then
    info "public /api/config not available; in-container /v1/models + /api/tags already succeeded (saved URLs live in the DB, not in this response)"
  else
    warn "could not read $OPENWEBUI_URL/api/config (auth/version may differ)"
    echo "       verify manually: Open WebUI -> Settings -> Connections"
  fi
elif [ "$HAVE_JQ" != 1 ]; then
  if [ "$container_api_probes_ok" = "1" ]; then
    info "install jq to inspect Open WebUI config; container API probes already passed"
  else
    warn "install jq to inspect Open WebUI config automatically"
  fi
else
  oai_urls=$(printf '%s' "$config_json"   | jq -r '.. | objects | .openai_api_base_urls? // empty | .[]?' 2>/dev/null | paste -sd, -)
  ollama_urls=$(printf '%s' "$config_json" | jq -r '.. | objects | .ollama_base_urls?     // empty | .[]?' 2>/dev/null | paste -sd, -)
  [ -n "$oai_urls"   ] && ok "OpenAI URLs in public config: $oai_urls"
  [ -n "$ollama_urls" ] && ok "Ollama URLs in public config: $ollama_urls"

  bridge_port="${BRIDGE_URL##*:}"
  bridge_port="${bridge_port%%/*}"
  if printf '%s\n%s' "$oai_urls" "$ollama_urls" | grep -qE ":${bridge_port}(/|$|,)"; then
    ok "public config lists the bridge (port $bridge_port)"
  elif [ "$container_api_probes_ok" = "1" ]; then
    info "public config does not list bridge URLs, but in-container /v1/models and /api/tags already succeeded; DB may still be correct for the UI"
    target="${reachable:-$BRIDGE_URL}"
    echo "       if the UI shows no models, set: ${DIM}${target}/v1${RST} (OpenAI) or ${DIM}${target}${RST} (Ollama)"
  else
    warn "Open WebUI is not pointed at the bridge in public /api/config"
    target="${reachable:-$BRIDGE_URL}"
    echo "       set in UI -> Settings -> Connections -> OpenAI API:"
    echo "         URL:     ${DIM}${target}/v1${RST}"
    echo "         API key: ${DIM}none${RST}"
    echo "       (or use Ollama API with URL: ${DIM}${target}${RST})"
  fi
fi

# ---------------------------------------------------------------------------
hdr "Summary"
# ---------------------------------------------------------------------------
if [ "$errors" -eq 0 ]; then
  ok "stack looks healthy"
  exit 0
else
  fail "$errors check(s) failed"
  exit 1
fi
