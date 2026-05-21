#!/usr/bin/env bash
# Start the ARCHGPU bridge + Open WebUI and verify wiring, or show status only.
#
# Usage:
#   scripts/stack.sh              # same as "up" (start if needed; does not stop running services)
#   scripts/stack.sh up           # start bridge (if needed), Open WebUI (if needed), then status
#   scripts/stack.sh restart      # stop both, free bridge port, start again, then status (re-connect)
#   scripts/stack.sh status       # checks only (delegates to scripts/status.sh)
#   scripts/stack.sh down         # stop bridge pid (if we started it) + docker stop Open WebUI
#   scripts/stack.sh help
#   scripts/unified-stack.sh      # default: restart + connect (wrapper; see that file)
#
# Environment (override defaults) — see also README "Stack / Open WebUI":
#   BRIDGE_URL                default http://127.0.0.1:8000
#   OPENWEBUI_NAME            default openwebui
#   OPENWEBUI_URL             default http://127.0.0.1:${OPENWEBUI_HOST_PORT}
#   OPENWEBUI_HOST_PORT       default 3000 (published host port in bridge network mode)
#   OPENWEBUI_IMAGE           default ghcr.io/open-webui/open-webui:main
#   OPENWEBUI_DATA_VOLUME     default openwebui-data (or openwebui-archgpu-data if ARCHGPU_UNATTENDED_DEV=1)
#   OPENWEBUI_NETWORK_MODE    default bridge — use host for --network host + 127.0.0.1 bridge URLs
#   RECREATE_OPENWEBUI=1      with "up": remove existing Open WebUI container and create (data volume kept)
#   AUTO_RECREATE_OPENWEBUI=1  if in-container /v1/models still fails after wait, remove & recreate once
#   ARCHGPU_UNATTENDED_DEV=1  fresh-profile dev: default volume openwebui-archgpu-data, WEBUI_AUTH=False,
#                             ENABLE_PERSISTENT_CONFIG=False (local only; no existing users in volume)
#   BRIDGE_WAIT_SECONDS       default 90

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8000}"
OPENWEBUI_NAME="${OPENWEBUI_NAME:-openwebui}"
OPENWEBUI_IMAGE="${OPENWEBUI_IMAGE:-ghcr.io/open-webui/open-webui:main}"
OPENWEBUI_HOST_PORT="${OPENWEBUI_HOST_PORT:-3000}"
RECREATE_OPENWEBUI="${RECREATE_OPENWEBUI:-0}"
AUTO_RECREATE_OPENWEBUI="${AUTO_RECREATE_OPENWEBUI:-0}"
ARCHGPU_UNATTENDED_DEV="${ARCHGPU_UNATTENDED_DEV:-0}"
OPENWEBUI_NETWORK_MODE="${OPENWEBUI_NETWORK_MODE:-bridge}"
BRIDGE_WAIT_SECONDS="${BRIDGE_WAIT_SECONDS:-90}"

if [ "$ARCHGPU_UNATTENDED_DEV" = "1" ]; then
  OPENWEBUI_DATA_VOLUME="${OPENWEBUI_DATA_VOLUME:-openwebui-archgpu-data}"
else
  OPENWEBUI_DATA_VOLUME="${OPENWEBUI_DATA_VOLUME:-openwebui-data}"
fi

OPENWEBUI_URL="${OPENWEBUI_URL:-http://127.0.0.1:${OPENWEBUI_HOST_PORT}}"

bridge_port="${BRIDGE_URL##*:}"
bridge_port="${bridge_port%%/*}"
case "$bridge_port" in
  ''|*[!0-9]*)
    bridge_port=8000
    ;;
esac

if [ -t 1 ]; then
  RED=$'\033[0;31m'; GRN=$'\033[0;32m'; YEL=$'\033[0;33m'
  BLU=$'\033[0;34m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; RST=$'\033[0m'
else
  RED=""; GRN=""; YEL=""; BLU=""; DIM=""; BOLD=""; RST=""
fi

log() { printf "%s\n" "$*"; }
ok()   { printf "%s[ ok ]%s %s\n"   "$GRN" "$RST" "$*"; }
warn() { printf "%s[warn]%s %s\n"   "$YEL" "$RST" "$*"; }
fail() { printf "%s[fail]%s %s\n"   "$RED" "$RST" "$*"; }
info() { printf "%s[info]%s %s\n"   "$BLU" "$RST" "$*"; }
hdr()  { printf "\n%s== %s ==%s\n"  "$DIM" "$*" "$RST"; }

bridge_healthy() {
  curl -fsS --max-time 2 "${BRIDGE_URL}/health" >/dev/null 2>&1
}

setup_pythonpath() {
  if [ -d "$ROOT/.pkg" ]; then
    export PYTHONPATH="$ROOT/.pkg:$ROOT/src"
  else
    export PYTHONPATH="$ROOT/src"
  fi
}

start_bridge_background() {
  mkdir -p "$ROOT/data"
  local logfile="$ROOT/data/bridge.log"
  local pidfile="$ROOT/data/bridge.pid"

  setup_pythonpath
  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found; install Python 3.12+ or activate your venv"
    return 1
  fi

  info "starting bridge in background (logs: $logfile)"
  (
    cd "$ROOT" || exit 1
    nohup env \
      PYTHONPATH="$PYTHONPATH" \
      ARCHGPU_BRIDGE_LOG_LEVEL="${ARCHGPU_BRIDGE_LOG_LEVEL:-INFO}" \
      ARCHGPU_BRIDGE_HOST="${ARCHGPU_BRIDGE_HOST:-0.0.0.0}" \
      ARCHGPU_BRIDGE_PORT="${ARCHGPU_BRIDGE_PORT:-$bridge_port}" \
      python3 -m archgpu_ollama_bridge >>"$logfile" 2>&1 &
    echo $! >"$pidfile"
  )
  local waited=0
  while [ "$waited" -lt "$BRIDGE_WAIT_SECONDS" ]; do
    if bridge_healthy; then
      ok "bridge is up at $BRIDGE_URL"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  fail "bridge did not become healthy within ${BRIDGE_WAIT_SECONDS}s (see $logfile)"
  return 1
}

ensure_bridge() {
  hdr "Bridge"
  if bridge_healthy; then
    ok "already running at $BRIDGE_URL"
    return 0
  fi
  start_bridge_background
}

openwebui_exists() {
  command -v docker >/dev/null 2>&1 && docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$OPENWEBUI_NAME"
}

openwebui_running() {
  command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$OPENWEBUI_NAME"
}

# Build docker run -e / -p / --network args for Open WebUI (shared by create + shared logic).
_openwebui_dockerenv_and_urls() {
  OLLAMA_BASE_URLS_REF=""
  OPENAI_BASE_URLS_REF=""
  if [ "$OPENWEBUI_NETWORK_MODE" = "host" ]; then
    OLLAMA_BASE_URLS_REF="http://127.0.0.1:${bridge_port}"
    OPENAI_BASE_URLS_REF="http://127.0.0.1:${bridge_port}/v1"
  else
    OLLAMA_BASE_URLS_REF="http://host.docker.internal:${bridge_port}"
    OPENAI_BASE_URLS_REF="http://host.docker.internal:${bridge_port}/v1"
  fi
}

create_openwebui() {
  _openwebui_dockerenv_and_urls
  local -a xenv=()
  xenv+=(-e "ENABLE_OLLAMA_API=True")
  xenv+=(-e "OLLAMA_BASE_URLS=${OLLAMA_BASE_URLS_REF}")
  xenv+=(-e "OPENAI_API_BASE_URLS=${OPENAI_BASE_URLS_REF}")
  xenv+=(-e "OPENAI_API_KEYS=local-bridge")
  if [ "$ARCHGPU_UNATTENDED_DEV" = "1" ]; then
    xenv+=(-e "WEBUI_AUTH=False")
    xenv+=(-e "ENABLE_PERSISTENT_CONFIG=False")
    info "ARCHGPU_UNATTENDED_DEV=1: WEBUI_AUTH=False, ENABLE_PERSISTENT_CONFIG=False (local dev; requires empty user DB in volume)"
  fi

  if [ "$OPENWEBUI_NETWORK_MODE" = "host" ]; then
    info "creating Open WebUI (OPENWEBUI_NETWORK_MODE=host, PORT=3000)..."
    docker run -d --name "$OPENWEBUI_NAME" --restart unless-stopped --network host \
      -e PORT=3000 \
      "${xenv[@]}" \
      -v "${OPENWEBUI_DATA_VOLUME}:/app/backend/data" \
      "$OPENWEBUI_IMAGE"
  else
    local hport="${OPENWEBUI_HOST_PORT}"
    info "creating Open WebUI (OPENWEBUI_NETWORK_MODE=bridge, -p ${hport}:8080, host.docker.internal)..."
    docker run -d --name "$OPENWEBUI_NAME" --restart unless-stopped \
      --add-host=host.docker.internal:host-gateway \
      -p "${hport}:8080" \
      -e PORT=8080 \
      "${xenv[@]}" \
      -v "${OPENWEBUI_DATA_VOLUME}:/app/backend/data" \
      "$OPENWEBUI_IMAGE"
  fi
}

recreate_openwebui_if_requested() {
  if [ "$RECREATE_OPENWEBUI" != "1" ]; then
    return 0
  fi
  if ! openwebui_exists; then
    return 0
  fi
  info "stopping and removing Open WebUI container (volume ${OPENWEBUI_DATA_VOLUME} is kept)..."
  docker stop "$OPENWEBUI_NAME" >/dev/null 2>&1 || true
  docker rm "$OPENWEBUI_NAME" >/dev/null 2>&1 || true
}

# Returns 0 if GET /v1/models returns 200 and non-empty body from inside the container.
in_container_v1_models_ok() {
  local base
  for base in "http://127.0.0.1:${bridge_port}" "http://host.docker.internal:${bridge_port}" "http://172.17.0.1:${bridge_port}"; do
    if docker exec "$OPENWEBUI_NAME" curl -fsS --max-time 8 "${base}/v1/models" 2>/dev/null | grep -q '"data"'; then
      return 0
    fi
  done
  return 1
}

wait_openwebui_http() {
  local w=0
  while [ "$w" -lt 120 ]; do
    if curl -fsS --max-time 2 "$OPENWEBUI_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    w=$((w + 2))
  done
  return 1
}

# Wait for Open WebUI app + in-container access to the bridge API (with retries for cold start).
wait_openwebui_bridge_api() {
  local n=0
  while [ "$n" -lt 60 ]; do
    if in_container_v1_models_ok; then
      return 0
    fi
    sleep 2
    n=$((n + 1))
  done
  return 1
}

remove_openwebui_container() {
  docker stop "$OPENWEBUI_NAME" >/dev/null 2>&1 || true
  docker rm "$OPENWEBUI_NAME" >/dev/null 2>&1 || true
}

ensure_openwebui() {
  hdr "Open WebUI"
  if ! command -v docker >/dev/null 2>&1; then
    fail "docker not found; install Docker to run Open WebUI"
    return 1
  fi

  recreate_openwebui_if_requested

  if openwebui_exists; then
    if openwebui_running; then
      ok "container '$OPENWEBUI_NAME' already running"
    else
      info "starting container '$OPENWEBUI_NAME'..."
      docker start "$OPENWEBUI_NAME"
      ok "container started"
    fi
  else
    if ! create_openwebui; then
      fail "docker run failed"
      return 1
    fi
    ok "Open WebUI created; first boot may take a minute (DB migrations)"
  fi

  if wait_openwebui_http; then
    ok "Open WebUI responds at $OPENWEBUI_URL"
  else
    warn "Open WebUI did not respond at $OPENWEBUI_URL yet (it may still be starting)"
  fi

  if ! wait_openwebui_bridge_api; then
    if [ "$AUTO_RECREATE_OPENWEBUI" = "1" ] && openwebui_running; then
      warn "in-container /v1/models check failed; AUTO_RECREATE_OPENWEBUI=1 — recreating container (volume unchanged)..."
      remove_openwebui_container
      if ! create_openwebui; then
        fail "docker run after auto-recreate failed"
        return 1
      fi
      wait_openwebui_http || true
      if ! wait_openwebui_bridge_api; then
        warn "in-container API check still failing after recreate; see docker logs for $OPENWEBUI_NAME"
      else
        ok "in-container bridge API reachable after auto-recreate"
      fi
    else
      info "in-container /v1/models not ready yet (or bridge unreachable from container). Try AUTO_RECREATE_OPENWEBUI=1 or RECREATE_OPENWEBUI=1"
    fi
  else
    ok "in-container GET .../v1/models works (bridge reachable from Open WebUI)"
  fi
}

run_status() {
  bash "$ROOT/scripts/status.sh"
}

do_down() {
  hdr "Stopping"
  if [ -f "$ROOT/data/bridge.pid" ]; then
    local pid
    pid=$(tr -d ' \n' <"$ROOT/data/bridge.pid" || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      info "stopping bridge (pid $pid)..."
      kill "$pid" 2>/dev/null || true
      ok "bridge stop sent"
    else
      warn "stale or missing bridge pid file; not killing"
    fi
    rm -f "$ROOT/data/bridge.pid" 2>/dev/null || true
  else
    info "no $ROOT/data/bridge.pid (bridge was not started by this script, or file removed)"
  fi

  if openwebui_running; then
    info "stopping Open WebUI container..."
    docker stop "$OPENWEBUI_NAME"
    ok "Open WebUI stopped"
  else
    info "Open WebUI not running (or container missing)"
  fi
}

# If something is still listening on the bridge port (e.g. bridge started outside this script), try to stop it.
free_bridge_port() {
  local waited=0
  while [ "$waited" -lt 5 ] && bridge_healthy; do
    if command -v fuser >/dev/null 2>&1; then
      fuser -k -TERM "${bridge_port}/tcp" 2>/dev/null || true
    elif command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -t -iTCP:"$bridge_port" -sTCP:LISTEN 2>/dev/null || true)
      if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
      fi
    else
      warn "bridge still on port $bridge_port; install fuser (psmisc) or lsof, or stop the process manually"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  if bridge_healthy; then
    warn "bridge is still healthy on $BRIDGE_URL; stop the process using port $bridge_port manually"
  else
    info "bridge port $bridge_port is free (or bridge stopped)"
  fi
}

# Stop both, free the bridge port, then full bring-up (clean restart + connection check).
do_restart() {
  log "${BOLD}Restarting stack: stop, then start bridge + Open WebUI + status${RST}"
  do_down
  free_bridge_port
  sleep 1
  log "${BOLD}Starting stack (project: $ROOT)${RST}"
  log "${DIM}OPENWEBUI_NETWORK_MODE=${OPENWEBUI_NETWORK_MODE} OPENWEBUI_DATA_VOLUME=${OPENWEBUI_DATA_VOLUME}${RST}"
  ensure_bridge || exit 1
  ensure_openwebui || true
  hdr "Full status"
  run_status || true
  log ""
  info "API docs: ${DIM}$BRIDGE_URL/docs${RST}  ·  Open WebUI: ${DIM}$OPENWEBUI_URL${RST}"
}

print_help() {
  cat <<EOF
${BOLD}ARCHGPU OLLAMA Bridge — stack helper${RST}

  ${BOLD}scripts/stack.sh${RST} [command]

${BOLD}Commands${RST}
  (default)   Same as ${DIM}up${RST}
  up, start   Ensure bridge, Open WebUI, then ${DIM}status.sh${RST} (no stop if already running)
  restart, reconnect  ${DIM}down${RST} + free bridge port + ${DIM}up${RST} (use this to re-connect a bad state)
  status      Health / connectivity only (no starts)
  down        Stop background bridge (if pid known) and ${DIM}docker stop${RST} Open WebUI
  help        This text

${BOLD}Notable environment${RST}
  OPENWEBUI_NETWORK_MODE   ${DIM}bridge${RST} (default) or ${DIM}host${RST}
  OPENWEBUI_HOST_PORT      host port in bridge mode (default 3000 -> container 8080)
  RECREATE_OPENWEBUI=1     remove & recreate Open WebUI container; keeps data volume
  AUTO_RECREATE_OPENWEBUI=1  if in-container /v1/models fails, same as recreate once
  ARCHGPU_UNATTENDED_DEV=1   use volume ${DIM}openwebui-archgpu-data${RST} by default; WEBUI_AUTH=False;
                             ENABLE_PERSISTENT_CONFIG=False (fresh DB only, local only)

${BOLD}Typical use${RST}
  cd $ROOT
  ${DIM}./scripts/stack.sh up${RST}                 # start if needed
  ${DIM}./scripts/stack.sh restart${RST}            # stop + start + connect check
  ${DIM}./scripts/unified-stack.sh${RST}            # same as restart (no args)
  # Full local automation (new volume, no login):${DIM} ARCHGPU_UNATTENDED_DEV=1 ./scripts/stack.sh up${RST}

${BOLD}Open WebUI${RST}:  ${DIM}$OPENWEBUI_URL${RST}
${BOLD}Bridge${RST}:     ${DIM}$BRIDGE_URL${RST}  (inside container: use host.docker.internal with bridge network)
EOF
}

main() {
  local cmd=${1:-up}
  case "$cmd" in
    help|-h|--help)
      print_help
      ;;
    status)
      run_status
      ;;
    down|stop)
      do_down
      ;;
    restart|reconnect)
      do_restart
      ;;
    up|start|"")
      log "${BOLD}Bringing up stack (project: $ROOT)${RST}"
      log "${DIM}OPENWEBUI_NETWORK_MODE=${OPENWEBUI_NETWORK_MODE} OPENWEBUI_DATA_VOLUME=${OPENWEBUI_DATA_VOLUME}${RST}"
      ensure_bridge || exit 1
      ensure_openwebui || true
      hdr "Full status"
      run_status || true
      log ""
      info "API docs: ${DIM}$BRIDGE_URL/docs${RST}  ·  Open WebUI: ${DIM}$OPENWEBUI_URL${RST}"
      ;;
    *)
      fail "unknown command: $cmd"
      print_help
      exit 1
      ;;
  esac
}

main "$@"
