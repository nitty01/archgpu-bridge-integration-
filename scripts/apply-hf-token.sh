#!/usr/bin/env bash
# Merge ARCHGPU_BRIDGE_HF_TOKEN into the system bridge env and restart the service.
set -euo pipefail

ENV_FILE="/etc/default/archgpu-bridge"
SERVICE_NAME="archgpu-bridge.service"

resolve_user_env() {
  if [[ -n "${ARCHGPU_BRIDGE_USER_ENV:-}" ]]; then
    printf '%s\n' "$ARCHGPU_BRIDGE_USER_ENV"
    return
  fi
  local real_user="${SUDO_USER:-}"
  if [[ -z "$real_user" && "$(id -u)" -eq 0 ]]; then
    real_user="$(logname 2>/dev/null || true)"
  fi
  if [[ -z "$real_user" ]]; then
    real_user="${USER:-}"
  fi
  if [[ -n "$real_user" && "$real_user" != "root" ]]; then
    local home
    home="$(getent passwd "$real_user" | cut -d: -f6)"
    printf '%s/.config/archgpu-bridge/environment\n' "$home"
    return
  fi
  printf '%s/.config/archgpu-bridge/environment\n' "${HOME}"
}

USER_ENV="$(resolve_user_env)"

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo ARCHGPU_BRIDGE_USER_ENV="$USER_ENV" bash "$0"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "[fail] Missing ${ENV_FILE}. Run scripts/install-system-service.sh first."
  exit 1
fi

TOKEN=""
if [[ -f "$USER_ENV" ]]; then
  TOKEN="$(grep -E '^ARCHGPU_BRIDGE_HF_TOKEN=' "$USER_ENV" | tail -1 | cut -d= -f2- || true)"
fi
if [[ -z "$TOKEN" ]]; then
  echo "[fail] Set ARCHGPU_BRIDGE_HF_TOKEN in ${USER_ENV}"
  exit 1
fi

TMP="$(mktemp)"
grep -v '^ARCHGPU_BRIDGE_HF_TOKEN=' "$ENV_FILE" >"$TMP"
printf 'ARCHGPU_BRIDGE_HF_TOKEN=%s\n' "$TOKEN" >>"$TMP"
install -m 0640 -o root -g root "$TMP" "$ENV_FILE"
chmod 640 "$ENV_FILE"
rm -f "$TMP"

systemctl restart "$SERVICE_NAME"
echo "[ok] HF token applied to ${ENV_FILE} and ${SERVICE_NAME} restarted"
