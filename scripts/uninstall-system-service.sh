#!/usr/bin/env bash
# Uninstall ARCHGPU bridge system service.
# By default keeps persisted models/state in /var/lib for safety.

set -euo pipefail

SERVICE_NAME="archgpu-bridge.service"
INSTALL_DIR="${INSTALL_DIR:-/opt/archgpu-ollama-bridge}"
DATA_DIR="${DATA_DIR:-/var/lib/archgpu-ollama-bridge}"
ENV_FILE="/etc/default/archgpu-bridge"
SYSTEMD_FILE="/etc/systemd/system/${SERVICE_NAME}"
PURGE_DATA=0

usage() {
  cat <<EOF
Usage: scripts/uninstall-system-service.sh [options]

Options:
  --install-dir <path>   Install dir to remove (default: ${INSTALL_DIR})
  --data-dir <path>      Data dir to remove when --purge-data is used (default: ${DATA_DIR})
  --purge-data           Also remove persisted models/state
  -h, --help             Show this help

By default, model/state data is preserved.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir) shift; INSTALL_DIR="${1:?missing value}" ;;
    --data-dir) shift; DATA_DIR="${1:?missing value}" ;;
    --purge-data) PURGE_DATA=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[fail] Unknown argument: $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[info] Re-running with sudo..."
  exec sudo bash "$0" \
    --install-dir "$INSTALL_DIR" \
    --data-dir "$DATA_DIR" \
    $([[ "$PURGE_DATA" -eq 1 ]] && echo "--purge-data")
fi

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

log "Stopping and disabling ${SERVICE_NAME}"
systemctl stop "$SERVICE_NAME" 2>/dev/null || true
systemctl disable "$SERVICE_NAME" 2>/dev/null || true

if [[ -f "$SYSTEMD_FILE" ]]; then
  log "Removing unit file ${SYSTEMD_FILE}"
  rm -f "$SYSTEMD_FILE"
fi

if [[ -f "$ENV_FILE" ]]; then
  log "Removing env file ${ENV_FILE}"
  rm -f "$ENV_FILE"
fi

log "Reloading systemd daemon"
systemctl daemon-reload

if [[ -d "$INSTALL_DIR" ]]; then
  log "Removing install dir ${INSTALL_DIR}"
  rm -rf "$INSTALL_DIR"
fi

if [[ "$PURGE_DATA" -eq 1 ]]; then
  if [[ -d "$DATA_DIR" ]]; then
    log "Purging data dir ${DATA_DIR}"
    rm -rf "$DATA_DIR"
  fi
else
  log "Keeping data dir ${DATA_DIR} (use --purge-data to remove)"
fi

log "Uninstall complete"
