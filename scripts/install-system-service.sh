#!/usr/bin/env bash
# Install ARCHGPU bridge as a system service with:
# - Code under /opt
# - Data under /var/lib
# - Service enabled on boot
#
# This script requires sudo/root privileges.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-/opt/archgpu-ollama-bridge}"
DATA_DIR="${DATA_DIR:-/var/lib/archgpu-ollama-bridge}"
STATE_DIR="${DATA_DIR}/state"
MODELS_DIR="${DATA_DIR}/models"
SERVICE_NAME="archgpu-bridge.service"
SERVICE_USER="${SERVICE_USER:-archgpu-bridge}"
SERVICE_GROUP="${SERVICE_GROUP:-archgpu-bridge}"
ENV_FILE="/etc/default/archgpu-bridge"
SYSTEMD_FILE="/etc/systemd/system/${SERVICE_NAME}"

SOURCE_MODELS_DIR="${SOURCE_MODELS_DIR:-$HOME/llm/models}"
BACKEND_IMAGE="${BACKEND_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-intel}"
SERVICE_PORT="${SERVICE_PORT:-8000}"

usage() {
  cat <<EOF
Usage: scripts/install-system-service.sh [options]

Options:
  --install-dir <path>      Install code path (default: ${INSTALL_DIR})
  --data-dir <path>         Persistent data path (default: ${DATA_DIR})
  --source-models <path>    Existing models source directory to migrate
                            (default: ${SOURCE_MODELS_DIR})
  --backend-image <image>   llama.cpp server image (default: ${BACKEND_IMAGE})
  --port <number>           bridge listen port (default: ${SERVICE_PORT})
  --service-user <name>     system user for service (default: ${SERVICE_USER})
  --service-group <name>    system group for service (default: ${SERVICE_GROUP})
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir) shift; INSTALL_DIR="${1:?missing value}" ;;
    --data-dir) shift; DATA_DIR="${1:?missing value}"; STATE_DIR="${DATA_DIR}/state"; MODELS_DIR="${DATA_DIR}/models" ;;
    --source-models) shift; SOURCE_MODELS_DIR="${1:?missing value}" ;;
    --backend-image) shift; BACKEND_IMAGE="${1:?missing value}" ;;
    --port) shift; SERVICE_PORT="${1:?missing value}" ;;
    --service-user) shift; SERVICE_USER="${1:?missing value}" ;;
    --service-group) shift; SERVICE_GROUP="${1:?missing value}" ;;
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
    --source-models "$SOURCE_MODELS_DIR" \
    --backend-image "$BACKEND_IMAGE" \
    --port "$SERVICE_PORT" \
    --service-user "$SERVICE_USER" \
    --service-group "$SERVICE_GROUP"
fi

if ! [[ "$SERVICE_PORT" =~ ^[0-9]+$ ]] || (( SERVICE_PORT < 1 || SERVICE_PORT > 65535 )); then
  echo "[fail] Invalid --port value: ${SERVICE_PORT}"
  exit 1
fi

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

log "Checking dependencies"
apt-get update -y
apt-get install -y rsync python3 python3-venv python3-pip docker.io curl jq

if ! getent group "$SERVICE_GROUP" >/dev/null 2>&1; then
  log "Creating service group ${SERVICE_GROUP}"
  groupadd --system "$SERVICE_GROUP"
fi

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  log "Creating service user ${SERVICE_USER}"
  useradd --system --home "$DATA_DIR" --shell /usr/sbin/nologin --gid "$SERVICE_GROUP" "$SERVICE_USER"
fi

if getent group docker >/dev/null 2>&1; then
  log "Adding ${SERVICE_USER} to docker group"
  usermod -aG docker "$SERVICE_USER"
fi

log "Creating install/data directories"
mkdir -p "$INSTALL_DIR" "$MODELS_DIR" "$STATE_DIR"

log "Copying bridge source into ${INSTALL_DIR}"
rsync -a --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".pkg/" \
  --exclude "__pycache__/" \
  --exclude "data/" \
  "$ROOT/" "$INSTALL_DIR/"

if [[ -d "$SOURCE_MODELS_DIR" ]]; then
  log "Migrating existing models from ${SOURCE_MODELS_DIR}"
  rsync -a --ignore-existing "$SOURCE_MODELS_DIR/" "$MODELS_DIR/"
else
  log "No source models directory found at ${SOURCE_MODELS_DIR}; skipping migration"
fi

log "Setting ownership and permissions"
chown -R "$SERVICE_USER:$SERVICE_GROUP" "$DATA_DIR"
chown -R root:root "$INSTALL_DIR"

log "Creating virtualenv + installing bridge"
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --upgrade pip wheel setuptools
"$INSTALL_DIR/.venv/bin/pip" install "$INSTALL_DIR"

log "Generating runtime models config"
python3 - <<PY
from pathlib import Path
import yaml

install_dir = Path("${INSTALL_DIR}")
models_dir = Path("${MODELS_DIR}")
src = install_dir / "config" / "models.yaml"
dst = Path("${STATE_DIR}") / "models.yaml"

data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
models = data.get("models", [])
for item in models:
    gguf = item.get("gguf_path")
    if gguf:
        item["gguf_path"] = str(models_dir / Path(gguf).name)

dst.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

if [[ ! -f "$STATE_DIR/downloaded_models.yaml" ]]; then
  cat > "$STATE_DIR/downloaded_models.yaml" <<'EOF'
models: []
EOF
fi

if [[ ! -f "$STATE_DIR/runtime-state.json" ]]; then
  cat > "$STATE_DIR/runtime-state.json" <<'EOF'
{}
EOF
fi

chown -R "$SERVICE_USER:$SERVICE_GROUP" "$STATE_DIR"

HF_CACHE_PATH="${STATE_DIR}/hf_gguf_index.json"
REPO_HF_CACHE="${ROOT}/data/hf_gguf_index.json"
if [[ -f "$REPO_HF_CACHE" ]]; then
  log "Seeding HF index cache from ${REPO_HF_CACHE}"
  install -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0644 "$REPO_HF_CACHE" "$HF_CACHE_PATH"
elif [[ ! -f "$HF_CACHE_PATH" ]]; then
  log "No HF index cache found; live catalogue will rely on HF API until cache is built"
fi

EXISTING_HF_TOKEN=""
if [[ -f "$ENV_FILE" ]]; then
  EXISTING_HF_TOKEN="$(grep -E '^ARCHGPU_BRIDGE_HF_TOKEN=' "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
fi
if [[ -z "$EXISTING_HF_TOKEN" ]]; then
  INVOKING_USER="${SUDO_USER:-$USER}"
  if [[ -n "$INVOKING_USER" && "$INVOKING_USER" != "root" ]]; then
    USER_ENV_FILE="$(getent passwd "$INVOKING_USER" | cut -d: -f6)/.config/archgpu-bridge/environment"
    if [[ -f "$USER_ENV_FILE" ]]; then
      EXISTING_HF_TOKEN="$(grep -E '^ARCHGPU_BRIDGE_HF_TOKEN=' "$USER_ENV_FILE" | tail -1 | cut -d= -f2- || true)"
    fi
  fi
fi

log "Writing service environment file"
{
  cat <<EOF
ARCHGPU_BRIDGE_LOG_LEVEL=INFO
ARCHGPU_BRIDGE_HOST=0.0.0.0
ARCHGPU_BRIDGE_PORT=${SERVICE_PORT}
ARCHGPU_BRIDGE_REGISTRY_PATH=${STATE_DIR}/models.yaml
ARCHGPU_BRIDGE_STATE_PATH=${STATE_DIR}/runtime-state.json
ARCHGPU_BRIDGE_DYNAMIC_MODELS_PATH=${STATE_DIR}/downloaded_models.yaml
ARCHGPU_BRIDGE_BACKEND_MODELS_HOST_DIR=${MODELS_DIR}
ARCHGPU_BRIDGE_BACKEND_IMAGE=${BACKEND_IMAGE}
ARCHGPU_BRIDGE_BACKEND_STARTUP_TIMEOUT_SECONDS=300
ARCHGPU_BRIDGE_HF_DISCOVERY_ENABLED=true
ARCHGPU_BRIDGE_HF_INDEX_MODE=live_page
ARCHGPU_BRIDGE_HF_INDEX_CACHE_PATH=${HF_CACHE_PATH}
EOF
  if [[ -n "$EXISTING_HF_TOKEN" ]]; then
    printf 'ARCHGPU_BRIDGE_HF_TOKEN=%s\n' "$EXISTING_HF_TOKEN"
  fi
} > "$ENV_FILE"

log "Writing systemd unit ${SYSTEMD_FILE}"
cat > "$SYSTEMD_FILE" <<EOF
[Unit]
Description=ARCHGPU OLLAMA Bridge (system service)
After=network-online.target docker.service
Wants=network-online.target docker.service

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${INSTALL_DIR}/.venv/bin/python -m archgpu_ollama_bridge
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

log "Reloading and enabling service"
systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"

log "Service status"
systemctl --no-pager --full status "$SERVICE_NAME" || true

log "Done. Bridge installed system-wide."
echo "Code:        ${INSTALL_DIR}"
echo "Data/models: ${DATA_DIR}"
echo "Service:     ${SERVICE_NAME}"
echo "Port:        ${SERVICE_PORT}"
