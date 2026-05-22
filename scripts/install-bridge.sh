#!/usr/bin/env bash
# ARCHGPU Bridge installer:
# - Checks required tools
# - Optionally installs missing packages on apt-based systems
# - Installs Python deps into .pkg
# - Optionally builds local/llama.cpp:server-intel image
# - Optionally installs/enables user systemd service

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO_INSTALL_PACKAGES=0
BUILD_INTEL_IMAGE=1
INSTALL_USER_SERVICE=0
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llm/llama.cpp}"

usage() {
  cat <<'EOF'
Usage: scripts/install-bridge.sh [options]

Options:
  --auto-install-packages   Install missing apt packages via sudo apt (Ubuntu/Debian)
  --skip-image-build        Do not build local/llama.cpp:server-intel
  --build-image             Force image build (default)
  --install-user-service    Install + enable scripts/archgpu-bridge.user.service
  --llama-cpp-dir <path>    Path to llama.cpp checkout (default: $HOME/llm/llama.cpp)
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-install-packages) AUTO_INSTALL_PACKAGES=1 ;;
    --skip-image-build) BUILD_INTEL_IMAGE=0 ;;
    --build-image) BUILD_INTEL_IMAGE=1 ;;
    --install-user-service) INSTALL_USER_SERVICE=1 ;;
    --llama-cpp-dir)
      shift
      LLAMA_CPP_DIR="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[fail] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

maybe_install_apt() {
  local pkg="$1"
  if [[ "$AUTO_INSTALL_PACKAGES" -ne 1 ]]; then
    return 1
  fi
  if ! need_cmd apt-get; then
    return 1
  fi
  sudo apt-get update -y
  sudo apt-get install -y "$pkg"
}

echo "[info] project root: $ROOT"
cd "$ROOT"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[warn] This installer is designed for Linux; continuing best-effort."
fi

# Core CLI dependencies
for cmd in python3 pip3 docker make curl jq; do
  if ! need_cmd "$cmd"; then
    echo "[warn] missing command: $cmd"
    case "$cmd" in
      python3) maybe_install_apt python3 || true ;;
      pip3) maybe_install_apt python3-pip || true ;;
      docker) maybe_install_apt docker.io || true ;;
      make) maybe_install_apt make || true ;;
      curl) maybe_install_apt curl || true ;;
      jq) maybe_install_apt jq || true ;;
    esac
  fi
done

missing=()
for cmd in python3 pip3 docker make curl jq; do
  if ! need_cmd "$cmd"; then
    missing+=("$cmd")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "[fail] missing required commands: ${missing[*]}"
  echo "       Re-run with --auto-install-packages on Ubuntu/Debian, or install them manually."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[fail] Docker daemon is not reachable."
  echo "       Start Docker and ensure this user can run docker commands."
  exit 1
fi

mkdir -p .pkg data

echo "[info] installing Python dependencies into .pkg"
python3 -m pip install --upgrade pip setuptools wheel >/dev/null
python3 -m pip install --upgrade --target .pkg -e .

if [[ "$BUILD_INTEL_IMAGE" -eq 1 ]]; then
  if docker image inspect "local/llama.cpp:server-intel" >/dev/null 2>&1; then
    echo "[ok] Docker image local/llama.cpp:server-intel already exists"
  else
    echo "[info] building local/llama.cpp:server-intel from $LLAMA_CPP_DIR"
    if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
      echo "[fail] llama.cpp source not found at: $LLAMA_CPP_DIR"
      echo "       Clone llama.cpp and pass --llama-cpp-dir <path>, or rerun with --skip-image-build."
      exit 1
    fi
    docker build -t local/llama.cpp:server-intel --target server -f "$LLAMA_CPP_DIR/.devops/intel.Dockerfile" "$LLAMA_CPP_DIR"
  fi
fi

if [[ "$INSTALL_USER_SERVICE" -eq 1 ]]; then
  mkdir -p "$HOME/.config/systemd/user"
  cp scripts/archgpu-bridge.user.service "$HOME/.config/systemd/user/archgpu-bridge.service"
  systemctl --user daemon-reload
  systemctl --user enable --now archgpu-bridge.service
  echo "[ok] user service enabled: archgpu-bridge.service"
fi

echo "[ok] installation checks complete"
echo "[info] next steps:"
echo "       1) make stack"
echo "       2) open http://127.0.0.1:3000"
