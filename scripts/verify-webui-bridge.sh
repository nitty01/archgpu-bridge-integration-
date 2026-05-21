#!/usr/bin/env bash
# Headless verification: bridge + Open WebUI container can list models (same checks as make status).
# Optional: POST /v1/chat/completions (may start Docker backends — slow).
#
# Usage:
#   scripts/verify-webui-bridge.sh
#   VERIFY_CHAT=1 MODEL=mistral.gguf scripts/verify-webui-bridge.sh
#
# Environment: same as scripts/status.sh (BRIDGE_URL, OPENWEBUI_NAME, OPENWEBUI_URL).

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8000}"
VERIFY_CHAT="${VERIFY_CHAT:-0}"
MODEL="${MODEL:-mistral.gguf}"

echo "== verify-webui-bridge: status (bridge + in-container API) =="
bash "$ROOT/scripts/status.sh" || exit 1

if [ "$VERIFY_CHAT" = "1" ]; then
  echo ""
  echo "== verify-webui-bridge: POST $BRIDGE_URL/v1/chat/completions (model=$MODEL) =="
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found" >&2
    exit 1
  fi
  payload=$(printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"stream":false,"max_tokens":4}' "$MODEL")
  if curl -fsS --max-time 120 -X POST "$BRIDGE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$payload" | head -c 2000; then
    echo ""
    echo "OK: chat completion returned (may have started a backend container)"
  else
    echo "FAIL: chat completion request failed" >&2
    exit 1
  fi
else
  echo "Tip: run with VERIFY_CHAT=1 for a one-shot chat test (may load a model; slower)."
fi

echo "OK: verify-webui-bridge finished"
