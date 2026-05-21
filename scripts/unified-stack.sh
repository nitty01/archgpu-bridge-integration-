#!/usr/bin/env bash
# Unified entry for bridge + Open WebUI: by default **restarts** both and verifies connection.
#
#   ./scripts/unified-stack.sh              # same as: stack.sh restart  (stop → start → status)
#   ./scripts/unified-stack.sh up           # gentle: only start if not running (stack.sh up)
#   ./scripts/unified-stack.sh restart      # explicit restart
#   ./scripts/unified-stack.sh status|down|help
#
# See scripts/stack.sh for all options and environment variables.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ "$#" -eq 0 ]; then
  exec "$SCRIPT_DIR/stack.sh" restart
fi
exec "$SCRIPT_DIR/stack.sh" "$@"
