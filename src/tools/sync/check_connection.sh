#!/usr/bin/env bash
set -euo pipefail

HEALTH_URL="${1:-}"
if [ -z "$HEALTH_URL" ]; then
  echo "Usage: bash src/tools/sync/check_connection.sh https://.../health"
  exit 2
fi

echo "[check] probing: $HEALTH_URL"
if curl -fsS --max-time 8 "$HEALTH_URL" | grep -q "^ok$"; then
  echo "[check] connection OK"
else
  echo "[check] connection FAILED"
  exit 1
fi
