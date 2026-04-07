#!/usr/bin/env bash
# pull.sh — instance side: git pull + apply FlagGems patch
# Usage: bash src/tools/pull.sh
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE"

REPO="https://github.com/SCIERke/FlagOS-Challenge-Qwen3-Optimization.git"
PATCH_FILE="$WORKSPACE/knowledge/rotary/flaggems.patch"
FLAGGEMS="$WORKSPACE/FlagGems"

# 1) Pull latest workspace (src/, knowledge/, vllm-plugin-FL/)
if [ -d ".git" ]; then
    echo "[pull] Pulling workspace..."
    git pull origin main
else
    echo "[pull] Not a git repo yet — cloning..."
    git clone "$REPO" .
fi

# 2) Apply FlagGems patch if it exists and FlagGems dir exists
if [ ! -d "$FLAGGEMS" ]; then
    echo "[pull] WARNING: FlagGems directory not found at $FLAGGEMS — skipping patch."
    exit 0
fi

if [ ! -s "$PATCH_FILE" ]; then
    echo "[pull] No FlagGems patch to apply (empty or missing: $PATCH_FILE)."
    exit 0
fi

echo "[pull] Applying FlagGems patch..."
cd "$FLAGGEMS"
if patch -p0 --dry-run < "$PATCH_FILE" &>/dev/null; then
    patch -p0 < "$PATCH_FILE"
    echo "[pull] FlagGems patch applied OK."
else
    # Try with -p1 (path prefix strip)
    if patch -p1 --dry-run < "$PATCH_FILE" &>/dev/null; then
        patch -p1 < "$PATCH_FILE"
        echo "[pull] FlagGems patch applied OK (p1)."
    else
        echo "[pull] WARNING: patch did not apply cleanly — may already be applied or conflict."
        echo "       Patch file: $PATCH_FILE"
        echo "       Apply manually: cd $FLAGGEMS && patch -p1 < $PATCH_FILE"
    fi
fi
