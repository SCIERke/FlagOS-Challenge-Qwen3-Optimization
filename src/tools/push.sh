#!/usr/bin/env bash
# push.sh — local side: commit tracked changes + generate FlagGems patch, then push
# Usage: bash src/tools/push.sh "your commit message"
set -euo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKSPACE"

MSG="${1:-auto: sync workspace}"

# 1) Generate patch for FlagGems changes (tracked by git in FlagGems tree if any)
FLAGGEMS="$WORKSPACE/FlagGems"
PATCH_DIR="$WORKSPACE/knowledge/rotary"
PATCH_FILE="$PATCH_DIR/flaggems.patch"

mkdir -p "$PATCH_DIR"
# The patch for FlagGems (git-ignored) is maintained manually in knowledge/rotary/flaggems.patch.
# It was created once from known good diffs. If you need to update it, edit it directly.
if [ -s "$PATCH_FILE" ]; then
    echo "[push] FlagGems patch already exists: $PATCH_FILE ($(wc -l < "$PATCH_FILE") lines)"
else
    echo "[push] WARNING: $PATCH_FILE is empty or missing — FlagGems changes will not be synced."
fi

echo "[push] Patch written: $PATCH_FILE ($(wc -l < "$PATCH_FILE") lines)"

# 2) Stage everything tracked in workspace (src/, knowledge/, vllm-plugin-FL/)
git add src/ knowledge/ vllm-plugin-FL/ setup.sh .gitignore 2>/dev/null || true
git add "$PATCH_FILE" 2>/dev/null || true

# 3) Commit if there are staged changes
if git diff --cached --quiet; then
    echo "[push] Nothing to commit, pushing existing HEAD."
else
    git commit -m "$MSG"
fi

# 4) Push
git push origin main
echo "[push] Done — pull on instance with: bash src/tools/pull.sh"
