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
if [ -d "$FLAGGEMS/.git" ]; then
    git -C "$FLAGGEMS" diff > "$PATCH_FILE"
    if [ ! -s "$PATCH_FILE" ]; then
        git -C "$FLAGGEMS" diff HEAD > "$PATCH_FILE" 2>/dev/null || true
    fi
else
    # No git in FlagGems — diff against a reference copy if exists, else snapshot changed files
    echo "# FlagGems has no .git; patch generated from working copy diffs" > "$PATCH_FILE"
    diff -u \
        <(git show HEAD:"FlagGems/src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py" 2>/dev/null || echo "") \
        "$FLAGGEMS/src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py" \
        >> "$PATCH_FILE" || true
    diff -u \
        <(git show HEAD:"FlagGems/src/flag_gems/modules/rotary_embedding.py" 2>/dev/null || echo "") \
        "$FLAGGEMS/src/flag_gems/modules/rotary_embedding.py" \
        >> "$PATCH_FILE" || true
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
