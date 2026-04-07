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

# Only track the specific files we actually modified
TARGETS=(
    "src/flag_gems/runtime/backend/_ascend/fused/rotary_embedding.py"
    "src/flag_gems/modules/rotary_embedding.py"
)
PREFIX="FlagGems"

: > "$PATCH_FILE"
for RELPATH in "${TARGETS[@]}"; do
    FULL="$FLAGGEMS/$RELPATH"
    GIT_PATH="$PREFIX/$RELPATH"
    if [ ! -f "$FULL" ]; then
        echo "# WARNING: $FULL not found, skipping" >> "$PATCH_FILE"
        continue
    fi
    ORIG=$(git show "HEAD:$GIT_PATH" 2>/dev/null || true)
    if [ -z "$ORIG" ]; then
        # File not tracked in monorepo git — diff against itself is empty
        echo "# $GIT_PATH: not in git HEAD, skipping diff" >> "$PATCH_FILE"
        continue
    fi
    diff -u \
        <(echo "$ORIG") \
        "$FULL" \
        --label "a/$GIT_PATH" \
        --label "b/$GIT_PATH" \
        >> "$PATCH_FILE" || true
done

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
