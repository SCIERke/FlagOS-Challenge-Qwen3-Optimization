#!/usr/bin/env bash
# Apply a sync bundle into workspace safely.
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
BUNDLE_PATH="${1:-}"
extract_items=()
NON_INTERACTIVE=0
if [ "${FORCE_NON_INTERACTIVE:-0}" = "1" ] || [ ! -t 0 ]; then
  NON_INTERACTIVE=1
fi

_pick_bundle_interactive() {
  local cands=()
  local i=1
  while IFS= read -r line; do
    cands+=("$line")
  done < <(
    {
      ls -1t /tmp/flagos_sync/*.tgz 2>/dev/null || true
      ls -1t /tmp/flagos_sync_bundle*.tgz 2>/dev/null || true
    } | awk '!seen[$0]++'
  )

  if [ "${#cands[@]}" -eq 0 ]; then
    if [ "$NON_INTERACTIVE" -eq 1 ]; then
      echo "[apply] bundle path is required in non-interactive mode"
      exit 2
    fi
    read -r -p "Bundle path: " BUNDLE_PATH
    return
  fi

  echo "[apply] Select bundle:"
  for p in "${cands[@]}"; do
    echo "  [$i] $p"
    i=$((i + 1))
  done
  if [ "$NON_INTERACTIVE" -eq 1 ]; then
    choice="1"
  else
    read -r -p "Index (Enter for [1]): " choice
  fi
  choice="${choice:-1}"
  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#cands[@]}" ]; then
    BUNDLE_PATH="${cands[$((choice - 1))]}"
  else
    echo "[apply] invalid choice"
    exit 2
  fi
}

_pick_extract_items() {
  local tops=()
  local i=1
  while IFS= read -r line; do
    tops+=("$line")
  done < <(tar -tzf "$BUNDLE_PATH" | awk -F/ 'NF>0{print $1}' | sort -u)
  if [ "${#tops[@]}" -eq 0 ]; then
    echo "[apply] empty archive"
    exit 1
  fi
  echo "[apply] Bundle top-level entries:"
  for p in "${tops[@]}"; do
    echo "  [$i] $p"
    i=$((i + 1))
  done
  if [ "$NON_INTERACTIVE" -eq 1 ]; then
    choice=""
  else
    read -r -p "Extract indexes (comma-separated, Enter = all): " choice
  fi
  if [ -z "$choice" ]; then
    extract_items=("${tops[@]}")
    return
  fi
  IFS=',' read -r -a idxs <<< "$choice"
  extract_items=()
  for idx in "${idxs[@]}"; do
    idx="$(echo "$idx" | xargs)"
    if [[ "$idx" =~ ^[0-9]+$ ]] && [ "$idx" -ge 1 ] && [ "$idx" -le "${#tops[@]}" ]; then
      extract_items+=("${tops[$((idx - 1))]}")
    fi
  done
  if [ "${#extract_items[@]}" -eq 0 ]; then
    echo "[apply] no valid selection"
    exit 2
  fi
}

if [ -z "$BUNDLE_PATH" ]; then
  _pick_bundle_interactive
fi

if [ ! -f "$BUNDLE_PATH" ]; then
  echo "[apply] bundle not found: $BUNDLE_PATH"
  exit 2
fi

cd "$WORKSPACE"
echo "[apply] workspace: $WORKSPACE"
echo "[apply] bundle: $BUNDLE_PATH"

_pick_extract_items

echo "[apply] extracting: ${extract_items[*]}"
tar -xzf "$BUNDLE_PATH" -C "$WORKSPACE" "${extract_items[@]}"
echo "[apply] done"
