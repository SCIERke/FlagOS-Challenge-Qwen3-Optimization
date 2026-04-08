#!/usr/bin/env bash
# Local-side sender: pack and upload selected files/directories.
# Robust mode:
# - Layer 1: auto chunk upload to avoid 413
# - Layer 2: retry + resume via per-session sent-files cache
set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
URL="${1:-}"

default_paths=("FlagGems" "vllm-plugin-FL" "src")
selected_paths=()
STATE_FILE="${WORKSPACE}/.sync_send_state.tsv"
changed_only="yes"
tar_flags=()
MAX_CHUNK_KB="${MAX_CHUNK_KB:-350}"   # safe default for strict proxies
RETRY_COUNT="${RETRY_COUNT:-3}"

_hash_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  else
    shasum -a 256 "$f" | awk '{print $1}'
  fi
}

_hash_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print $1}'
  else
    printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
  fi
}

_is_excluded_path() {
  local p="$1"
  case "$p" in
    FlagGems/docs/*|FlagGems/docs)
      return 0
      ;;
    */.git/*|.git/*|*/.git|.git)
      return 0
      ;;
    */.DS_Store|.DS_Store|*/._*|._*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_prompt_url() {
  local input
  read -r -p "Upload URL (must end with /upload): " input
  echo "$input"
}

_collect_paths_interactive() {
  local i=1
  local all=()
  while IFS= read -r line; do
    all+=("$line")
  done < <(cd "$WORKSPACE" && ls -1A)
  echo "[send] Select files/folders to include (comma-separated indexes)."
  echo "       Press Enter for defaults: ${default_paths[*]}"
  for p in "${all[@]}"; do
    echo "  [$i] $p"
    i=$((i + 1))
  done
  echo
  echo "[hint] You can also type custom relative paths (comma-separated), e.g.:"
  echo "       FlagGems/src,src,vllm-plugin-FL"
  local choice
  read -r -p "Indexes OR paths: " choice
  if [ -z "$choice" ]; then
    selected_paths=("${default_paths[@]}")
    return
  fi

  IFS=',' read -r -a idxs <<< "$choice"
  selected_paths=()
  for idx in "${idxs[@]}"; do
    idx="$(echo "$idx" | xargs)"
    if [[ "$idx" =~ ^[0-9]+$ ]]; then
      if [ "$idx" -ge 1 ] && [ "$idx" -le "${#all[@]}" ]; then
        selected_paths+=("${all[$((idx - 1))]}")
      fi
    else
      selected_paths+=("$idx")
    fi
  done
  if [ "${#selected_paths[@]}" -eq 0 ]; then
    echo "[send] no valid selection"
    exit 2
  fi
}

_collect_selected_files() {
  local list_file="$1"
  : > "$list_file"
  while IFS= read -r file; do
    if _is_excluded_path "$file"; then
      continue
    fi
    printf '%s\n' "$file" >> "$list_file"
  done < <(
    cd "$WORKSPACE"
    for p in "${selected_paths[@]}"; do
      if [ -f "$p" ]; then
        printf '%s\n' "$p"
      elif [ -d "$p" ]; then
        find "$p" -type f \
          -not -path 'FlagGems/docs/*' \
          -not -path '*/.git/*' \
          -not -name '.DS_Store' \
          -not -name '._*' | sort
      fi
    done
  )
}

_collect_changed_files() {
  local selected_file_list="$1"
  local changed_list="$2"
  local prev_map="$3"
  local new_map="$4"
  : > "$changed_list"
  : > "$new_map"

  local changed_count=0
  local file h old
  while IFS= read -r file; do
    h="$(_hash_file "$WORKSPACE/$file")"
    printf '%s\t%s\n' "$h" "$file" >> "$new_map"
    old="$(awk -F'\t' -v p="$file" '$2==p {print $1; exit}' "$prev_map" 2>/dev/null || true)"
    if [ "$h" != "$old" ]; then
      printf '%s\n' "$file" >> "$changed_list"
      changed_count=$((changed_count + 1))
    fi
  done < "$selected_file_list"
  echo "$changed_count"
}

_bundle_size_kb() {
  local f="$1"
  local bytes
  bytes="$(wc -c < "$f" | tr -d ' ')"
  echo $(((bytes + 1023) / 1024))
}

_build_chunk_tar() {
  # args: input_list start_index out_tar out_chunk_list
  local input_list="$1"
  local start_idx="$2"
  local out_tar="$3"
  local out_list="$4"
  local test_list="${out_list}.test"
  local test_tar="${out_tar}.test"
  local max_kb="$MAX_CHUNK_KB"

  : > "$out_list"
  local idx=0
  local added=0
  local line candidate_size

  while IFS= read -r line; do
    idx=$((idx + 1))
    if [ "$idx" -lt "$start_idx" ]; then
      continue
    fi

    cp "$out_list" "$test_list"
    printf '%s\n' "$line" >> "$test_list"
    if [ "${#tar_flags[@]}" -gt 0 ]; then
      COPYFILE_DISABLE=1 tar "${tar_flags[@]}" -czf "$test_tar" -T "$test_list"
    else
      COPYFILE_DISABLE=1 tar -czf "$test_tar" -T "$test_list"
    fi
    candidate_size="$(_bundle_size_kb "$test_tar")"

    if [ "$candidate_size" -le "$max_kb" ] || [ "$added" -eq 0 ]; then
      mv "$test_list" "$out_list"
      mv "$test_tar" "$out_tar"
      added=$((added + 1))
    else
      rm -f "$test_list" "$test_tar"
      break
    fi
  done < "$input_list"

  echo "$added"
}

_upload_with_retry() {
  local tar_path="$1"
  local attempt=1
  local http_code resp_file
  resp_file="/tmp/sync_upload_resp_$$.txt"

  while [ "$attempt" -le "$RETRY_COUNT" ]; do
    http_code="$(curl -sS -o "$resp_file" -w "%{http_code}" -X POST -F "file=@${tar_path}" "$URL" || true)"
    if [ "$http_code" = "200" ] || [ "$http_code" = "202" ]; then
      cat "$resp_file"
      rm -f "$resp_file"
      return 0
    fi
    echo "[send] upload failed (HTTP $http_code), attempt $attempt/$RETRY_COUNT"
    if [ -f "$resp_file" ]; then
      cat "$resp_file" || true
    fi
    attempt=$((attempt + 1))
    sleep 1
  done
  rm -f "$resp_file"
  return 1
}

if [ -z "$URL" ]; then
  URL="$(_prompt_url)"
fi

if [[ "$URL" != */upload ]]; then
  echo "[send] URL should end with /upload"
  exit 2
fi

HEALTH_URL="${URL%/upload}/health"
echo "[send] checking receiver: $HEALTH_URL"
if ! curl -fsS --max-time 8 "$HEALTH_URL" | grep -q "^ok$"; then
  echo "[send] receiver not ready (health check failed)"
  exit 1
fi

cd "$WORKSPACE"
ALL_SELECTED_LIST="/tmp/flagos_sync_selected_$$.txt"
LIST_FILE="/tmp/flagos_sync_files_$$.txt"
NEW_STATE="/tmp/flagos_sync_state_$$.tsv"

if tar --help 2>/dev/null | grep -q -- '--no-mac-metadata'; then
  tar_flags+=(--no-mac-metadata)
fi

_collect_paths_interactive
read -r -p "Send changed files only? [Y/n]: " mode_ans
if [ -n "$mode_ans" ] && [[ "$mode_ans" =~ ^[Nn]$ ]]; then
  changed_only="no"
fi
read -r -p "Max chunk size KB (default ${MAX_CHUNK_KB}): " kb_ans
if [[ "${kb_ans:-}" =~ ^[0-9]+$ ]] && [ "$kb_ans" -gt 0 ]; then
  MAX_CHUNK_KB="$kb_ans"
fi

for p in "${selected_paths[@]}"; do
  if [ ! -e "$WORKSPACE/$p" ]; then
    echo "[send] path not found in workspace: $p"
    exit 2
  fi
done

echo "[send] include: ${selected_paths[*]}"
_collect_selected_files "$ALL_SELECTED_LIST"

if [ "$changed_only" = "yes" ]; then
  touch "$STATE_FILE"
  if [ ! -s "$STATE_FILE" ]; then
    echo "[send] first changed-only run for this workspace/state file."
    echo "       baseline is empty, so many files will be treated as changed."
  fi
  changed_count="$(_collect_changed_files "$ALL_SELECTED_LIST" "$LIST_FILE" "$STATE_FILE" "$NEW_STATE")"
  if [ "$changed_count" -eq 0 ]; then
    echo "[send] no changed files detected; nothing to upload"
    rm -f "$ALL_SELECTED_LIST" "$LIST_FILE" "$NEW_STATE"
    exit 0
  fi
  echo "[send] changed files: $changed_count"
else
  cp "$ALL_SELECTED_LIST" "$LIST_FILE"
fi

session_key="$(_hash_text "${URL}|${changed_only}|${selected_paths[*]}|$(cat "$LIST_FILE" 2>/dev/null || true)")"
RESUME_FILE="${WORKSPACE}/.sync_upload_resume_${session_key}.txt"
touch "$RESUME_FILE"

PENDING_LIST="/tmp/flagos_sync_pending_$$.txt"
grep -Fvx -f "$RESUME_FILE" "$LIST_FILE" > "$PENDING_LIST" || true
pending_count="$(wc -l < "$PENDING_LIST" | tr -d ' ')"
if [ "$pending_count" -eq 0 ]; then
  echo "[send] nothing pending (resume cache already complete)."
  rm -f "$ALL_SELECTED_LIST" "$LIST_FILE" "$NEW_STATE" "$PENDING_LIST"
  exit 0
fi
echo "[send] pending files: $pending_count"
echo "[send] chunk size target: ${MAX_CHUNK_KB} KB"

start_idx=1
chunk_idx=0
total_lines="$pending_count"
while [ "$start_idx" -le "$total_lines" ]; do
  chunk_idx=$((chunk_idx + 1))
  CHUNK_LIST="/tmp/flagos_sync_chunk_${$}_${chunk_idx}.txt"
  CHUNK_TAR="/tmp/flagos_sync_chunk_${$}_${chunk_idx}.tgz"

  added="$(_build_chunk_tar "$PENDING_LIST" "$start_idx" "$CHUNK_TAR" "$CHUNK_LIST")"
  if [ "$added" -le 0 ]; then
    echo "[send] failed to build chunk at index $start_idx"
    exit 1
  fi

  size_kb="$(_bundle_size_kb "$CHUNK_TAR")"
  echo "[send] uploading chunk #$chunk_idx: files=$added size=${size_kb}KB"
  if ! _upload_with_retry "$CHUNK_TAR"; then
    echo "[send] chunk #$chunk_idx failed after retries."
    echo "[send] resume later with same selection; already sent files are cached."
    rm -f "$ALL_SELECTED_LIST" "$LIST_FILE" "$NEW_STATE" "$PENDING_LIST" "$CHUNK_LIST" "$CHUNK_TAR"
    exit 1
  fi

  cat "$CHUNK_LIST" >> "$RESUME_FILE"
  start_idx=$((start_idx + added))
  rm -f "$CHUNK_LIST" "$CHUNK_TAR"
done

# Update changed-only state only after all chunks uploaded successfully.
if [ "$changed_only" = "yes" ]; then
  mv "$NEW_STATE" "$STATE_FILE"
fi
rm -f "$RESUME_FILE"
echo
echo "[send] done (all chunks uploaded)"
rm -f "$ALL_SELECTED_LIST" "$LIST_FILE" "$NEW_STATE" "$PENDING_LIST"
