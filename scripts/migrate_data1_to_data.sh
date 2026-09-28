#!/usr/bin/env bash
# Safe CM1 data migration: copy, SHA-256 verify, delete source, add old-path symlink.
set -Eeuo pipefail

readonly SRC="/data1/home/zhangyx/data"
readonly DST="/data/zhangyx/DATA"
readonly STATE_DIR="/data/zhangyx/.tc_dynamic_data_migration"
readonly RUN_LOG="${STATE_DIR}/migration.log"
readonly SOURCE_SNAPSHOT="${STATE_DIR}/source_snapshot.tsv"
readonly SOURCE_SNAPSHOT_FINAL="${STATE_DIR}/source_snapshot_final.tsv"
readonly SOURCE_SHA256="${STATE_DIR}/source.sha256"
readonly TARGET_SHA256="${STATE_DIR}/target.sha256"
readonly VERIFY_REPORT="${STATE_DIR}/verification_report.txt"
readonly COMPLETE_MARKER="${STATE_DIR}/MIGRATION_COMPLETE"

mkdir -p "${STATE_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

log() { printf '[%s] %s\n' "$(date '+%F %T %z')" "$*"; }
fail() { log "ERROR: $*"; exit 1; }

snapshot_tree() {
  local tree="$1" output="$2"
  (cd "${tree}" && find . -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort) > "${output}"
}

hash_tree() {
  local tree="$1" output="$2"
  (cd "${tree}" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum) > "${output}"
}

[[ -L "${SRC}" ]] && {
  if [[ "$(readlink -f "${SRC}")" == "${DST}" && -f "${COMPLETE_MARKER}" ]]; then
    log "Migration already completed; ${SRC} points to ${DST}."
    exit 0
  fi
  fail "Unexpected source symlink: ${SRC} -> $(readlink "${SRC}")"
}
[[ -d "${SRC}" ]] || fail "Missing source directory: ${SRC}"
[[ "$(findmnt -n -o TARGET -T "${SRC}")" == "/data1" ]] || fail "Source is not on /data1"
[[ "$(findmnt -n -o TARGET -T "/data/zhangyx")" == "/data" ]] || fail "Destination parent is not on /data"
mkdir -p "${DST}"
[[ "$(readlink -f "${SRC}")" != "$(readlink -f "${DST}")" ]] || fail "Source and destination are identical"

special_count="$(find "${SRC}" -mindepth 1 ! -type f ! -type d | wc -l)"
[[ "${special_count}" -eq 0 ]] || fail "Source contains ${special_count} non-regular entries"

snapshot_tree "${SRC}" "${SOURCE_SNAPSHOT}"
source_count="$(wc -l < "${SOURCE_SNAPSHOT}")"
source_bytes="$(awk -F '\t' '{sum += $2} END {printf "%.0f", sum}' "${SOURCE_SNAPSHOT}")"
available_bytes="$(df -B1 --output=avail "${DST}" | tail -1 | tr -d ' ')"
required_bytes=$((source_bytes + source_bytes / 10))
(( available_bytes > required_bytes )) || fail "Destination lacks source size plus 10% safety margin"
log "Preflight passed: ${source_count} files, ${source_bytes} bytes; ${available_bytes} bytes free."

log "Starting resumable copy; source files remain untouched."
rsync -aH --numeric-ids --partial --partial-dir=.rsync-partial \
  --info=progress2 --human-readable "${SRC}/" "${DST}/"
rm -rf -- "${DST}/.rsync-partial"
log "Copy pass completed."

snapshot_tree "${SRC}" "${SOURCE_SNAPSHOT_FINAL}"
cmp -s "${SOURCE_SNAPSHOT}" "${SOURCE_SNAPSHOT_FINAL}" || fail "Source changed during copy; nothing deleted"
target_count="$(find "${DST}" -type f | wc -l)"
target_bytes="$(find "${DST}" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
[[ "${target_count}" -eq "${source_count}" ]] || fail "File-count mismatch: ${source_count} vs ${target_count}"
[[ "${target_bytes}" -eq "${source_bytes}" ]] || fail "Byte-count mismatch: ${source_bytes} vs ${target_bytes}"

log "Computing SHA-256 for every source file."
hash_tree "${SRC}" "${SOURCE_SHA256}"
log "Computing SHA-256 for every destination file."
hash_tree "${DST}" "${TARGET_SHA256}"
if ! cmp -s "${SOURCE_SHA256}" "${TARGET_SHA256}"; then
  diff -u "${SOURCE_SHA256}" "${TARGET_SHA256}" > "${VERIFY_REPORT}" || true
  fail "SHA-256 verification failed; nothing deleted"
fi

snapshot_tree "${SRC}" "${SOURCE_SNAPSHOT_FINAL}"
cmp -s "${SOURCE_SNAPSHOT}" "${SOURCE_SNAPSHOT_FINAL}" || fail "Source changed during verification; nothing deleted"
{
  echo "status=verified"
  echo "verified_at=$(date --iso-8601=seconds)"
  echo "source=${SRC}"
  echo "destination=${DST}"
  echo "file_count=${source_count}"
  echo "total_bytes=${source_bytes}"
  echo "checksum=SHA-256"
  echo "source_manifest=${SOURCE_SHA256}"
  echo "target_manifest=${TARGET_SHA256}"
} > "${VERIFY_REPORT}"
log "All ${source_count} files passed SHA-256 verification."

log "Deleting verified source files."
find "${SRC}" -type f -delete
find "${SRC}" -mindepth 1 -depth -type d -empty -delete
[[ -z "$(find "${SRC}" -mindepth 1 -print -quit)" ]] || fail "Unexpected source entries remain"
rmdir -- "${SRC}"
ln -s -- "${DST}" "${SRC}"
ln -sfn -- "${DST}/cm1out_Morrison.nc" \
  "/data1/home/zhangyx/project/TC_dynamic/dataset/cm1out.nc"
{
  echo "completed_at=$(date --iso-8601=seconds)"
  echo "old_path=${SRC}"
  echo "new_path=${DST}"
  echo "compatibility_symlink=${SRC} -> ${DST}"
  echo "verification_report=${VERIFY_REPORT}"
} > "${COMPLETE_MARKER}"
log "Migration complete; old path is now a compatibility symlink."
df -h "${SRC}" "${DST}"
