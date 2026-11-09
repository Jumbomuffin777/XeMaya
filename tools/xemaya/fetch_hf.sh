#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# fetch_hf.sh REPO SUBDIR DEST [RATE]
# Resumable, rate-limited download of every .gguf under SUBDIR of a Hugging Face model repo into DEST, each file
# checked against the sha256 Hugging Face records for it (LFS oid).  A verified file gets FILE.sha256ok and is skipped
# on the next run.  Low CPU and IO priority; one file at a time.
set -u
repo=${1:?repo}; sub=${2:?subdir}; dest=${3:?dest}; rate=${4:-50M}
mkdir -p "$dest"; cd "$dest" || exit 1
list=$(curl -sf "https://huggingface.co/api/models/$repo/tree/main/$sub" | python3 -c '
import json,sys
for x in json.load(sys.stdin):
    if x["path"].endswith(".gguf") and x.get("lfs"): print(x["path"], x["lfs"]["oid"], x["size"])') || { echo "listing failed"; exit 1; }
echo "$list" | while read -r path sha size; do
  f=$(basename "$path")
  [ -f "$f.sha256ok" ] && { echo "ok (verified before): $f"; continue; }
  echo "$(date -Is) fetching $f ($((size / 2**20)) MiB)"
  for try in 1 2 3 4 5; do
    nice -n 19 ionice -c3 curl -sfL -C - --limit-rate "$rate" --retry 5 --retry-delay 10 -o "$f" \
      "https://huggingface.co/$repo/resolve/main/$path" </dev/null && break
    echo "  attempt $try failed, retrying"; sleep 30
  done
  got=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$got" = "$size" ] || { echo "SIZE MISMATCH $f: $got != $size"; exit 1; }
  h=$(nice -n 19 ionice -c3 sha256sum "$f" | cut -d' ' -f1)
  [ "$h" = "$sha" ] || { echo "SHA256 MISMATCH $f"; exit 1; }
  echo "$sha  $f" > "$f.sha256ok"; echo "$(date -Is) verified $f"
done
echo "$(date -Is) ALL DONE"
