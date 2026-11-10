#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Runs a GPU test and stops it at the first sign that it is hurting the host (the development host, 2026-11-09: sustained expert
# streaming to the B60 behind the chipset starved the RTL8126 NIC beside it - two NETDEV WATCHDOG timeouts - and the host
# lost its network until it was rebooted).
#
#   safety_guard.sh [--log FILE] -- CMD ARGS...
#
# CMD runs in its own process group. The guard stops it (SIGTERM to the group, SIGKILL 30 s later if it is still there)
# and exits 99 when any of these appears:
#   - a kernel message: NETDEV WATCHDOG, a GPU engine reset or fault (xe), an OOM kill, a hung task, an NVMe timeout or
#     reset, a PCIe AER error
#   - the default gateway misses 3 pings in a row (5 s apart)
#   - MemAvailable under 12 GiB
# Otherwise it exits with CMD's status. It never touches anything but CMD's process group.
set -u
LOG=/dev/stderr
if [ "${1:-}" = "--log" ]; then LOG=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
[ $# -ge 1 ] || { echo "usage: safety_guard.sh [--log FILE] -- CMD ARGS..." >&2; exit 2; }
say() { if [ "$LOG" = /dev/stderr ]; then echo "$(date -u +%T) safety_guard: $*" >&2; else echo "$(date -u +%T) safety_guard: $*" | tee -a "$LOG" >&2; fi; }
PAT='NETDEV WATCHDOG|transmit queue .* timed out|xe .*(Engine reset|Fault response|GT reset|wedged)|Out of memory|oom-kill|Killed process|hung_task|blocked for more than|nvme.*(timeout|reset)|AER:.*(error|Error)'
REASON=$(mktemp)
setsid "$@" &
CHILD=$!
PGID=$CHILD
say "running pid $CHILD: $*"
# the kernel's messages from now on
( journalctl -k -f -n 0 --no-pager 2>/dev/null | grep --line-buffered -E "$PAT" | while read -r l; do
    echo "kernel: $l" > "$REASON"; kill -USR1 $$ 2>/dev/null; break; done ) &
WATCH=$!
GW=$(ip route show default 2>/dev/null | awk '/default/ {print $3; exit}')
trip() {
  say "STOPPING: $(cat "$REASON")"
  kill -TERM -- -"$PGID" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
  if kill -0 "$CHILD" 2>/dev/null; then say "still running after 30 s: SIGKILL"; kill -KILL -- -"$PGID" 2>/dev/null; fi
  wait "$CHILD" 2>/dev/null
  pkill -P "$WATCH" 2>/dev/null; kill "$WATCH" 2>/dev/null
  rm -f "$REASON"
  exit 99
}
trap 'trip' USR1
misses=0
while kill -0 "$CHILD" 2>/dev/null; do
  if [ -n "$GW" ]; then
    if ping -c 1 -W 2 "$GW" > /dev/null 2>&1; then misses=0; else misses=$((misses + 1)); fi
    if [ $misses -ge 3 ]; then echo "the gateway $GW missed 3 pings" > "$REASON"; trip; fi
  fi
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ "$avail" -lt "${SAFETY_MIN_FREE_GIB:-12}" ]; then echo "MemAvailable ${avail} GiB" > "$REASON"; trip; fi
  sleep 5 & wait $!
done
wait "$CHILD"; rc=$?
pkill -P "$WATCH" 2>/dev/null; kill "$WATCH" 2>/dev/null
rm -f "$REASON"
say "finished, status $rc"
exit $rc
