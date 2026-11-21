#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# GPU gate (Phase 11, docs/release.md "GPU host"): meshes each input on the
# CPU and on the HIP back end and compares them.
#
#   gate.sh OUT_DIR LIST [SURFACE:SPACING ...]
#
# LIST holds one input file name per line, relative to $DATA. The default
# cases are ses:1 ses:0.5 ses:0.25. Each input and case writes one CSV row
# to OUT_DIR/gate-*.csv:
#
#   file,surface,spacing,rc_cpu,rc_exact,rc_fast,exact_same,faces_same,
#   fast_repeat_same,area_cpu,area_fast,volume_cpu,volume_fast,cpu_edges
#
#   exact_same        CPU and `--backend hip` (exact) wrote the same OBJ and
#                     the same statistics (timings, back end and device aside)
#   faces_same        `--refine fast` wrote the CPU's face lines
#   fast_repeat_same  two fast runs wrote the same OBJ
#   cpu_edges         fast mode's edges refined on the CPU
#
# Then `summary.py` prints the totals and exits non-zero on any difference.
#
# Environment: BIN (default target/release/sesmesh, built with
# `--features hip`), DATA (default tests/data/cache), JOBS (parallel shards,
# default 4; each run uses every core, so more shards mostly overlap the
# serial parts).
set -euo pipefail

if [ $# -lt 2 ]; then
  awk 'NR >= 6 && /^#/ { sub(/^# ?/, ""); print; next } NR >= 6 { exit }' "$0"
  exit 2
fi
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=$1
list=$2
shift 2
cases=("$@")
[ ${#cases[@]} -gt 0 ] || cases=(ses:1 ses:0.5 ses:0.25)
BIN=${BIN:-$root/target/release/sesmesh}
DATA=${DATA:-$root/tests/data/cache}
JOBS=${JOBS:-4}

"$BIN" surface --help | grep -q -- '--backend <' || { echo "gate.sh: $BIN has no --backend" >&2; exit 2; }
mkdir -p "$out"
rm -f "$out"/gate-*.csv "$out"/shard-*

# One shard: every case of every input in $1, rows to $2.
shard() {
  local inputs=$1 csv=$2 w f c surf h
  w=$(mktemp -d "$out/work.XXXXXX")
  while read -r f; do
    [ -n "$f" ] || continue
    for c in "${cases[@]}"; do
      surf=${c%%:*}
      h=${c#*:}
      local args=("$DATA/$f" --surface "$surf" -s "$h" --quiet)
      local rc rce rcf
      "$BIN" "${args[@]}" --backend cpu -o "$w/cpu.obj" --stats "$w/cpu.json" 2>/dev/null && rc=0 || rc=$?
      "$BIN" "${args[@]}" --backend hip -o "$w/exact.obj" --stats "$w/exact.json" 2>/dev/null && rce=0 || rce=$?
      "$BIN" "${args[@]}" --backend hip --refine fast -o "$w/f1.obj" --stats "$w/f1.json" 2>/dev/null && rcf=0 || rcf=$?
      "$BIN" "${args[@]}" --backend hip --refine fast -o "$w/f2.obj" 2>/dev/null || true
      local row
      row=$(python3 -I "$here/compare.py" "$w")
      echo "$f,$surf,$h,$rc,$rce,$rcf,$row" >> "$csv"
      rm -f "$w"/*
    done
  done < "$inputs"
  rm -rf "$w"
}

split -n "l/$JOBS" -d "$list" "$out/shard-"
pids=()
for s in "$out"/shard-*; do
  shard "$s" "$out/gate-$(basename "$s").csv" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
rm -f "$out"/shard-*
python3 -I "$here/summary.py" "$out"/gate-*.csv
