#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# GPU gate (Phase 11, 12; docs/release.md "GPU host"): meshes each input on
# the CPU and on a GPU back end and compares them.
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
# Then `summary.sh` prints the totals and exits non-zero on any difference.
#
# Environment: BIN (default target/release/sesmesh, built with the back
# end's feature), BACKEND (`hip`, default, or `cuda`), DATA (default
# tests/data/cache), JOBS (parallel shards, default 4; each run uses every
# core, so more shards mostly overlap the serial parts), SIMD (passed as
# `--simd` to every run when set).
#
# CPU references from another machine (a slow host CPU, or a GPU host
# without the time): MODE=cpu runs only the CPU and writes one reference
# record per case to OUT_DIR/ref (JSON: the SHA-256 of the OBJ and of its
# face lines, and the statistics); MODE=gpu with REF=that directory runs
# only the GPU and compares with the records. Both machines need the same
# binary and inputs (and SIMD level: set SIMD to one both support).
#
# Needs bash, GNU coreutils (split, sha256sum), grep and jq. OBJ files are
# only streamed (they reach tens of GB at 0.25 Å).
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
BACKEND=${BACKEND:-hip}
DATA=${DATA:-$root/tests/data/cache}
JOBS=${JOBS:-4}
MODE=${MODE:-all}
SIMD=${SIMD:-}
case $MODE in
  all | cpu) ;;
  gpu) [ -n "${REF:-}" ] || { echo "gate.sh: MODE=gpu needs REF" >&2; exit 2; } ;;
  *) echo "gate.sh: MODE is all, cpu or gpu" >&2; exit 2 ;;
esac

command -v jq > /dev/null || { echo "gate.sh: needs jq" >&2; exit 2; }
"$BIN" surface --help | grep -q -- '--backend <' || { echo "gate.sh: $BIN has no --backend" >&2; exit 2; }
mkdir -p "$out"
rm -f "$out"/gate-*.csv "$out"/shard-*
[ "$MODE" != cpu ] || { rm -rf "$out/ref"; mkdir -p "$out/ref"; }

# SHA-256 of a file, and of its face lines.
digest() { sha256sum "$1" | cut -d' ' -f1; }
faces() { { LC_ALL=C grep -a '^f ' "$1" || [ $? -eq 1 ]; } | sha256sum | cut -d' ' -f1; }

# Statistics keys that differ between back ends.
STRIP='del(.timings_ms) | del(.engine.backend, .engine.device, .engine.gpu_device,
  .engine.refine_cpu_edges, .parameters.backend, .parameters.device,
  .parameters.gpu_device, .parameters.refine_cpu_edges)'

# The CPU run's record in work dir $1 (cpu.obj, cpu.json) as one JSON line;
# {} when the run wrote no files.
reference() {
  local w=$1
  if [ -f "$w/cpu.obj" ] && [ -f "$w/cpu.json" ]; then
    jq -cS --arg obj "$(digest "$w/cpu.obj")" --arg faces "$(faces "$w/cpu.obj")" \
      '{obj: $obj, faces: $faces, stats: .}' "$w/cpu.json" || echo '{}'
  else
    echo '{}'
  fi
}

# The CSV columns exact_same … cpu_edges for work dir $1: the CPU record
# cpu.ref against exact.obj/.json, f1.obj/.json and f2.obj ('-' where a
# file or the record is missing).
compare() {
  local w=$1 ref=$1/cpu.ref same=- fsame=- rep=- nums=-,-,-,-,-
  [ -s "$ref" ] || echo '{}' > "$ref"
  if [ -f "$w/exact.obj" ] && [ -f "$w/exact.json" ] && jq -e 'has("obj")' "$ref" > /dev/null; then
    same=$(jq -r --arg obj "$(digest "$w/exact.obj")" --slurpfile e "$w/exact.json" \
      "def strip: $STRIP;
       if .obj == \$obj and (.stats | strip) == (\$e[0] | strip) then \"yes\" else \"no\" end" "$ref")
  fi
  if [ -f "$w/f1.obj" ] && jq -e 'has("faces")' "$ref" > /dev/null; then
    if [ "$(jq -r .faces "$ref")" = "$(faces "$w/f1.obj")" ]; then fsame=yes; else fsame=no; fi
    if [ -f "$w/f2.obj" ]; then
      if [ "$(digest "$w/f1.obj")" = "$(digest "$w/f2.obj")" ]; then rep=yes; else rep=no; fi
    fi
  fi
  if [ -f "$w/f1.json" ] && jq -e '.stats.mesh' "$ref" > /dev/null; then
    nums=$(jq -r --slurpfile f "$w/f1.json" '[.stats.mesh.area, $f[0].mesh.area,
      .stats.mesh.volume, $f[0].mesh.volume, ($f[0].engine.refine_cpu_edges // 0)]
      | map(tostring) | join(",")' "$ref")
  fi
  echo "$same,$fsame,$rep,$nums"
}

# One shard: every case of every input in $1, rows to $2.
shard() {
  local inputs=$1 csv=$2 w f c surf h
  w=$(mktemp -d "$out/work.XXXXXX")
  while read -r f; do
    [ -n "$f" ] || continue
    for c in "${cases[@]}"; do
      surf=${c%%:*}
      h=${c#*:}
      local args=("$DATA/$f" --surface "$surf" -s "$h" --quiet ${SIMD:+--simd "$SIMD"})
      local key rc rce rcf
      key=$(printf '%s-%s-%s' "$f" "$surf" "$h" | tr '/' '_')
      if [ "$MODE" = gpu ]; then
        rc=$(cat "$REF/$key.rc" 2>/dev/null || echo missing)
        cp "$REF/$key.json" "$w/cpu.ref" 2>/dev/null || echo '{}' > "$w/cpu.ref"
      else
        "$BIN" "${args[@]}" --backend cpu -o "$w/cpu.obj" --stats "$w/cpu.json" 2>/dev/null && rc=0 || rc=$?
        reference "$w" > "$w/cpu.ref"
      fi
      if [ "$MODE" = cpu ]; then
        echo "$rc" > "$out/ref/$key.rc"
        mv "$w/cpu.ref" "$out/ref/$key.json"
        rm -f "$w"/*
        continue
      fi
      "$BIN" "${args[@]}" --backend "$BACKEND" -o "$w/exact.obj" --stats "$w/exact.json" 2>/dev/null && rce=0 || rce=$?
      "$BIN" "${args[@]}" --backend "$BACKEND" --refine fast -o "$w/f1.obj" --stats "$w/f1.json" 2>/dev/null && rcf=0 || rcf=$?
      "$BIN" "${args[@]}" --backend "$BACKEND" --refine fast -o "$w/f2.obj" 2>/dev/null || true
      local row
      row=$(compare "$w")
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
if [ "$MODE" = cpu ]; then
  echo "$(find "$out/ref" -name '*.rc' | wc -l) CPU reference record(s) in $out/ref"
  exit 0
fi
"$here/summary.sh" "$out"/gate-*.csv
