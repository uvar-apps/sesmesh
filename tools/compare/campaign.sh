#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Launches comparison streams on a host, each a detached podman container
# pinned to its own CPUs and capped in memory, appending to runs/<name>.jsonl:
#   campaign.sh stream NAME CPUS MEM -- "run.py arguments" ["run.py arguments"...]
#   campaign.sh status
# Each quoted argument is one run.py invocation; they run in order.
# Example: campaign.sh stream acc-msms 4 12g -- "run --tool msms --spacing 0.5 @validation"
set -euo pipefail
B=${BENCH:-$HOME/sesmesh-bench}
case "${1:-}" in
  stream)
    name=$2 cpus=$3 mem=$4; shift 5
    mkdir -p "$B/runs"
    podman run -d --name "cmp-$name" --replace --cpuset-cpus "$cpus" --memory "$mem" \
      -e BENCH_HOST="$(hostname)" -e MEM_CAP="$mem" -e SCRATCH="/work/scratch/$name" \
      -v "$B:/work" localhost/sesmesh-compare \
      bash -c "$(for c in "$@"; do printf 'python3 /work/src/sesmesh/tools/compare/run.py %s --out /work/runs/%s.jsonl >> /work/runs/%s.log 2>&1; ' "$c" "$name" "$name"; done) echo EXIT \$? >> /work/runs/$name.log"
    ;;
  status)
    podman ps -a --filter name=cmp- --format '{{.Names}} {{.Status}}'
    for f in "$B"/runs/*.log; do printf '%-28s %5s lines  %s\n' "$(basename "$f")" "$(wc -l < "$f")" "$(tail -1 "$f" | cut -c1-110)"; done
    ;;
  *) echo "usage: campaign.sh stream NAME CPUS MEM -- ARGS... | status" >&2; exit 2 ;;
esac
