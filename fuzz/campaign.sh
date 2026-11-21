#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Long fuzzing campaign (Plan T9.8, issue #22) on x86-64 Linux with a nightly
# toolchain and cargo-fuzz:
#
#   fuzz/campaign.sh build                      build every target (ASan,
#                                               release, debug assertions)
#   fuzz/campaign.sh run TARGET WORKERS SECONDS one target, WORKERS libFuzzer
#                                               processes in fork mode for
#                                               SECONDS (WORKERS × SECONDS ≥
#                                               86 400 for 24 CPU-hours)
#
# The corpus is seeded from tests/data/fixtures (decompressed) and grows
# in fuzz/corpus/TARGET; crashes, timeouts and out-of-memory inputs are
# kept in fuzz/artifacts/TARGET/ and the run goes on. The libFuzzer log
# and GNU time's report (CPU time of all workers) go to
# fuzz/logs/TARGET.{log,time}.
set -euo pipefail
cd "$(dirname "$0")"
triple=x86_64-unknown-linux-gnu
case "${1:-}" in
  build)
    cargo fuzz build --target "$triple"
    ;;
  run)
    tgt=$2 workers=$3 secs=$4
    mkdir -p "corpus/$tgt" "artifacts/$tgt" logs
    for f in ../tests/data/fixtures/*; do
      case "$f" in
        *.gz) gzip -dc "$f" > "corpus/$tgt/$(basename "$f" .gz)" ;;
        *.md) ;;
        *) cp "$f" "corpus/$tgt/" ;;
      esac
    done
    /usr/bin/time -v -o "logs/$tgt.time" "target/$triple/release/$tgt" "corpus/$tgt" \
      -fork="$workers" -max_total_time="$secs" -rss_limit_mb=2048 \
      -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1 \
      -artifact_prefix="artifacts/$tgt/" > "logs/$tgt.log" 2>&1
    ;;
  *)
    echo "usage: fuzz/campaign.sh build | run TARGET WORKERS SECONDS" >&2
    exit 2
    ;;
esac
