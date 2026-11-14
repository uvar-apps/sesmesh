#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Performance campaign of the cross-tool comparison: one stream per host,
# all 16 cores, no other load, so wall times are comparable. The two
# hosts are identical (AMD Ryzen AI Max+ 395, 16 cores, SMT off,
# performance governor), so the programs are split between them:
#   perf.sh a   sesmesh (16 and 1 threads), NanoShaper, MSMS, EDTSurf, sesa,
#               then sesmesh SIMD levels and thread scaling (issues #8, #10)
#   perf.sh b   ESES, ChimeraX
# Multi-threaded programs run with 16 threads; MSMS, EDTSurf and sesa are
# single-threaded. Matched resolution: grid spacing h (MSMS density 1.5/h²).
set -euo pipefail
C=$(dirname "$0")/campaign.sh
small="1CRN 4HHB"            # repeated 3 times
case "${1:-}" in
  a)
    bash "$C" stream perf-a 0-15 110g -- \
      "run --tool sesmesh --spacing 0.25 --threads 16 --repeat 3 1CRN 4HHB 1AON" \
      "run --tool sesmesh --spacing 0.5 --threads 16 --repeat 3 4HHB 1AON 1JJ2 4V6X" \
      "run --tool sesmesh --spacing 1.0 --threads 16 --repeat 3 3J3Q 1K4R:asm1" \
      "run --tool sesmesh --spacing 0.25 --threads 1 --repeat 3 $small" \
      "run --tool sesmesh --spacing 0.5 --threads 1 --repeat 3 1AON" \
      "run --tool nanoshaper --spacing 0.25 --threads 16 --repeat 3 --timeout 3600 1CRN 4HHB 1AON" \
      "run --tool nanoshaper --spacing 0.5 --threads 16 --repeat 3 --timeout 3600 4HHB 1AON 1JJ2 4V6X" \
      "run --tool nanoshaper --spacing 1.0 --threads 16 --timeout 3600 3J3Q 1K4R:asm1" \
      "run --tool nanoshaper --spacing 0.25 --threads 1 --repeat 3 $small" \
      "run --tool msms --spacing 0.25 --threads 1 --repeat 3 --timeout 3600 1CRN 4HHB" \
      "run --tool msms --spacing 0.25 --threads 1 --timeout 3600 1AON" \
      "run --tool msms --spacing 0.5 --threads 1 --timeout 3600 4HHB 1AON 1JJ2 4V6X" \
      "run --tool msms --spacing 1.0 --threads 1 --timeout 3600 3J3Q 1K4R:asm1" \
      "run --tool edtsurf --spacing 0.25 --threads 1 --repeat 3 --timeout 3600 1CRN 4HHB 1AON" \
      "run --tool edtsurf --spacing 0.5 --threads 1 --repeat 3 --timeout 3600 4HHB 1AON 1JJ2 4V6X" \
      "run --tool edtsurf --spacing 1.0 --threads 1 --timeout 3600 3J3Q 1K4R:asm1" \
      "run --tool sesa --spacing 0.5 --threads 1 --timeout 3600 1CRN 4HHB 1AON 1JJ2 4V6X 3J3Q 1K4R:asm1" \
      "run --tool sesmesh --spacing 0.25 --threads 16 --repeat 3 --opt '--simd scalar' 4HHB 1AON" \
      "run --tool sesmesh --spacing 0.25 --threads 16 --repeat 3 --opt '--simd sse4.2' 4HHB 1AON" \
      "run --tool sesmesh --spacing 0.25 --threads 16 --repeat 3 --opt '--simd avx2' 4HHB 1AON" \
      "run --tool sesmesh --spacing 0.25 --threads 16 --repeat 3 --opt '--simd avx512' 4HHB 1AON" \
      "run --tool sesmesh --spacing 0.25 --threads 1 --repeat 3 --opt '--simd scalar' 4HHB" \
      "run --tool sesmesh --spacing 0.25 --threads 1 --repeat 3 --opt '--simd avx2' 4HHB" \
      "run --tool sesmesh --spacing 0.25 --threads 1 --repeat 3 --opt '--simd avx512' 4HHB" \
      "run --tool sesmesh --spacing 0.5 --threads 2 --repeat 3 1AON 1JJ2" \
      "run --tool sesmesh --spacing 0.5 --threads 4 --repeat 3 1AON 1JJ2" \
      "run --tool sesmesh --spacing 0.5 --threads 8 --repeat 3 1AON 1JJ2" \
      "run --tool sesmesh --spacing 0.5 --threads 12 --repeat 3 1AON 1JJ2" \
      "run --tool sesmesh --spacing 0.5 --threads 1 --repeat 3 1JJ2" \
      "run --tool sesmesh --spacing 0.5 --threads 16 --repeat 3 1JJ2"
    ;;
  b)
    bash "$C" stream perf-b 0-15 110g -- \
      "run --tool chimerax --spacing 0.25 --threads 16 --repeat 3 --timeout 3600 1CRN 4HHB" \
      "run --tool chimerax --spacing 0.25 --threads 16 --timeout 3600 1AON" \
      "run --tool chimerax --spacing 0.5 --threads 16 --timeout 3600 4HHB 1AON 1JJ2 4V6X" \
      "run --tool chimerax --spacing 1.0 --threads 16 --timeout 3600 3J3Q 1K4R:asm1" \
      "run --tool eses --spacing 0.25 --threads 16 --repeat 3 --timeout 3600 1CRN 4HHB" \
      "run --tool eses --spacing 0.25 --threads 16 --timeout 3600 1AON" \
      "run --tool eses --spacing 0.5 --threads 16 --timeout 3600 4HHB 1AON 1JJ2 4V6X" \
      "run --tool eses --spacing 1.0 --threads 16 --timeout 3600 3J3Q 1K4R:asm1"
    ;;
  *) echo "usage: perf.sh a|b" >&2; exit 2 ;;
esac
