#!/usr/bin/env bash
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Totals of gate.sh's CSV files; exits 1 on any difference.
#
#   summary.sh GATE_CSV...
#
# Fails when the exit codes differ, when exact mode's files are not the
# CPU's, when fast mode's faces are not the CPU's or a repeated fast run
# differs, or when fast mode's area or volume is more than 1e-6 (relative)
# from the CPU's.
set -euo pipefail

awk -F, -v tol=1e-6 '
function abs(x) { return x < 0 ? -x : x }
function rel(a, b) { a += 0; b += 0; return a != 0 ? abs(b / a - 1) : abs(b) }
{
  n++
  what = $1 " " $2 " " $3
  if ($4 == "0" && $5 == "0" && $6 == "0") ok++
  if (!($4 == $5 && $5 == $6)) {
    bad[++nb] = what ": exit codes cpu " $4 ", exact " $5 ", fast " $6
    next
  }
  if ($4 != "0") next
  if ($7 != "yes") bad[++nb] = what ": exact files differ"
  if ($8 != "yes") bad[++nb] = what ": fast faces differ"
  if ($9 != "yes") bad[++nb] = what ": fast repeat differ"
  if ($10 != "-") {
    da = rel($10, $11)
    dv = rel($12, $13)
    edges += $14
    if (da > mda) mda = da
    if (dv > mdv) mdv = dv
    if (da > tol || dv > tol)
      bad[++nb] = sprintf("%s: area %.2e, volume %.2e from the CPU'\''s", what, da, dv)
  }
}
END {
  printf "%d cases, %d meshed on every back end; fast |dA| max %.2e, |dV| max %.2e (relative); %d edges refined on the CPU in fast mode\n", n, ok, mda, mdv, edges
  for (i = 1; i <= nb && i <= 50; i++) print "  " bad[i]
  printf "%d difference(s)\n", nb
  exit (nb > 0 ? 1 : 0)
}' "$@"
