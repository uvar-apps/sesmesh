#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Totals of gate.sh's CSV files; exits 1 on any difference.

Fails when the exit codes differ, when exact mode's files are not the CPU's,
when fast mode's faces are not the CPU's or a repeated fast run differs, or
when fast mode's area or volume is more than 1e-6 (relative) from the CPU's.
"""

import csv
import sys

TOLERANCE = 1e-6


def rel(a, b):
    a, b = float(a), float(b)
    return abs(b / a - 1) if a else abs(b)


def main(paths):
    rows = [r for p in paths for r in csv.reader(open(p))]
    bad = []
    da, dv, edges = [], [], 0
    for r in rows:
        f, surf, h, rc, rce, rcf, same, faces, rep, ac, af, vc, vf, e = r
        what = f"{f} {surf} {h}"
        if not rc == rce == rcf:
            bad.append(f"{what}: exit codes cpu {rc}, exact {rce}, fast {rcf}")
            continue
        if rc != "0":
            continue
        for name, v in (("exact files", same), ("fast faces", faces), ("fast repeat", rep)):
            if v != "yes":
                bad.append(f"{what}: {name} differ")
        if ac != "-":
            da.append(rel(ac, af))
            dv.append(rel(vc, vf))
            edges += int(e)
            if da[-1] > TOLERANCE or dv[-1] > TOLERANCE:
                bad.append(f"{what}: area {da[-1]:.2e}, volume {dv[-1]:.2e} from the CPU's")
    ok = sum(1 for r in rows if r[3] == r[4] == r[5] == "0")
    print(
        f"{len(rows)} cases, {ok} meshed on every back end; "
        f"fast |dA| max {max(da, default=0):.2e}, |dV| max {max(dv, default=0):.2e} (relative); "
        f"{edges} edges refined on the CPU in fast mode"
    )
    for b in bad[:50]:
        print("  " + b)
    print(f"{len(bad)} difference(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
