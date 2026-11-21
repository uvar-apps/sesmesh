#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Compares one gate case's files (gate.sh): prints the CSV columns
exact_same,faces_same,fast_repeat_same,area_cpu,area_fast,volume_cpu,
volume_fast,cpu_edges ('-' where a file is missing)."""

import json
import sys
from pathlib import Path


def stats(path):
    """Statistics without the keys that differ between back ends."""
    d = json.loads(path.read_text())
    d.pop("timings_ms", None)
    for k in ("engine", "parameters"):
        for x in ("backend", "device", "gpu_device", "refine_cpu_edges"):
            d.get(k, {}).pop(x, None)
    return d


def faces(path):
    with path.open("rb") as f:
        return [line for line in f if line.startswith(b"f ")]


def main():
    w = Path(sys.argv[1])
    cpu, exact, f1, f2 = (w / n for n in ("cpu.obj", "exact.obj", "f1.obj", "f2.obj"))
    cols = []
    try:
        same = cpu.read_bytes() == exact.read_bytes()
        same = same and stats(w / "cpu.json") == stats(w / "exact.json")
        cols.append("yes" if same else "no")
    except OSError:
        cols.append("-")
    try:
        cols.append("yes" if faces(cpu) == faces(f1) else "no")
        cols.append("yes" if f1.read_bytes() == f2.read_bytes() else "no")
    except OSError:
        cols += ["-", "-"]
    try:
        c = json.loads((w / "cpu.json").read_text())["mesh"]
        f = json.loads((w / "f1.json").read_text())
        e = f.get("engine", {}).get("refine_cpu_edges", 0)
        f = f["mesh"]
        cols += [str(c["area"]), str(f["area"]), str(c["volume"]), str(f["volume"]), str(e)]
    except (OSError, KeyError, ValueError):
        cols += ["-"] * 5
    print(",".join(cols))


if __name__ == "__main__":
    main()
