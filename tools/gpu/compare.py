#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Compares one gate case's files (gate.sh): prints the CSV columns
exact_same,faces_same,fast_repeat_same,area_cpu,area_fast,volume_cpu,
volume_fast,cpu_edges ('-' where a file is missing).

`compare.py --reference DIR` instead prints the CPU run's reference record
(digests of the OBJ and its face lines, and the statistics) as JSON, so the
GPU runs can be checked on another machine (gate.sh's REF). Where DIR holds
`cpu.ref` and no `cpu.obj`, the comparison uses that record.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def strip(d):
    """Statistics without the keys that differ between back ends."""
    d.pop("timings_ms", None)
    for k in ("engine", "parameters"):
        for x in ("backend", "device", "gpu_device", "refine_cpu_edges"):
            d.get(k, {}).pop(x, None)
    return d


def stats(path):
    return strip(json.loads(path.read_text()))


def _sha256(cmd, stdin=None):
    """SHA-256 printed by `sha256sum` reading `stdin` (or the file in `cmd`)."""
    out = subprocess.run(cmd, stdin=stdin, capture_output=True, check=False)
    if out.returncode != 0:
        raise OSError(out.stderr.decode(errors="replace").strip())
    return out.stdout.split()[0].decode()


def faces(path):
    """SHA-256 of the face lines (`grep` and `sha256sum`: OBJ files reach tens
    of GB, too large for a Python loop)."""
    if not path.is_file():
        raise OSError(f"{path}: no such file")
    env = dict(os.environ, LC_ALL="C")
    grep = subprocess.Popen(["grep", "-a", "^f ", str(path)], stdout=subprocess.PIPE, env=env)
    try:
        h = _sha256(["sha256sum"], stdin=grep.stdout)
    finally:
        grep.stdout.close()
        rc = grep.wait()
    if rc > 1:  # 1: no face lines, hashed as empty
        raise OSError(f"grep failed on {path}")
    return h


def digest(path):
    """SHA-256 of the file (`sha256sum`)."""
    return _sha256(["sha256sum", str(path)])


def reference(w):
    """The CPU run's record: digests and statistics (raises OSError when the
    run wrote no files)."""
    return {
        "obj": digest(w / "cpu.obj"),
        "faces": faces(w / "cpu.obj"),
        "stats": json.loads((w / "cpu.json").read_text()),
    }


def main():
    if sys.argv[1] == "--reference":
        try:
            print(json.dumps(reference(Path(sys.argv[2])), sort_keys=True))
        except OSError:
            print("{}")
        return
    w = Path(sys.argv[1])
    exact, f1, f2 = (w / n for n in ("exact.obj", "f1.obj", "f2.obj"))
    if (w / "cpu.obj").exists() or not (w / "cpu.ref").exists():
        try:
            ref = reference(w)
        except OSError:
            ref = {}
    else:
        ref = json.loads((w / "cpu.ref").read_text())
    cols = []
    try:
        same = ref["obj"] == digest(exact)
        same = same and strip(json.loads(json.dumps(ref["stats"]))) == stats(w / "exact.json")
        cols.append("yes" if same else "no")
    except (OSError, KeyError):
        cols.append("-")
    try:
        cols.append("yes" if ref["faces"] == faces(f1) else "no")
        cols.append("yes" if digest(f1) == digest(f2) else "no")
    except (OSError, KeyError):
        cols += ["-", "-"]
    try:
        c = ref["stats"]["mesh"]
        f = json.loads((w / "f1.json").read_text())
        e = f.get("engine", {}).get("refine_cpu_edges", 0)
        f = f["mesh"]
        cols += [str(c["area"]), str(f["area"]), str(c["volume"]), str(f["volume"]), str(e)]
    except (OSError, KeyError, ValueError):
        cols += ["-"] * 5
    print(",".join(cols))


if __name__ == "__main__":
    main()
