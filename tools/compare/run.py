#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-tool comparison harness (GitHub issues #5, #6).

Runs inside the sesmesh-compare container (tools/compare/Containerfile) with
the bench directory mounted at /work. Standard library only.

  run.py prepare ENTRY...                 sphere files for structures
  run.py run --tool T --spacing H ENTRY... [--threads N] [--repeat R]
             [--timeout S] [--out FILE.jsonl] [--keep] [--tag TAG]

ENTRY is a PDB id in /work/data/cache (ID.cif.gz), "@set" for a set of
tests/data/sets.toml, a synthetic xyzr name in /work/data/synthetic, or
ID:asm1 for biological assembly 1. Every run appends one JSON object per
(tool, entry, repeat) to the output file: wall/user/sys time and peak RSS
from GNU time, the program's own reported area/volume where it prints them,
and the meshcheck analysis of its mesh (topology, area, volume, distance of
the vertices to the exact SES of the same spheres, normal errors).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

W = Path("/work")
BIN = W / "opt" / "bin"
DATA = W / "data"
SPH = DATA / "spheres"
SYN = DATA / "synthetic"
SETS = W / "src" / "sesmesh" / "tests" / "data" / "sets.toml"
PROBE = 1.4


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def entries(args):
    out = []
    for a in args:
        if a.startswith("@"):
            sets = tomllib.loads(SETS.read_text())["sets"]
            out += [f.split(".")[0] for f in sets[a[1:]]["files"]]
        else:
            out.append(a)
    return out


def prefix(entry):
    return SPH / entry.replace(":", "_")


def synth():
    """Degenerate geometries (Design §15 robustness corpus), probe 1.4 Å."""
    import math
    import random
    SYN.mkdir(parents=True, exist_ok=True)
    sets = {}
    ful = W / "src" / "NanoShaper" / "example" / "fullerene.xyzr"
    if ful.exists():
        sets["fullerene"] = [tuple(map(float, l.split()[:4])) for l in ful.read_text().splitlines() if l.strip()]
    sets["lattice"] = [(3.0 * i, 3.0 * j, 3.0 * k, 1.6) for i in range(4) for j in range(4) for k in range(4)]
    phi = (1 + 5 ** 0.5) / 2
    ico = [(0, s1, s2 * phi) for s1 in (-1, 1) for s2 in (-1, 1)]
    ico = [p for v in ico for p in (v, (v[1], v[2], v[0]), (v[2], v[0], v[1]))]
    dod = [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    dod += [p for s1 in (-1, 1) for s2 in (-1, 1)
            for p in ((0, s1 / phi, s2 * phi), (s1 / phi, s2 * phi, 0), (s2 * phi, 0, s1 / phi))]
    cage = []
    for v in ico + dod:
        n = math.sqrt(sum(c * c for c in v))
        cage.append((6.0 * v[0] / n, 6.0 * v[1] / n, 6.0 * v[2] / n, 1.5))
    sets["cage"] = cage
    sets["collinear"] = [(2.0 * i, 0.0, 0.0, 1.5) for i in range(8)]
    rng = random.Random(3)
    box = [(rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(1.2, 2.0)) for _ in range(30)]
    box += [box[i] for i in range(0, 30, 4)]            # exact duplicates
    box.append((box[0][0] + 0.1, box[0][1], box[0][2], box[0][3] - 0.5))  # nested sphere
    sets["duplicates"] = box
    a = 4 * (1.6 + 1.4) / math.sqrt(6)                  # probe fits the tetrahedral hole exactly
    tet = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    sets["exactfit"] = [(a / (2 * math.sqrt(2)) * x, a / (2 * math.sqrt(2)) * y, a / (2 * math.sqrt(2)) * z, 1.6)
                        for x, y, z in tet]
    sets["tangent"] = [(0.0, 0.0, 0.0, 1.6), (6.0, 0.0, 0.0, 1.6)]  # SAS spheres touch
    chain = [(0.0, 0.0, 0.0, 1.7)]
    for _ in range(19):
        x, y, z, _r = chain[rng.randrange(len(chain))]
        u = [rng.gauss(0, 1) for _ in range(3)]
        n = math.sqrt(sum(c * c for c in u))
        d = rng.uniform(1.3, 1.6)
        chain.append((x + d * u[0] / n, y + d * u[1] / n, z + d * u[2] / n, rng.uniform(1.2, 2.0)))
    sets["offset"] = [(x + 1.0e4, y - 2.0e4, z + 1.5e4, r) for x, y, z, r in chain]
    sets["single"] = [(0.3, -0.2, 0.1, 1.8)]
    for name, rows in sets.items():
        (SYN / f"{name}.xyzr").write_text("".join(f"{x:.4f} {y:.4f} {z:.4f} {r:.4f}\n" for x, y, z, r in rows))
        print(name, len(rows))


def prepare(entry):
    p = prefix(entry)
    if p.with_suffix(".xyzr").exists():
        return json.loads(p.with_suffix(".json").read_text())
    SPH.mkdir(parents=True, exist_ok=True)
    if (SYN / f"{entry}.xyzr").exists():
        rows = [l.split() for l in (SYN / f"{entry}.xyzr").read_text().splitlines() if l.strip()]
        p.with_suffix(".xyzr").write_text("".join(f"{float(x):.3f} {float(y):.3f} {float(z):.3f} {float(r):.4f}\n"
                                                  for x, y, z, r in rows))
        pqr, eses, pdb = [], [], []
        for k, (x, y, z, r) in enumerate(rows, 1):
            x, y, z, r = map(float, (x, y, z, r))
            pqr.append(f"ATOM {k} C UNK A 1 {x:.3f} {y:.3f} {z:.3f} 0.0000 {r:.4f}")
            eses.append(f"ATOM  {k % 100000:>5}  C   UNK A   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}{0:>8.4f}{r:>7.4f}")
            pdb.append(f"ATOM  {k % 100000:>5}  C   UNK A   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C")
        p.with_suffix(".pqr").write_text("\n".join(pqr) + "\n")
        p.with_suffix(".eses.pqr").write_text("\n".join(eses) + "\n")
        p.with_suffix(".pdb").write_text("\n".join(pdb) + "\nEND\n")
        info = {"atoms": len(rows), "warnings": 0}
    else:
        pid, _, asm = entry.partition(":")
        cmd = [str(BIN / "meshcheck"), "spheres", str(DATA / "cache" / f"{pid}.cif.gz"), str(p)]
        if asm:
            cmd += ["--assembly", asm.removeprefix("asm")]
        r = sh(cmd)
        if r.returncode:
            raise RuntimeError(f"{entry}: {r.stderr.strip()}")
        info = json.loads(r.stdout)
    p.with_suffix(".json").write_text(json.dumps(info))
    return info


# ----------------------------------------------------------------- adapters
# Each returns (argv, files to write {name: text}, mesh file, mesh format,
# environment additions). Run in a fresh scratch directory.

def a_sesmesh(e, h, n, opts):
    cmd = [str(BIN / "sesmesh"), str(prefix(e).with_suffix(".pqr")), "-s", str(h),
           "-j", str(n), "-o", "mesh.ply", "--stats", "stats.json"] + opts
    return cmd, {}, "mesh.ply", "ply", {}


def a_msms(e, h, n, opts):
    d = 1.5 / h ** 2  # sesmesh vertices·h²/area = 1.5 (validation report §3)
    cmd = ["msms", "-if", str(prefix(e).with_suffix(".xyzr")), "-of", "mesh",
           "-probe_radius", str(PROBE), "-density", f"{d:g}", "-all_components"] + opts
    return cmd, {}, "mesh.vert", "msms", {}


def a_nanoshaper(e, h, n, opts):
    smooth = "true" if "smooth" in opts else "false"
    conf = f"""Grid_scale = {1 / h:g}
Grid_perfil = 80.0
XYZR_FileName = {prefix(e).with_suffix('.xyzr')}
Build_epsilon_maps = false
Build_status_map = true
Surface = ses
Smooth_Mesh = {smooth}
Number_thread = {n}
Cavity_Detection_Filling = false
Probe_Radius = {PROBE}
Accurate_Triangulation = true
Triangulation = true
Check_duplicated_vertices = true
Save_Mesh_MSMS_Format = false
Compute_Vertex_Normals = true
"""
    return [str(BIN / "NanoShaper"), "conf.prm"], {"conf.prm": conf}, "triangulatedSurf.off", "off", {}


def a_eses(e, h, n, opts):
    return ([str(BIN / "MS_Intersection"), str(prefix(e)) + ".eses.pqr", str(PROBE), str(h), "2.0"],
            {}, "marching_cubes.obj", "obj", {"OMP_NUM_THREADS": str(n)})


def a_edtsurf(e, h, n, opts):
    # Own radii and atom typing (input: PDB of the selected atoms); -h 1 keeps
    # cavities, -s 3 molecular surface, -f grid points per Å.
    return (["EDTSurf", "-i", str(prefix(e).with_suffix(".pdb")), "-o", "mesh", "-s", "3", "-c", "1",
             "-p", str(PROBE), "-f", f"{1 / h:g}", "-h", "1"] + opts, {}, "mesh.ply", "ply", {})


CX_SCRIPT = """import time
import numpy as np
from chimerax.core.commands import run
# ChimeraX reads no PQR: open the PDB of the same atoms, then set every
# atom radius from the PQR (matched by coordinates) so the spheres are
# identical to the other programs'.
rad = {{}}
for line in open('{pqr}'):
    t = line.split()
    if t and t[0] in ('ATOM', 'HETATM'):
        rad[(t[-5], t[-4], t[-3])] = float(t[-1])
m = run(session, 'open {pdb} format pdb')[0]
atoms = m.atoms
atoms.radii = np.array([rad[tuple('%.3f' % v for v in c)] for c in atoms.coords], dtype=np.float32)
t = time.perf_counter()
surfs = run(session, 'surface #1 enclose #1 include #1 probeRadius {probe} gridSpacing {h} nthread {n}')
dt = time.perf_counter() - t
with open('mesh.obj', 'w') as f:
    base = 1
    for s in surfs:
        v, n, tri = s.vertices, s.normals, s.triangles
        f.writelines('v %.5f %.5f %.5f\\n' % tuple(p) for p in v)
        if n is not None:
            f.writelines('vn %.5f %.5f %.5f\\n' % tuple(q) for q in n)
        f.writelines('f %d//%d %d//%d %d//%d\\n' % (a, a, b, b, c, c) for a, b, c in tri + base)
        base += len(v)
open('cx_time.txt', 'w').write('%f %d\\n' % (dt, len(surfs)))
"""


def a_chimerax(e, h, n, opts):
    script = CX_SCRIPT.format(pqr=prefix(e).with_suffix(".pqr"), pdb=prefix(e).with_suffix(".pdb"),
                              probe=PROBE, h=h, n=n)
    return (["chimerax", "--nogui", "--offscreen", "--exit", "--script", "{wd}/cx.py"],
            {"cx.py": script}, "mesh.obj", "obj", {"OMP_NUM_THREADS": str(n)})


def a_sesa(e, h, n, opts):
    # Analytic SES area only; own CHARMM radii, adds hydrogens.
    return ([ "/opt/sesA/bin/sesa", "in.pdb"], {}, None, None, {})


ADAPTERS = {"sesmesh": a_sesmesh, "msms": a_msms, "nanoshaper": a_nanoshaper, "eses": a_eses,
            "edtsurf": a_edtsurf, "chimerax": a_chimerax, "sesa": a_sesa}


def reported(tool, out, wd):
    """Area/volume the program itself prints (analytic where available)."""
    r = {}
    if tool == "msms":
        # "Comp. probe_radius SES_volume SES_area" (numerical) and the
        # analytical "Comp. probe_radius, reent, toric and contact SES area"
        # MSMS prints the analytic areas once per attempt (it restarts with
        # perturbed radii after triangulation problems): use the last block.
        blocks = out.split("ANALYTICAL SURFACE AREA")
        r["attempts"] = len(blocks) - 1
        f = r"-?[\d.]+"
        if len(blocks) > 1:
            ana = re.findall(rf"^\s*(\d+)\s+{f}\s+({f})\s+({f})\s+({f})\s+({f})\s+({f})\s*$", blocks[-1], re.M)
            r["ses_area_analytic"] = sum(float(a[4]) for a in ana)
            r["ses_area_analytic_comp0"] = float(ana[0][4]) if ana else None
            r["analytic_components"] = len(ana)
        nb = out.split("NUMERICAL VOLUMES AND AREA")
        if len(nb) > 1:
            num = re.findall(rf"^\s*(\d+)\s+{f}\s+({f})\s+({f})\s*$", nb[-1], re.M)
            r["num_volume"] = sum(float(v) for _, v, _ in num)
            r["num_area"] = sum(float(a) for _, _, a in num)
            r["mesh_components"] = len(num)
        for k, pat in (("w_hole", "warning hole"), ("w_genus", "Genus mismatch"),
                       ("w_geodesic", "geodesic which intersects"), ("w_pbr", "PBR problems")):
            r[k] = out.count(pat)
    elif tool == "nanoshaper":
        for k, pat in (("area", r"Total (?:surface )?area\s*[:=]?\s*([\d.eE+-]+)"),
                       ("volume", r"Total (?:surface )?volume\s*[:=]?\s*([\d.eE+-]+)")):
            m = re.search(pat, out, re.I)
            if m:
                r[k] = float(m.group(1))
    elif tool == "sesa":
        f = next(iter(wd.glob("*protSES.txt")), None)
        if f:
            m = re.search(r"Total:\s+sasArea=\s*([\d.]+)\s+torusArea=\s*([\d.]+)\s+probeArea=\s*([\d.]+)", f.read_text())
            if m:
                r["ses_area_analytic"] = sum(float(x) for x in m.groups())
            m = re.search(r"sasAreaOfAllCavities=\s*([\d.]+)", f.read_text())
            if m:
                r["cavities_sas_area"] = float(m.group(1))
    elif tool == "sesmesh":
        f = wd / "stats.json"
        if f.exists():
            s = json.loads(f.read_text())
            r["area"] = s.get("mesh", {}).get("area")
            r["volume"] = s.get("mesh", {}).get("volume")
            r["kernel"] = s.get("engine", {}).get("kernel")
            r["timings_ms"] = s.get("timings_ms")
    elif tool == "chimerax":
        f = wd / "cx_time.txt"
        if f.exists():
            dt, ns = f.read_text().split()
            r["surface_s"], r["surfaces"] = float(dt), int(ns)
    return r


def run_one(tool, entry, h, n, repeat, timeout, keep, opts, scratch):
    info = prepare(entry)
    cmd, files, mesh, fmt, env = ADAPTERS[tool](entry, h, n, opts)
    wd = scratch / f"{tool}-{entry.replace(':', '_')}-{h}-{n}-{repeat}"
    cmd = [c.replace("{wd}", str(wd)) for c in cmd]
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    for k, v in files.items():
        (wd / k).write_text(v)
    if tool == "sesa":
        shutil.copy(prefix(entry).with_suffix(".pdb"), wd / "in.pdb")
        if not (wd.parent / "bioParam").exists():
            shutil.copytree("/opt/sesA/bioParam", wd.parent / "bioParam")
    rec = {"tool": tool, "entry": entry, "atoms": info["atoms"], "spacing": h, "threads": n,
           "host": os.environ.get("BENCH_HOST", ""), "mem_cap": os.environ.get("MEM_CAP", ""),
           "repeat": repeat, "opts": opts, "cmd": " ".join(cmd)}
    t0 = time.time()
    try:
        p = subprocess.run(["/usr/bin/time", "-f", "%e %U %S %M", "-o", "time.txt"] + cmd, cwd=wd,
                           capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **env})
        rec["exit"] = p.returncode
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as ex:
        rec["exit"] = "timeout"
        out = (ex.stdout or b"").decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
    rec["elapsed"] = round(time.time() - t0, 3)
    tf = wd / "time.txt"
    if tf.exists():
        try:
            e_, u_, s_, m_ = tf.read_text().split()[-4:]
            rec.update(wall=float(e_), user=float(u_), sys=float(s_), maxrss_mb=int(m_) / 1024)
        except ValueError:
            pass
    rec["reported"] = reported(tool, out, wd)
    rec["log_tail"] = out[-1500:]
    if mesh:
        mp = wd / mesh
        if mp.exists() and mp.stat().st_size > 0:
            rec["mesh_bytes"] = mp.stat().st_size
            # Content hash for the determinism check: comment lines (`#` in
            # text formats, `comment` in a PLY header) are skipped, since some
            # programs write a creation date there (NanoShaper's OFF header).
            hsh = hashlib.sha256()
            for part in (sorted(wd.glob("mesh*.vert")) + sorted(wd.glob("mesh*.face")) if fmt == "msms" else [mp]):
                if not part.exists():
                    continue
                data = part.read_bytes()
                if fmt == "ply":
                    head, sep, body = data.partition(b"end_header\n")
                    hsh.update(b"\n".join(l for l in head.split(b"\n") if not l.startswith(b"comment")) + sep + body)
                else:
                    hsh.update(b"\n".join(l for l in data.split(b"\n") if not l.lstrip().startswith(b"#")))
            rec["mesh_sha256"] = hsh.hexdigest()
            c = sh([str(BIN / "meshcheck"), "check", str(prefix(entry).with_suffix(".xyzr")), str(mp),
                    "--format", fmt, "--probe", str(PROBE), "--components"])
            rec["check"] = json.loads(c.stdout) if c.returncode == 0 else {"error": c.stderr.strip()[-500:]}
        else:
            rec["check"] = {"error": "no mesh written"}
    rec["status"] = ("ok" if rec["exit"] == 0 and (not mesh or "error" not in rec.get("check", {}))
                     else "fail")
    if not keep:
        shutil.rmtree(wd, ignore_errors=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p1 = sp.add_parser("prepare")
    p1.add_argument("entries", nargs="+")
    sp.add_parser("synth")
    p2 = sp.add_parser("run")
    p2.add_argument("--tool", required=True, choices=sorted(ADAPTERS))
    p2.add_argument("--spacing", type=float, default=0.5)
    p2.add_argument("--threads", type=int, default=os.cpu_count())
    p2.add_argument("--repeat", type=int, default=1)
    p2.add_argument("--timeout", type=float, default=1800)
    p2.add_argument("--out", default="/work/runs/results.jsonl")
    p2.add_argument("--keep", action="store_true")
    p2.add_argument("--opt", action="append", default=[], help="extra program option(s)")
    p2.add_argument("entries", nargs="+")
    a = ap.parse_args()
    if a.cmd == "synth":
        synth()
        return
    if a.cmd == "prepare":
        for e in entries(a.entries):
            try:
                print(e, prepare(e))
            except Exception as ex:  # noqa: BLE001 - report and continue
                print(e, "ERROR", ex)
        return
    scratch = Path(os.environ.get("SCRATCH", "/work/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = [o for s in a.opt for o in s.split()]
    for e in entries(a.entries):
        for r in range(a.repeat):
            rec = run_one(a.tool, e, a.spacing, a.threads, r, a.timeout, a.keep, opts, scratch)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            c = rec.get("check", {})
            print(f"{a.tool:10s} {e:14s} h={a.spacing} j={a.threads} {rec['status']:4s} "
                  f"wall={rec.get('wall')} rss={rec.get('maxrss_mb', 0):.0f}MB "
                  f"tri={c.get('triangles')} closed={c.get('closed')} area={c.get('area')} "
                  f"dmax={c.get('dist', {}).get('max')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
