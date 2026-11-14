#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Condense the comparison runs (runs/*.jsonl from run.py) into CSV tables
and a Markdown summary. Standard library only; runs anywhere.

  report.py RUNS_DIR OUT_DIR [--accuracy docs/validation/accuracy.csv]

Writes OUT_DIR/runs.csv (one row per run), OUT_DIR/summary.md and the
per-campaign tables used by docs/comparison.md and the paper.
"""

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

TOOLS = ["sesmesh", "nanoshaper", "eses", "msms", "chimerax", "edtsurf", "sesa"]
SAME_SPHERES = {"sesmesh", "nanoshaper", "eses", "msms", "chimerax"}


def load(runs):
    recs = []
    for f in sorted(Path(runs).glob("*.jsonl")):
        campaign = f.stem.split("-")[0]
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["campaign"] = campaign
                r["source"] = f.stem
                recs.append(r)
    return recs


def ext_refs(path):
    """sesmesh h -> 0 extrapolation (A0, V0) per entry from accuracy.csv."""
    out = {}
    if not path or not Path(path).exists():
        return out
    for r in csv.DictReader(open(path)):
        if float(r["spacing"]) == 0.25:
            a, v = float(r["area"]), float(r["volume"])
            out[r["entry"]] = (a / (1 + float(r["area_err"])), v / (1 + float(r["volume_err"])))
    return out


HOSTS = {}


def anon_host(name):
    """Machine names stay out of the published data: host-a, host-b, … in the
    order of the sorted names (stable across runs of this script)."""
    return HOSTS.get(name, "")


def row(r, refs):
    c = r.get("check", {}) or {}
    d = c.get("dist", {}) or {}
    n = c.get("normals", {}) or {}
    w = c.get("welded") or {}
    rep = r.get("reported", {}) or {}
    comps = c.get("component_list") or []
    out = {
        "campaign": r["campaign"], "host": anon_host(r.get("host", "")), "tool": r["tool"], "entry": r["entry"],
        "atoms": r.get("atoms"), "spacing": r.get("spacing"), "threads": r.get("threads"),
        "repeat": r.get("repeat"), "opts": " ".join(r.get("opts", [])), "status": r.get("status"),
        "exit": r.get("exit"), "wall_s": r.get("wall"), "user_s": r.get("user"),
        "maxrss_mb": round(r["maxrss_mb"], 1) if r.get("maxrss_mb") else None,
        "mem_cap": r.get("mem_cap", ""), "vertices": c.get("vertices"), "triangles": c.get("triangles"),
        "closed": c.get("closed"), "oriented": c.get("oriented"),
        "boundary_edges": c.get("boundary_edges"), "nonmanifold_edges": c.get("nonmanifold_edges"),
        "misoriented_edges": c.get("misoriented_edges"),
        "coincident_vertices": c.get("coincident_vertices"),
        "closed_welded": w.get("closed", c.get("closed")), "oriented_welded": w.get("oriented", c.get("oriented")),
        "degenerate_tris": (c.get("degenerate_area") or 0) + (c.get("degenerate_index") or 0) if c else None,
        "duplicate_tris": c.get("duplicate_triangles"),
        "components": c.get("components"), "cavities": sum(1 for x in comps if x[2] < 0) if comps else None,
        "area": c.get("area"), "volume": c.get("volume"),
        "dist_p50": d.get("p50"), "dist_p99": d.get("p99"), "dist_max": d.get("max"),
        "frac_gt_0.01": d.get("gt_0.01"), "frac_gt_0.1": d.get("gt_0.1"),
        "normal_mean_deg": n.get("mean_deg"), "normal_p99_deg": n.get("p99_deg"), "normals_inward": n.get("inward"),
        "msms_analytic_area": rep.get("ses_area_analytic"), "msms_attempts": rep.get("attempts"),
        "reported_area": rep.get("area") or rep.get("ses_area_analytic"),
        "sha256": r.get("mesh_sha256", ""),
        "kernel": rep.get("kernel"),
        "mesh_ms": (rep.get("timings_ms") or {}).get("mesh"),
        "total_ms": (rep.get("timings_ms") or {}).get("total"),
        "chimerax_surface_s": rep.get("surface_s"),
    }
    ref = refs.get(r["entry"])
    if ref and c.get("area"):
        out["area_err_vs_A0"] = c["area"] / ref[0] - 1
        out["volume_err_vs_V0"] = c["volume"] / ref[1] - 1 if c.get("volume") else None
        if rep.get("ses_area_analytic"):
            out["analytic_area_err_vs_A0"] = rep["ses_area_analytic"] / ref[0] - 1
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def pct(x, d=2):
    return "–" if x is None else f"{100 * x:+.{d}f}"


def fmt(x, spec):
    return "–" if x is None else format(x, spec)


def summary(rows):
    md = ["# Cross-tool comparison: summary", ""]
    by = defaultdict(list)
    for r in rows:
        by[(r["campaign"], r["tool"], r["spacing"])].append(r)
    md += ["## Status and topology per campaign", "",
           "| campaign | tool | h (Å) | runs | ok | closed+oriented as written | after welding | with non-manifold edges | with vertices > 0.1 Å off | median p99 distance (Å) | worst distance (Å) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (camp, tool, h), rs in sorted(by.items(), key=lambda k: (k[0][0], TOOLS.index(k[0][1]) if k[0][1] in TOOLS else 9, k[0][2] or 0)):
        ok = [r for r in rs if r["status"] == "ok"]
        meshed = [r for r in ok if r["triangles"]]
        co = sum(1 for r in meshed if r["closed"] and r["oriented"])
        cw = sum(1 for r in meshed if r["closed_welded"] and r["oriented_welded"])
        nm = sum(1 for r in meshed if (r["nonmanifold_edges"] or 0) > 0)
        off = sum(1 for r in meshed if (r["dist_max"] or 0) > 0.1)
        md.append(f"| {camp} | {tool} | {h} | {len(rs)} | {len(ok)} | {co}/{len(meshed)} | {cw}/{len(meshed)} | {nm} | {off} | "
                  f"{fmt(med([r['dist_p99'] for r in meshed]), '.2e')} | {fmt(max([r['dist_max'] or 0 for r in meshed], default=None), '.3g')} |")
    md += ["", "## Area and volume against the sesmesh h → 0 extrapolation (same spheres)", "",
           "| tool | h (Å) | entries | area: median | area: worst | volume: median | volume: worst |", "|---|---|---|---|---|---|---|"]
    for (camp, tool, h), rs in sorted(by.items(), key=lambda k: (TOOLS.index(k[0][1]) if k[0][1] in TOOLS else 9, k[0][2] or 0)):
        if camp != "acc" or tool not in SAME_SPHERES:
            continue
        a = [r["area_err_vs_A0"] for r in rs if r.get("area_err_vs_A0") is not None]
        v = [r["volume_err_vs_V0"] for r in rs if r.get("volume_err_vs_V0") is not None]
        if a:
            md.append(f"| {tool} | {h} | {len(a)} | {pct(med(a))} | {pct(max(a, key=abs))} | {pct(med(v))} | {pct(max(v, key=abs)) if v else '–'} |")
    an = [r["analytic_area_err_vs_A0"] for r in rows if r["tool"] == "msms" and r.get("analytic_area_err_vs_A0") is not None and r["spacing"] == 0.5 and r["campaign"] == "acc"]
    if an:
        md += ["", f"MSMS analytic SES area vs sesmesh A0 ({len(an)} entries): median {pct(med(an), 3)} %, range {pct(min(an), 3)} … {pct(max(an), 3)} %."]
    return "\n".join(md) + "\n"


def cavities(recs):
    """Cavity agreement with sesmesh (negative-volume components matched by
    centroid within 1 Å) on the accuracy corpus."""
    import math
    by = defaultdict(dict)
    for r in recs:
        c = r.get("check") or {}
        if r["campaign"] == "acc" and "component_list" in c:
            by[(r["entry"], r["spacing"])][r["tool"]] = [x for x in c["component_list"] if x[2] < 0]

    def unmatched(a_, b_):
        return [x for x in a_ if not b_ or min(math.dist(x[3], y[3]) for y in b_) > 1.0]

    md = ["", "## Cavities on the accuracy corpus (matched to sesmesh by centroid within 1 Å)", "",
          "| tool | h (Å) | entries | cavities | sesmesh cavities | not matched | median max dist of unmatched (Å) | sesmesh cavities not found |",
          "|---|---|---|---|---|---|---|---|"]
    rows = []
    for h in (0.5, 0.25):
        for t in TOOLS:
            pairs = [(d[t], d["sesmesh"]) for (e, hh), d in by.items() if hh == h and t in d and "sesmesh" in d]
            if not pairs or t == "sesmesh":
                continue
            n = sum(len(a_) for a_, _ in pairs)
            ns = sum(len(b_) for _, b_ in pairs)
            ua = [x for a_, b_ in pairs for x in unmatched(a_, b_)]
            ub = [x for a_, b_ in pairs for x in unmatched(b_, a_)]
            mf = med([x[4] for x in ua])
            md.append(f"| {t} | {h} | {len(pairs)} | {n} | {ns} | {len(ua)} | {fmt(mf, '.3f')} | {len(ub)} |")
            rows.append({"tool": t, "spacing": h, "entries": len(pairs), "cavities": n, "sesmesh_cavities": ns,
                         "unmatched": len(ua), "unmatched_median_maxdist": mf, "sesmesh_not_found": len(ub)})
    for e in ("2PTN",):
        for h in (0.5, 0.25):
            d = by.get((e, h), {})
            if d:
                md.append("")
                md.append(f"{e} at {h} Å: " + "; ".join(f"{t} {len(c)} cavities, {sum(-x[2] for x in c):.1f} Å³"
                                                         for t, c in sorted(d.items(), key=lambda kv: TOOLS.index(kv[0]))))
    return "\n".join(md) + "\n", rows


def determinism(recs):
    """Distinct mesh contents per (tool, entry) over thread counts/repeats."""
    # Runs whose hash skips comment lines (NanoShaper writes a date into its
    # OFF header) replace older runs of the same program.
    det = [r for r in recs if r["campaign"] == "det" and r.get("mesh_sha256")]
    newer = {r["tool"] for r in det if r["source"] != "det"}
    by = defaultdict(lambda: defaultdict(list))
    for r in det:
        if r["tool"] in newer and r["source"] == "det":
            continue
        by[r["tool"]][r["entry"]].append((r["threads"], r["mesh_sha256"]))
    md = ["", "## Determinism (h = 0.5 Å; distinct mesh contents over thread counts and repeats)", "",
          "| tool | entry | runs (threads) | distinct outputs |", "|---|---|---|---|"]
    for t in TOOLS:
        for e, v in sorted(by.get(t, {}).items()):
            th = sorted({x[0] for x in v})
            md.append(f"| {t} | {e} | {len(v)} ({', '.join(map(str, th))}) | {len({x[1] for x in v})} |")
    return "\n".join(md) + "\n"


def perf(rows):
    """Median wall time and peak RSS per (tool, entry, h, threads, opts)."""
    by = defaultdict(list)
    for r in rows:
        if r["campaign"] == "perf":
            by[(r["tool"], r["entry"], r["spacing"], r["threads"], r["opts"])].append(r)
    md = ["", "## Performance (one idle 16-core Zen 5 host; median of repeats)", "",
          "| tool | entry | h (Å) | threads | opts | runs | status | wall (s) | peak RSS (MB) | triangles |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for (t, e, h, n, o), rs in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][2], TOOLS.index(kv[0][0]) if kv[0][0] in TOOLS else 9, -(kv[0][3] or 0))):
        ok = [r for r in rs if r["status"] == "ok"]
        w = med([r["wall_s"] for r in ok])
        m = max([r["maxrss_mb"] or 0 for r in ok], default=None)
        status = "ok" if ok else (rs[0]["exit"] if rs else "?")
        md.append(f"| {t} | {e} | {h} | {n} | {o} | {len(rs)} | {status} | {fmt(w, '.3g')} | {fmt(m, '.0f')} | {ok[0]['triangles'] if ok else '–'} |")
        out.append({"tool": t, "entry": e, "spacing": h, "threads": n, "opts": o, "runs": len(rs), "status": status,
                    "wall_s_median": w, "maxrss_mb": m, "triangles": ok[0]["triangles"] if ok else None})
    return "\n".join(md) + "\n", out


PROVENANCE = {
    "msms": ("MSMS 2.6.2 (Sanner lab release of 2026-11-06, static Linux x86-64 binary "
             "msms.x86_64Linux2.2.6.2.staticgcc)",
             "msms -if ID.xyzr -of mesh -probe_radius 1.4 -density 1.5/h^2 -all_components",
             "same spheres as sesmesh (xyzr)"),
    "nanoshaper": ("NanoShaper 0.7.5 (source github.com/shi-scala/NanoShaper @ 107d844; CGAL 5.6, "
                   "Boost 1.83, GCC 13.3, one missing #include <memory> added)",
                   "NanoShaper conf.prm with Surface = ses, Grid_scale = 1/h, Grid_perfil = 80, Probe_Radius = 1.4, "
                   "Accurate_Triangulation = true, Smooth_Mesh = false, Build_status_map = true, "
                   "Cavity_Detection_Filling = false, Compute_Vertex_Normals = true",
                   "same spheres as sesmesh (xyzr)"),
    "eses": ("ESES (source github.com/WeilabMSU/ESES @ 0b4cd51, GCC 13.3, OpenMP)",
             "MS_Intersection ID.eses.pqr 1.4 h 2.0",
             "same spheres as sesmesh (fixed-column PQR, ATOM records only)"),
    "chimerax": ("UCSF ChimeraX 1.12 (2026-06-12, Ubuntu 24.04 package)",
                 "open ID.pdb; set every atom radius from ID.pqr; "
                 "surface #1 enclose #1 include #1 probeRadius 1.4 gridSpacing h nthread N",
                 "same spheres as sesmesh (radii set from the PQR)"),
    "edtsurf": ("EDTSurf 0.2009 (Debian/Ubuntu package 0.2009-10)",
                "EDTSurf -i ID.pdb -o mesh -s 3 -c 1 -p 1.4 -f 1/h -h 1",
                "own radii and atom typing from the PDB records (not comparable sphere by sphere)"),
    "sesa": ("sesa V001 (github.com/wlincong/SESexact, sesaV001.tar.gz Linux binary)",
             "sesa ID.pdb",
             "own CHARMM radii, adds hydrogens (not comparable sphere by sphere)"),
}


def toml_val(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(round(v, 6))
    return '"' + str(v).replace('"', "'") + '"'


def references(recs, outdir):
    """tests/data/reference/<tool>.toml: derived numbers per corpus entry and
    spacing (never the programs' output files), with provenance."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in recs:
        if r["campaign"] == "acc" and r["tool"] in PROVENANCE:
            by[r["tool"]].append(r)
    for tool, rs in by.items():
        ver, cmd, sph = PROVENANCE[tool]
        lines = [
            "# sesmesh — Solvent Excluded Surface meshes for molecules",
            "# Copyright (c) 2026 Ugo Varetto",
            "# SPDX-License-Identifier: MIT OR Apache-2.0",
            "#",
            f"# Reference numbers from {tool} for the validation corpus (GitHub issue #5),",
            "# written by tools/compare/report.py from the runs of tools/compare/run.py.",
            "# Only derived numbers are stored, never the program's output files.",
            "# area/volume are those of the program's mesh (signed volume, cavities",
            "# negative), measured by tools/compare/meshcheck; cavities = closed",
            "# components with negative volume; dist_* = distance of the mesh vertices",
            "# to the exact SES of the same spheres (sesmesh's SesField).",
            "",
            "[provenance]",
            f"program = {toml_val(ver)}",
            f"command = {toml_val(cmd)}",
            f"spheres = {toml_val(sph + '; atoms: sesmesh default selection (first model, no H or waters, highest-occupancy alt-locs), Bondi radii, probe 1.4 A')}",
            'machine = "AMD Ryzen AI Max+ 395 (16 cores), Ubuntu 24.04 container (tools/compare/Containerfile)"',
            'date = "2026-11-14"',
            "",
        ]
        for r in sorted(rs, key=lambda r: (r["entry"], r["spacing"])):
            c = r.get("check") or {}
            rep = r.get("reported") or {}
            comps = c.get("component_list") or []
            e = {"id": r["entry"], "spacing": r["spacing"], "atoms": r["atoms"], "status": r["status"]}
            if c and "area" in c:
                e.update(triangles=c["triangles"], area=c["area"], volume=c["volume"],
                         components=c["components"], cavities=sum(1 for x in comps if x[2] < 0),
                         closed=bool(c["closed"] and c["oriented"]),
                         dist_p99=c["dist"]["p99"], dist_max=c["dist"]["max"])
            if rep.get("ses_area_analytic") is not None:
                e["analytic_area"] = rep["ses_area_analytic"]
            if rep.get("num_volume") is not None:
                e["numerical_volume"] = rep["num_volume"]
            if rep.get("attempts"):
                e["attempts"] = rep["attempts"]
            if r["status"] != "ok":
                e["exit"] = str(r.get("exit"))
            lines.append("[[entry]]")
            lines += [f"{k} = {toml_val(v)}" for k, v in e.items()]
            lines.append("")
        (outdir / f"{tool}.toml").write_text("\n".join(lines))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("out")
    ap.add_argument("--references", help="write tests/data/reference/*.toml here")
    ap.add_argument("--accuracy", default=str(Path(__file__).resolve().parents[2] / "docs" / "validation" / "accuracy.csv"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = ext_refs(a.accuracy)
    recs = load(a.runs)
    for k, h in enumerate(sorted({r.get("host") for r in recs if r.get("host")})):
        HOSTS[h] = f"host-{chr(ord('a') + k)}"
    rows = [row(r, refs) for r in recs]
    write_csv(out / "runs.csv", rows)
    cav_md, cav_rows = cavities(recs)
    perf_md, perf_rows = perf(rows)
    write_csv(out / "cavities.csv", cav_rows)
    write_csv(out / "performance.csv", perf_rows)
    (out / "summary.md").write_text(summary(rows) + cav_md + determinism(recs) + perf_md)
    if a.references:
        references(recs, a.references)
    print(f"{len(rows)} runs -> {out}/runs.csv, summary.md")


if __name__ == "__main__":
    main()
