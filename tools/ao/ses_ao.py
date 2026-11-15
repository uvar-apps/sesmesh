#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Ambient-occluded molecular surfaces: sesmesh, then aomesh.

    python3 tools/ao/ses_ao.py data/1crn.cif -o out -- -q high --vertex-colors element

For each input, sesmesh writes an OBJ surface and aomesh bakes ambient
occlusion into it. The output directory receives aomesh's three files:

    <stem>_ao.obj   the surface, with one `vt u u` per vertex: its AO value
    <stem>_ao.mtl   one white material whose `map_Kd` is the ramp below
    <stem>_ao.png   a 256 x 1 grey ramp, black (occluded) to white (open)

Options after `--` go to `sesmesh surface` unchanged; the script names the
outputs, so `-o`, `--output`, `--output-dir` and `--output-format` are refused
there. An input ending in `.obj` is taken as an existing surface and only
baked. The intermediate surface is deleted unless `--keep-surface` is given.

`--bake-colors` also writes `<stem>_ao.ply`, a binary PLY laid out like
sesmesh's (x y z, nx ny nz, red green blue, triangles) whose colours are the
surface colours times the AO value, for viewers that show vertex colours but
not textures; `tools/site/ply2web.py` reads it too. Without
`--vertex-colors` the surface is white and the PLY is grey.

The tools are found from `--sesmesh` / `--aomesh`, the `SESMESH` / `AOMESH`
environment variables, `PATH`, then this checkout's `target/release`.
aomesh 0.4.0 or later is expected: older versions write `map_Kd -imfchan l`,
which three.js reads as part of the file name, and drop the surface's header,
objects and groups. Python standard library only.
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The script decides where sesmesh writes; these would move the surface away.
RESERVED = ("--output", "--output-dir", "--output-format")
MIN_AOMESH = (0, 4, 0)


class Failure(Exception):
    """A step failed; `code` is the exit status to report."""

    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def find_tool(name, explicit=None):
    """Path of `name`: the option, the environment, PATH, this checkout's build."""
    for candidate in (explicit, os.environ.get(name.upper())):
        if candidate:
            return candidate
    found = shutil.which(name)
    if found:
        return found
    local = REPO / "target" / "release" / name
    return str(local) if local.is_file() else None


def tool_version(path):
    """(major, minor, patch) from `<tool> --version`, or None."""
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    return tuple(int(x) for x in match.groups()) if match else None


def input_stem(path):
    """Output stem of an input, as sesmesh names it: `1crn.cif.gz` -> `1crn`."""
    name = Path(path).name
    for suffix in (".gz", ".GZ"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return Path(name).stem or "surface"


def check_passthrough(args):
    """Refuse sesmesh options that would write the surface somewhere else."""
    for arg in args:
        name = arg.split("=", 1)[0]
        short_o = arg.startswith("-o") and not arg.startswith("--")
        if name in RESERVED or short_o:
            raise Failure(
                f"'{arg}' cannot be passed to sesmesh: ses_ao.py names the outputs "
                "(use -o/--output-dir of ses_ao.py)",
                2,
            )


def visibility(u):
    """AO value in [0, 1] from an aomesh texture coordinate.

    aomesh writes u = (0.5 + 255 vis) / 256, the centre of the ramp texel
    that holds the grey of vis.
    """
    return min(1.0, max(0.0, (256.0 * u - 0.5) / 255.0))


def read_ao_obj(path):
    """The parts of an aomesh OBJ that a coloured PLY needs.

    Returns (header, positions, colours, normals, ao, triangles): header comment
    lines, flat x y z floats, flat r g b floats in [0, 1] (None without vertex
    colours), flat per-vertex normals (None without `vn`), one AO value per
    vertex, flat triangle indices. A polygon is split into a fan.
    """
    header, positions, colours, coords, normal_table = [], [], [], [], []
    corners_seen = []  # (v, vn) per corner, bound after reading
    triangles = []
    with open(path, encoding="utf-8") as f:
        at_start = True
        for line in f:
            if at_start and line.startswith("#"):
                header.append(line[1:].rstrip("\n"))
                continue
            at_start = False
            if line.startswith("v "):
                p = line.split()
                positions.extend((float(p[1]), float(p[2]), float(p[3])))
                if len(p) >= 7:
                    colours.extend((float(p[4]), float(p[5]), float(p[6])))
            elif line.startswith("vt "):
                coords.append(float(line.split()[1]))
            elif line.startswith("vn "):
                p = line.split()
                normal_table.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                face = []
                for corner in line.split()[1:]:
                    fields = corner.split("/")
                    v = int(fields[0]) - 1
                    face.append(v)
                    if len(fields) == 3 and fields[2]:
                        corners_seen.append((v, int(fields[2]) - 1))
                for k in range(1, len(face) - 1):
                    triangles.extend((face[0], face[k], face[k + 1]))

    count = len(positions) // 3
    if len(coords) != count:
        raise Failure(
            f"{path}: {len(coords)} texture coordinates for {count} vertices; "
            "not an aomesh output"
        )
    if colours and len(colours) != 3 * count:
        raise Failure(f"{path}: only some vertices carry colours")
    normals = None
    if normal_table:
        # A position takes the normal its first corner names (objserde's rule).
        normals = [0.0] * (3 * count)
        bound = [False] * count
        for v, vn in corners_seen:
            if not bound[v]:
                bound[v] = True
                normals[3 * v : 3 * v + 3] = normal_table[vn]
    ao = [visibility(u) for u in coords]
    return header, positions, colours or None, normals, ao, triangles


def write_ply(path, header, positions, colours, normals, ao, triangles, strength):
    """Binary little-endian PLY with colour x (1 - strength (1 - ao)) per vertex."""
    count = len(ao)
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {h.strip()}" for h in header if h.strip()]
    lines.append(
        f"comment ses_ao.py: colours x ambient occlusion, strength {strength:g}"
    )
    lines.append(f"element vertex {count}")
    lines += [f"property float {c}" for c in ("x", "y", "z")]
    if normals is not None:
        lines += [f"property float {c}" for c in ("nx", "ny", "nz")]
    lines += [f"property uchar {c}" for c in ("red", "green", "blue")]
    lines.append(f"element face {len(triangles) // 3}")
    lines.append("property list uchar uint vertex_indices")
    lines.append("end_header")

    vertex = struct.Struct("<6f3B" if normals is not None else "<3f3B")
    face = struct.Struct("<B3I")
    out = bytearray(("\n".join(lines) + "\n").encode("ascii"))
    for i in range(count):
        shade = 1.0 - strength * (1.0 - ao[i])
        rgb = colours[3 * i : 3 * i + 3] if colours else (1.0, 1.0, 1.0)
        rgb = [min(255, max(0, round(255.0 * c * shade))) for c in rgb]
        xyz = positions[3 * i : 3 * i + 3]
        if normals is not None:
            out += vertex.pack(*xyz, *normals[3 * i : 3 * i + 3], *rgb)
        else:
            out += vertex.pack(*xyz, *rgb)
    for t in range(0, len(triangles), 3):
        out += face.pack(3, *triangles[t : t + 3])
    Path(path).write_bytes(out)


def bake_colors(obj_path, ply_path, strength):
    """Write the coloured PLY of an aomesh OBJ."""
    write_ply(ply_path, *read_ao_obj(obj_path), strength)


def run(command, quiet):
    """Run one tool; its failure becomes a Failure with the tool's exit code."""
    result = subprocess.run(command, stdout=subprocess.DEVNULL if quiet else None)
    if result.returncode != 0:
        name = Path(command[0]).name
        # A tool killed by signal N reports -N; a shell would report 128 + N.
        code = result.returncode if result.returncode > 0 else 128 - result.returncode
        raise Failure(f"{name} exited with status {code}", code)


def process(source, opts, sesmesh, aomesh, passthrough):
    """sesmesh then aomesh for one input; returns the files written."""
    out_dir = Path(opts.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_stem(source)
    target = out_dir / f"{stem}_ao.obj"
    written = [target, target.with_suffix(".mtl"), target.with_suffix(".png")]
    timings = []
    # The surface goes to a private directory: a file of the same name already
    # in the output directory is never overwritten, nor deleted afterwards.
    with tempfile.TemporaryDirectory(prefix=".ses_ao-", dir=out_dir) as tmp:
        if source.lower().endswith(".obj"):
            surface = Path(source)
        else:
            surface = Path(tmp) / f"{stem}.obj"
            command = [sesmesh, "surface", source, "-o", str(surface), *passthrough]
            if opts.quiet and "--quiet" not in passthrough:
                command.append("--quiet")
            start = time.perf_counter()
            run(command, opts.quiet)
            timings.append(f"sesmesh {time.perf_counter() - start:.2f} s")

        command = [aomesh, str(surface), "-o", str(target), "--samples", str(opts.samples)]
        if opts.preview:
            written.append(out_dir / f"{stem}_ao_preview.png")
        else:
            command.append("--no-preview")
        if opts.quiet:
            command.append("--quiet")
        start = time.perf_counter()
        run(command, opts.quiet)
        timings.append(f"aomesh {time.perf_counter() - start:.2f} s")

        if opts.bake_colors:
            ply = target.with_suffix(".ply")
            start = time.perf_counter()
            bake_colors(target, ply, opts.strength)
            timings.append(f"colours {time.perf_counter() - start:.2f} s")
            written.append(ply)
        if opts.keep_surface and surface.parent == Path(tmp):
            kept = out_dir / f"{stem}.obj"
            shutil.move(surface, kept)
            written.insert(0, kept)
    return written, timings


def parse_args(argv):
    """Split at `--`: ses_ao.py options before, sesmesh options after."""
    if "--" in argv:
        split = argv.index("--")
        own, passthrough = argv[:split], argv[split + 1 :]
    else:
        own, passthrough = argv, []
    parser = argparse.ArgumentParser(
        prog="ses_ao.py",
        description="Ambient-occluded molecular surfaces: sesmesh, then aomesh. "
        "Options after `--` go to `sesmesh surface`.",
        epilog="example: ses_ao.py data/1crn.cif -o out -- -q high --vertex-colors element",
    )
    parser.add_argument("inputs", nargs="+", help="structure files, or .obj surfaces")
    parser.add_argument(
        "-o", "--output-dir", default=".", help="directory for the outputs (default: .)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="aomesh viewpoints; cost is linear (default: 256)",
    )
    parser.add_argument(
        "--bake-colors",
        action="store_true",
        help="also write <stem>_ao.ply with colours x AO in the vertex colours",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="with --bake-colors: 0 leaves colours alone, 1 (default) is full AO",
    )
    parser.add_argument(
        "--preview", action="store_true", help="keep aomesh's <stem>_ao_preview.png"
    )
    parser.add_argument(
        "--keep-surface",
        action="store_true",
        help="keep sesmesh's surface as <stem>.obj",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with the next input after a failure",
    )
    parser.add_argument("--sesmesh", help="sesmesh executable")
    parser.add_argument("--aomesh", help="aomesh executable")
    parser.add_argument("--quiet", action="store_true", help="print errors only")
    opts = parser.parse_args(own)
    if opts.samples < 1:
        parser.error("--samples must be at least 1")
    if not 0.0 <= opts.strength <= 1.0:
        parser.error("--strength must be between 0 and 1")
    return opts, passthrough


def main(argv=None):
    """Command-line entry point; returns the exit status."""
    opts, passthrough = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        check_passthrough(passthrough)
        needs_sesmesh = any(not s.lower().endswith(".obj") for s in opts.inputs)
        sesmesh = find_tool("sesmesh", opts.sesmesh)
        aomesh = find_tool("aomesh", opts.aomesh)
        if needs_sesmesh and sesmesh is None:
            raise Failure("sesmesh not found: pass --sesmesh or set SESMESH", 2)
        if aomesh is None:
            raise Failure(
                "aomesh not found: `cargo install aomesh-cli`, or pass --aomesh "
                "or set AOMESH",
                2,
            )
        version = tool_version(aomesh)
        if version is not None and version < MIN_AOMESH and not opts.quiet:
            print(
                f"ses_ao: warning: aomesh {'.'.join(map(str, version))} writes "
                "`map_Kd -imfchan l` (three.js cannot load it) and drops groups "
                "and header comments; use 0.4.0 or later",
                file=sys.stderr,
            )
    except Failure as failure:
        print(f"ses_ao: error: {failure}", file=sys.stderr)
        return failure.code

    status = 0
    for source in opts.inputs:
        try:
            written, timings = process(source, opts, sesmesh, aomesh, passthrough)
        except Failure as failure:
            print(f"ses_ao: error: {source}: {failure}", file=sys.stderr)
            if not opts.keep_going:
                return failure.code
            status = 1
            continue
        if not opts.quiet:
            names = ", ".join(p.name for p in written)
            print(
                f"ses_ao: {source}: wrote {names} in {opts.output_dir} "
                f"({', '.join(timings)})",
                file=sys.stderr,
            )
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
