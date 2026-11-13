#!/usr/bin/env python3
"""Convert a sesmesh binary PLY (with vertex colours) into the compact,
gzip-compressed mesh format read by site/assets/viewer.js.

    python3 tools/site/ply2web.py in.ply site/models/name.sesm.gz

Format "SESM" version 1, little endian (display copy, not for analysis):
    0   magic  b"SESM"
    4   u32    version (1)
    8   u32    vertex count V
    12  u32    triangle count T
    16  f32x3  bounding-box minimum (Å)
    28  f32x3  bounding-box maximum (Å)
    40  u32    3T vertex indices (counter-clockwise seen from outside)
    ..  u16    3V positions, quantized in the bounding box
    ..  i8     3V normals × 127
    ..  u8     3V colours (RGB; 160 grey without colours)
"""
import gzip
import struct
import sys

import numpy as np

TYPES = {"float": "f4", "uchar": "u1", "uint": "u4", "int": "i4", "ushort": "u2",
         "char": "i1", "short": "i2", "double": "f8"}


def read_ply(path):
    data = open(path, "rb").read()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    assert header[0] == "ply" and header[1].startswith("format binary_little_endian")
    elements, cur = [], None
    for line in header:
        parts = line.split()
        if parts[0] == "element":
            cur = [parts[1], int(parts[2]), []]
            elements.append(cur)
        elif parts[0] == "property":
            cur[2].append(parts[1:])
    off = end
    vertex = faces = None
    for name, count, props in elements:
        if name == "vertex":
            dt = np.dtype([(p[1], "<" + TYPES[p[0]]) for p in props])
            vertex = np.frombuffer(data, dtype=dt, count=count, offset=off)
            off += dt.itemsize * count
        elif name == "face":
            # sesmesh writes "list uchar uint vertex_indices", always triangles.
            dt = np.dtype([("n", "u1"), ("i", "<u4", (3,))])
            faces = np.frombuffer(data, dtype=dt, count=count, offset=off)
            assert (faces["n"] == 3).all()
            off += dt.itemsize * count
    return vertex, faces["i"]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    v, tris = read_ply(src)
    pos = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    q = np.round((pos - lo) / span * 65535).astype("<u2")
    names = v.dtype.names
    if "nx" in names:
        n = np.stack([v["nx"], v["ny"], v["nz"]], axis=1)
    else:
        n = np.zeros_like(pos)
    nq = np.clip(np.round(n * 127), -127, 127).astype("i1")
    if "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype("u1")
    else:
        rgb = np.full((len(v), 3), 160, dtype="u1")
    out = struct.pack("<4sIII6f", b"SESM", 1, len(v), len(tris), *lo, *hi)
    out += tris.astype("<u4").tobytes() + q.tobytes() + nq.tobytes() + rgb.tobytes()
    with gzip.open(dst, "wb", compresslevel=9) as f:
        f.write(out)
    print(f"{dst}: {len(v)} vertices, {len(tris)} triangles, {len(out)} B raw")


if __name__ == "__main__":
    main()
