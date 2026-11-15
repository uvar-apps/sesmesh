#!/usr/bin/env python3
# sesmesh — Solvent Excluded Surface meshes for molecules
# Copyright (c) 2026 Ugo Varetto
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Tests of ses_ao.py: `python3 -m unittest discover -s tools/ao`.

The unit tests need nothing but Python. The end-to-end test runs sesmesh and
aomesh on crambin and is skipped when either executable is not found (see
ses_ao.py for where it looks) or aomesh is older than 0.4.0.
"""

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ses_ao  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "ses_ao.py"


def coordinate(vis):
    """aomesh's texture coordinate for a visibility (aomesh::texture)."""
    return (0.5 + 255.0 * vis) / 256.0


def read_ply(path):
    """(header lines, vertex records, faces) of a ses_ao PLY."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    count = int(next(h for h in header if h.startswith("element vertex")).split()[2])
    faces = int(next(h for h in header if h.startswith("element face")).split()[2])
    with_normals = "property float nx" in header
    vertex = struct.Struct("<6f3B" if with_normals else "<3f3B")
    off = end
    vertices = []
    for _ in range(count):
        vertices.append(vertex.unpack_from(data, off))
        off += vertex.size
    tris = []
    for _ in range(faces):
        n, a, b, c = struct.unpack_from("<B3I", data, off)
        assert n == 3
        tris.append((a, b, c))
        off += 13
    assert off == len(data), "trailing bytes"
    return header, vertices, tris


def ao_obj(colours=True, normals=True):
    """What aomesh 0.4 writes for a sesmesh surface: the header first, one
    `vt u u` per vertex, corners `v/vt/vn`. One quad, to check the fan."""
    rgb = [" 1 0 0", " 0 1 0", " 0 0 1", " 0.5 0.5 0.5"] if colours else [""] * 4
    xyz = ["0 0 0", "1 0 0", "1 1 0", "0 1 0"]
    lines = ["# sesmesh 1.1.2", "# area_A2: 1.0000"]
    lines += [f"v {p}{c}" for p, c in zip(xyz, rgb)]
    lines += [f"vt {coordinate(a)!r} {coordinate(a)!r}" for a in (1.0, 0.5, 0.0, 0.25)]
    if normals:
        lines += ["vn 0 0 1", "vn 0 0 -1"]
    lines += ["o ses", "g component_0", "usemtl aomesh_ao"]
    for face in (((1, 1), (2, 1), (3, 2)), ((1, 1), (3, 2), (4, 1), (2, 1))):
        corners = (f"{v}/{v}/{n}" if normals else f"{v}/{v}" for v, n in face)
        lines.append("f " + " ".join(corners))
    return "\n".join(lines) + "\n"


class Helpers(unittest.TestCase):
    def test_stem_matches_sesmesh(self):
        self.assertEqual(ses_ao.input_stem("data/1crn.cif"), "1crn")
        self.assertEqual(ses_ao.input_stem("x/4hhb.pdb.gz"), "4hhb")
        self.assertEqual(ses_ao.input_stem("2ptc.CIF.GZ"), "2ptc")
        self.assertEqual(ses_ao.input_stem("surface.obj"), "surface")

    def test_output_options_are_refused_after_the_separator(self):
        for bad in (
            ["-o", "x.obj"],
            ["-ox.obj"],
            ["--output", "x.obj"],
            ["--output=x.ply"],
            ["--output-dir", "d"],
            ["--output-format", "ply"],
        ):
            with self.subTest(bad=bad), self.assertRaises(ses_ao.Failure) as caught:
                ses_ao.check_passthrough(["-q", "high", *bad])
            self.assertEqual(caught.exception.code, 2)
        ses_ao.check_passthrough(["-q", "high", "--vertex-colors", "element", "--probe", "1.4"])

    def test_separator_splits_own_and_sesmesh_options(self):
        opts, rest = ses_ao.parse_args(["a.cif", "b.pdb", "--samples", "64", "--", "-s", "0.5"])
        self.assertEqual(opts.inputs, ["a.cif", "b.pdb"])
        self.assertEqual(opts.samples, 64)
        self.assertEqual(rest, ["-s", "0.5"])
        opts, rest = ses_ao.parse_args(["a.cif"])
        self.assertEqual((opts.output_dir, opts.samples, opts.strength, rest), (".", 256, 1.0, []))

    def test_visibility_inverts_the_aomesh_coordinate(self):
        for vis in (0.0, 0.25, 0.5, 1.0):
            self.assertAlmostEqual(ses_ao.visibility(coordinate(vis)), vis, places=12)
        # Out-of-range coordinates clamp instead of producing negative colours.
        self.assertEqual(ses_ao.visibility(0.0), 0.0)
        self.assertEqual(ses_ao.visibility(1.0), 1.0)


class BakeColors(unittest.TestCase):
    def bake(self, text, strength=1.0):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "m_ao.obj"
            obj.write_text(text)
            ply = Path(tmp) / "m_ao.ply"
            ses_ao.bake_colors(obj, ply, strength)
            return read_ply(ply)

    def test_colours_are_multiplied_by_the_ao_value(self):
        header, vertices, tris = self.bake(ao_obj())
        self.assertIn("comment sesmesh 1.1.2", header)
        self.assertIn("comment area_A2: 1.0000", header)
        self.assertEqual(header[-1], "end_header")
        rgb = [v[6:9] for v in vertices]
        # red x 1, green x 0.5, blue x 0, grey 0.5 x 0.25
        self.assertEqual(rgb, [(255, 0, 0), (0, 128, 0), (0, 0, 0), (32, 32, 32)])
        # Positions and the normal each vertex's first corner names.
        self.assertEqual(vertices[2][:6], (1.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        self.assertEqual(vertices[3][3:6], (0.0, 0.0, 1.0))
        # The quad 1 3 4 2 becomes the fan (1 3 4) (1 4 2).
        self.assertEqual(tris, [(0, 1, 2), (0, 2, 3), (0, 3, 1)])

    def test_strength_scales_the_darkening(self):
        _, vertices, _ = self.bake(ao_obj(), strength=0.5)
        self.assertEqual([v[6:9] for v in vertices][:3], [(255, 0, 0), (0, 191, 0), (0, 0, 128)])
        _, vertices, _ = self.bake(ao_obj(), strength=0.0)
        self.assertEqual(vertices[2][6:9], (0, 0, 255))

    def test_a_surface_without_colours_bakes_grey(self):
        _, vertices, _ = self.bake(ao_obj(colours=False))
        self.assertEqual([v[6:9] for v in vertices], [(255,) * 3, (128,) * 3, (0,) * 3, (64,) * 3])

    def test_a_surface_without_normals_has_no_normal_properties(self):
        header, vertices, _ = self.bake(ao_obj(normals=False))
        self.assertNotIn("property float nx", header)
        self.assertEqual(vertices[3], (0.0, 1.0, 0.0, 32, 32, 32))

    def test_an_obj_without_one_coordinate_per_vertex_is_refused(self):
        text = ao_obj().replace(f"vt {coordinate(0.25)!r} {coordinate(0.25)!r}\n", "")
        with self.assertRaises(ses_ao.Failure):
            self.bake(text)


SESMESH = ses_ao.find_tool("sesmesh")
AOMESH = ses_ao.find_tool("aomesh")
CRAMBIN = ses_ao.REPO / "data" / "1crn.cif"


@unittest.skipUnless(
    SESMESH and AOMESH and CRAMBIN.is_file() and (ses_ao.tool_version(AOMESH) or (0,)) >= ses_ao.MIN_AOMESH,
    "needs sesmesh and aomesh >= 0.4.0 (PATH, SESMESH/AOMESH, or target/release)",
)
class EndToEnd(unittest.TestCase):
    def test_crambin(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            # A file with the surface's name must survive the run.
            out.mkdir()
            (out / "1crn.obj").write_text("mine\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), str(CRAMBIN), "-o", str(out), "--samples", "32",
                 "--bake-colors", "--quiet", "--sesmesh", str(SESMESH), "--aomesh", str(AOMESH),
                 "--", "-s", "1.0", "--vertex-colors", "element", "--group-components"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            names = sorted(p.name for p in out.iterdir())
            self.assertEqual(names, ["1crn.obj", "1crn_ao.mtl", "1crn_ao.obj", "1crn_ao.ply", "1crn_ao.png"])
            self.assertEqual((out / "1crn.obj").read_text(), "mine\n")
            self.assertIn("\nmap_Kd 1crn_ao.png\n", (out / "1crn_ao.mtl").read_text())

            text = (out / "1crn_ao.obj").read_text()
            self.assertTrue(text.startswith("# sesmesh "), text[:80])
            self.assertIn("\ng component_0\n", text)
            self.assertIn("\ng cavity_1\n", text)
            vertices = sum(1 for line in text.splitlines() if line.startswith("v "))
            coords = sum(1 for line in text.splitlines() if line.startswith("vt "))
            self.assertEqual(vertices, coords)
            _, ply_vertices, _ = read_ply(out / "1crn_ao.ply")
            self.assertEqual(len(ply_vertices), vertices)

    def test_a_failing_input_stops_the_batch_unless_keep_going(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.cif")
            common = ["-o", tmp, "--samples", "8", "--quiet", "--sesmesh", str(SESMESH),
                      "--aomesh", str(AOMESH)]
            tail = ["--", "-s", "1.5"]
            stop = subprocess.run(
                [sys.executable, str(SCRIPT), missing, str(CRAMBIN), *common, *tail],
                capture_output=True, text=True,
            )
            self.assertNotEqual(stop.returncode, 0)
            self.assertFalse((Path(tmp) / "1crn_ao.obj").exists())
            going = subprocess.run(
                [sys.executable, str(SCRIPT), missing, str(CRAMBIN), *common, "--keep-going", *tail],
                capture_output=True, text=True,
            )
            self.assertEqual(going.returncode, 1)
            self.assertTrue((Path(tmp) / "1crn_ao.obj").exists())
            self.assertIn("missing.cif", going.stderr)


if __name__ == "__main__":
    unittest.main()
