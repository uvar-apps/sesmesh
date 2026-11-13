# sesmesh output formats

Reference for the files written by `sesmesh surface` (Spec FR-OUT-1/2/5/6/7,
Design §10). All coordinates are in Å in the input frame; areas in Å²,
volumes in Å³.

## OBJ mesh (`-o <path>`)

```
# sesmesh 0.1.0
# input: 1crn.cif  format: mmcif  model: 1  atoms: 327 selected / 327 read
# surface: ses  probe_radius: 1.4000  grid_spacing: 0.2500  radii: bondi  refine: exact
# cavities: keep  min_cavity_volume: 0  min_component_area: 0  normals: analytic
# vertices: 55542  triangles: 111076  components: 2  cavities: 1  removed: 0  euler: 4
# area_A2: 2314.2175  volume_A3: 4662.7007
# normals point to the solvent; faces counter-clockwise seen from outside
o ses
v 3.2500 5.9735 -8.5000
...
vn 0.1234 -0.9876 0.0954
...
g component_0
f 1//1 2//2 3//3
...
g cavity_1
f ...
```

* `-o -` writes to standard output; a `.gz` suffix compresses.
* `--precision N` (0–9, default 4) sets the decimals of `v` and `vn`.
* `--no-normals` drops `vn` and writes faces as `f a b c`;
  `--face-normals` writes area-weighted face-averaged normals instead of the
  analytic ones.
* `--group-components` adds one `g` record per connected component, named
  `component_<i>` (outer surfaces) or `cavity_<i>` (internal cavities, negative
  signed volume); `<i>` is the component index of the statistics below.
  Triangles of a component are always contiguous.
* `--vertex-colors element|chain|patch|component` writes `v x y z r g b`
  (colour components 0–1, four decimals; read by MeshLab, Blender and others):
  * `element` — the nearest atom's element, from the bundled palette
    (`crates/sesmesh/data/atom-colors.json`, 118 elements) or a file given with
    `--palette` in the same format:
    `{"format": "eigenvista-atom-colors", "version": 1, "colors": [{"z": 6,
    "symbol": "C", "color": "#535b64"}, ...]}` (each entry needs `z` or
    `symbol`, and `color`; entries override the built-in colours);
  * `chain` — ten categorical colours in order of first appearance of the
    nearest atom's chain;
  * `patch` — convex (light blue), saddle (green), concave (orange);
  * `component` — categorical colour of the component index.

## Binary PLY (`--output-format ply`, or `-o name.ply`)

Little-endian binary PLY 1.0 with header comments (the same lines as the OBJ
header):

| Element / property | Type | Present |
|---|---|---|
| `vertex`: `x`, `y`, `z` | float | always |
| `nx`, `ny`, `nz` | float | unless `--no-normals` |
| `red`, `green`, `blue` | uchar | with `--vertex-colors` |
| `atom` | uint | always: 0-based nearest selected atom |
| `patch` | uchar | always: 0 convex, 1 saddle, 2 concave |
| `component` | uint | always (from `sesmesh`) |
| `face`: `vertex_indices` | list uchar uint | always (triangles) |

`-o -` writes PLY to standard output when `--output-format ply` is given.

## MSMS `.vert` / `.face` (`--output-format msms`, or `-o name.vert`)

Two files next to each other (`name.vert`, `name.face`), as written by
M. Sanner's MSMS:

```
# MSMS solvent excluded surface vertices for 1crn.cif
#vertices #sphere density probe_r
  13812     327  6.00  1.40
    4.500     3.500    -7.104    -0.321    -0.168    -0.932       1     131  3
```

* `.vert` rows: position, normal (`%9.3f` each), face number (here the
  1-based component), closest sphere (1-based selected atom), type.
* `.face` rows: three 1-based vertex indices (`%6d`), type (`%2d`), face
  number (`%6d`).
* Types follow the MSMS manual: 1 toric reentrant (saddle), 2 spheric
  reentrant (concave), 3 contact (convex); a triangle takes its vertices'
  common type, else 1.
* The header density is vertices per Å² of the written mesh.

## Batch mode

Several inputs (`sesmesh a.cif b.pdb ...`) write `<stem>.<ext>` into
`--output-dir` (default: the current directory). `--stats`, `--atom-areas`
and `--vertex-data` then name directories and get `<stem>.stats.json`,
`<stem>.atoms.csv` and `<stem>.vertices.csv`. `--keep-going` continues after
a failing input; the exit code is the first failure's.

## Per-vertex attributes (`--vertex-data <csv>`)

One row per OBJ vertex, in OBJ order:

| Column | Meaning |
|---|---|
| `vertex` | 1-based OBJ vertex index |
| `x`, `y`, `z` | position (4 decimals) |
| `atom` | 0-based index of the nearest atom among the selected atoms |
| `serial`, `chain`, `res_name`, `res_seq`, `ins_code`, `atom_name`, `element` | that atom's identification |
| `patch` | `convex`, `saddle` or `concave` |
| `component` | component index |

The nearest atom follows MSMS semantics: among the atoms generating the
vertex's patch, the one minimising `|v − c| − r`.

## Per-atom areas (`--atom-areas <csv>`)

One row per selected atom, in selection order: `atom` (0-based), the same
identification columns, `radius` (Å) and `area` (Å², 4 decimals). Each
triangle's area is split equally among its three vertices' nearest atoms; the
column sums to the total area of the written mesh.

## Statistics (`--stats <json>`, `-` for standard output)

```json
{
  "sesmesh": "0.1.0",
  "input": {"path": "1crn.cif", "format": "mmcif", "model": 1,
            "atoms_selected": 327, "atoms_in_model": 327},
  "parameters": {"surface": "ses", "probe_radius": 1.4, "grid_spacing": 0.25,
                 "radii": "bondi", "refine": "exact", "cavities": "keep",
                 "min_cavity_volume": 0.0, "min_component_area": 0.0,
                 "normals": "analytic", "brick_size": 10, "simd": "auto"},
  "mesh": {"vertices": 55542, "triangles": 111076, "area": 2314.2,
           "volume": 4662.7, "euler": 4, "components": 2, "cavities": 1,
           "removed_components": 0},
  "components": [{"index": 0, "cavity": false, "vertices": 54331,
                  "triangles": 108654, "first_triangle": 0, "area": 2269.4,
                  "volume": 4687.6, "euler": 2, "genus": 0}, "..."],
  "engine": {"kernel": "neon", "touched_bricks": 1100, "surface_bricks": 900,
             "refine_evaluations": 120000},
  "timings_ms": {"read": 1.0, "geometry": 2.8, "classify": 1.9, "mesh": 44.4,
                 "assemble": 1.0, "measure": 6.3, "write": 4.6, "total": 62.2},
  "warnings": []
}
```

* `volume` is signed: outer components count positive, cavities negative; the
  mesh total is their sum.
* `genus` is `(2 − euler) / 2` per closed component (tunnels through the
  molecule raise it).
* Keys are sorted alphabetically in the file; values above are illustrative.

## Cavity handling (`--cavities`)

* `keep` (default): every component, including internal cavities (the exact
  SES).
* `remove`: drops cavities and every component inside a removed cavity
  (MSMS-like output).
* `only`: keeps the cavities only.
* `--min-cavity-volume V` drops cavities with |volume| < V; `--min-component-area A`
  drops components with area < A. Anything inside a dropped cavity is dropped
  with it. The OBJ header and statistics report the number of removed
  components.
