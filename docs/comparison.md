# sesmesh compared with other SES programs

Status: measured 2026-11-14 with sesmesh 1.1.0 (GitHub issues #5, #6; also
the x86 measurements of #8 and #10). Harness, container and raw numbers:
[`tools/compare/`](../tools/compare/README.md),
[`docs/validation/compare/`](validation/compare/), stored references in
[`tests/data/reference/`](../tests/data/reference/). Section 7 is generated
by `tools/compare/report.py`; the text around it is written by hand.

## 1. Programs

| Program | Version used | Method | Licence | Built from |
|---|---|---|---|---|
| **sesmesh** | 1.1.0 | Exact implicit SES field (Quan–Stamm) sampled on a sparse narrow-band grid, Marching Cubes, exact vertex refinement | MIT or Apache-2.0 | this repository |
| **MSMS** | 2.6.2 (2026-11-06) | Analytic reduced surface, template triangulation of the patches | Sannerlab non-commercial source-available | static Linux binary |
| **NanoShaper** | 0.7.5 | Analytic patches (CGAL alpha shapes), ray casting along grid lines, Marching Cubes tables | GPL-3 | source (github.com/shi-scala/NanoShaper @ 107d844), CGAL 5.6, Boost 1.83 |
| **ESES** | git 0b4cd51 (2020) | Exact SES intersections with a Cartesian grid, Marching Cubes "Lagrangian" mesh | none stated | source (github.com/WeilabMSU/ESES), OpenMP |
| **EDTSurf** | 0.2009 | Euclidean distance transform of the voxelized SAS, Marching Cubes | permissive ("as is") | Debian/Ubuntu package 0.2009-10 |
| **UCSF ChimeraX** | 1.12 (2026-06-12) | Distance grid, contouring (`surface` command) | non-commercial (licence accepted at download) | Ubuntu 24.04 package |
| **sesa** | V001 | Analytic SES area only (no mesh) | GPL-3 | Linux binary from github.com/wlincong/SESexact |

sesx (same author as sesa, exact triangulation) is an interactive Qt viewer
without a documented way to export its mesh, so it is not measured.

## 2. Protocol

* **Same spheres.** `meshcheck spheres` writes the atoms and radii sesmesh
  uses by default (first model, no hydrogens or waters, highest-occupancy
  alternate locations, Bondi radii) as `xyzr` (MSMS, NanoShaper), a
  whitespace PQR (sesmesh), a fixed-column PQR (ESES) and a PDB file
  (EDTSurf, sesa, ChimeraX). Coordinates keep three decimals, as in the PDB
  archive, and every atom is written as an `ATOM` record, because ESES,
  EDTSurf and ChimeraX skip or treat `HETATM` ligands differently. ChimeraX
  reads no PQR: the script opens the PDB and sets every atom radius from the
  PQR. **EDTSurf and sesa cannot take radii from the user** (EDTSurf assigns
  its own by atom type; sesa uses CHARMM parameters and adds hydrogens), so
  their surfaces are of a different molecule model and only their robustness,
  topology and speed compare directly. Probe radius 1.4 Å everywhere.
* **Matched resolution.** Grid spacing h for sesmesh, ESES and ChimeraX;
  grid scale 1/h for NanoShaper and EDTSurf; vertex density 1.5/h² for MSMS
  (sesmesh's measured vertices·h²/area = 1.5).
* **Settings.** MSMS `-all_components` (cavities kept, as in sesmesh's
  default); NanoShaper `Surface = ses`, `Accurate_Triangulation = true`,
  `Smooth_Mesh = false` (its smoothing moves vertices off the surface);
  EDTSurf `-s 3 -h 1` (molecular surface, inner and outer); ChimeraX
  `surface #1 enclose #1 include #1` (one surface around all atoms, ligands
  included). Exact command lines: `tools/compare/run.py`.
* **Measurements.** One checker, `tools/compare/meshcheck`, reads every
  program's output (OBJ, OFF, PLY, MSMS `.vert/.face` including the
  per-component files) and reports topology (boundary, non-manifold and
  inconsistently oriented edges, coincident vertices, degenerate and
  duplicate triangles, components), area, signed volume, per-component
  area/volume/centroid, and the distance of every vertex to the **exact SES
  of the same spheres** (|F| from sesmesh's `SesField`, which agrees with a
  brute-force oracle) together with the angle between each vertex normal and
  the analytic normal. Wall time, CPU time and peak RSS come from GNU
  `time`.
* **Machines.** Two identical AMD Ryzen AI Max+ 395 systems (16 Zen 5 cores,
  SMT off, `performance` governor, 128 GB), CachyOS, all programs in one
  Ubuntu 24.04 container (`tools/compare/Containerfile`). Accuracy and
  robustness runs shared a host (several programs at once, each pinned to
  its own cores, memory caps recorded per run); performance runs had a host
  to themselves.
* **References.** sesmesh's h → 0 extrapolation of each corpus entry
  (`docs/validation/accuracy.csv`), MSMS's analytic area, and — the decisive
  check — NanoShaper and ESES, which also put their vertices on the exact
  surface but construct it independently.

## 3. Features (from each program's documentation)

| | sesmesh | MSMS | NanoShaper | ESES | EDTSurf | ChimeraX | sesa |
|---|---|---|---|---|---|---|---|
| Output | mesh | mesh | mesh | mesh | mesh | mesh | areas |
| Input formats | PDB, mmCIF, XYZ/extXYZ, PQR (+gzip), assemblies | xyzr | xyzr (pdb2xyzr script) | PQR | PDB | PDB, mmCIF and many more | PDB |
| User radii | yes | yes | yes | yes | no | yes (attribute) | no |
| Surfaces | SES, SAS, vdW | SES | SES, skin, blobby | SES | vdW, SAS, MS/SES | SES (also Gaussian) | SES and SAS areas |
| Resolution control | spacing, presets, vertex density | vertex density | grid scale | spacing | scale factor | grid spacing | – |
| Mesh formats | OBJ, binary PLY, MSMS (+gzip) | MSMS `.vert/.face` | OFF, OFF+N(+A), MSMS | OBJ | PLY | OBJ, STL, glTF, … | – |
| Vertex normals | analytic | analytic | analytic where possible, else averaged | none in the OBJ | none | from the grid | – |
| Per-vertex atom / patch | both | both | atom (OFF+A) | patch (intersection file) | – | – | – |
| Cavities | keep / remove / only, filters | `-all_components`, `-one_cavity` | flood-fill detection and filling | yes | `-h` inner/outer | yes | inner-surface areas |
| Per-atom areas | yes | yes (`-af`) | – | – | ASA file | not checked | yes |
| Level of detail | decimation, smoothing with exact re-projection, memory guard | density | smoothing | – | – | `surface dust` etc. | – |
| Parallel | yes (rayon) + SIMD | no | yes (Boost threads) | yes (OpenMP) | no | `nthread` option | not documented |
| Library API | Rust | not in the release | C++, Python (SWIG), DelPhi | – | – | Python | – |
| Platforms | Linux, macOS, Windows | Linux, macOS, Windows | Linux, Windows (binaries); macOS from source | Linux, macOS | Linux (source, Debian package) | Linux, macOS, Windows | Linux, Windows |

## 4. Accuracy

NanoShaper and ESES build the surface independently and also put their
vertices on the exact SES, so they are the decisive references. On the same
spheres and at the same grid spacing they agree with sesmesh on every corpus
entry they meshed (all 53 at 0.5 Å; 52 at 0.25 Å, where both failed on
4V6X), to **≤ 0.026 % (NanoShaper) and ≤ 0.016 % (ESES) in area and
≤ 0.006 % in volume at h = 0.25 Å**, ≤ 0.10 % / 0.07 % in area and
≤ 0.035 % in volume at 0.5 Å. All three lie below the h → 0 extrapolation of
`docs/validation/accuracy.csv` by the same O(h²) amount (inscribed meshes),
which confirms that extrapolation and the specification's protein bounds
(NFR-ACC) with external references.

| Program | area at 0.5 Å (median, worst) | area at 0.25 Å (median, worst) | volume at 0.25 Å (median) |
|---|---|---|---|
| sesmesh | −0.83 %, −0.93 % | −0.20 %, −0.23 % | −0.01 % |
| NanoShaper | −0.78 %, −0.86 % | −0.19 %, −0.21 % | −0.01 % |
| ESES | −0.80 %, −0.91 % | −0.20 %, −0.22 % | −0.01 % |
| MSMS (mesh) | −0.03 %, +1.82 % | +0.71 %, +2.67 % | −0.05 % |
| MSMS (analytic area) | +0.91 % (−0.33 … +2.79 %) | same | – |
| ChimeraX | −8.97 %, −12.75 % | −4.55 %, −6.56 % | +1.92 % |

(relative to sesmesh's h → 0 extrapolation; EDTSurf and sesa use other
radii and are not comparable. Exact figures: `validation/compare/summary.md`.)

## 5. Vertices, normals, mesh validity, cavities

* **sesmesh, NanoShaper, ESES**: closed, consistently oriented,
  edge-manifold meshes in every run that produced one; median 99th-percentile
  vertex distance to the exact SES 9·10⁻⁶, 8·10⁻⁴ and 4·10⁻⁵ Å. NanoShaper
  places a few vertices far from the surface on some inputs (up to 1.4 Å on
  8 of 200 sweep entries). ESES writes no normals.
* **MSMS**: vertices more than 0.1 Å off the SES in most meshes (up to
  3.4 Å), non-manifold edges in 31 of 104 corpus meshes (mostly at density
  24) and 18 of 195 sweep meshes. Example: on 5PTI at density 24
  (`meshcheck check` on MSMS's mesh, then `meshcheck probe` at the worst
  vertex) the vertex at (33.503, 5.587, −0.084) Å has F = +1.387 Å, its
  closest point of the probe-accessible region being (33.512, 5.594,
  −0.079) Å: the vertex lies in the solvent, 0.013 Å from a feasible probe
  centre. That is what concave patches of overlapping probe positions look
  like when they are not clipped against each other, and it would also make
  an analytic area too large; we did not trace MSMS's code to confirm the
  cause. The MSMS manual documents restarts that enlarge the radii of the
  atoms involved by 0.1 Å.
* **ChimeraX**: each patch has its own vertices, so no mesh is closed as
  written (nearly all are after merging coincident vertices); median
  99th-percentile vertex error 0.6 Å at 0.5 Å.
* **EDTSurf**: closed meshes; its own radii, so distances are not
  comparable.
* **sesmesh 1.1.0 defect found here**: on edges where F is not monotone
  (two concave patches meeting at a kink), the root finder could stop at an
  unevaluated point, up to 0.22 Å off (one corpus entry at 0.5 Å, three
  sweep entries, ≤ 0.011 % of the vertices of a mesh). Fixed after 1.1.0 by
  bisecting such edges (branch `claude/intelligent-pasteur-0b9022`).

**Cavities (issue #6).** 2PTN has 14 cavities in sesmesh, NanoShaper and
ESES at 0.5 and 0.25 Å, with the same total volume to 0.1 Å³ (707.8 Å³ at
0.25 Å); MSMS reports 16 and ChimeraX 11. Over the corpus, cavities that
NanoShaper or ESES find but sesmesh does not match have their vertices on the
exact SES: they are the same cavity space split or joined differently at
necks thinner than the grid, and their number falls from 55 / 263 at 0.5 Å
to 10 / 65 at 0.25 Å (NanoShaper / ESES). MSMS reports about twice as many cavities as the three
exact-vertex programs; half of its unmatched cavities have vertices ≥ 0.5 Å
off the SES. The C₆₀ fullerene has one internal cavity in sesmesh,
NanoShaper, ESES and ChimeraX and none in MSMS (as reported by Decherchi and
Rocchia, 2013).

## 6. Robustness and determinism

* sesmesh: every input meshed (corpus, sweep, synthetic, benchmark set).
* MSMS: failed on 5 of 200 sweep entries, on 1FFK at both densities and on
  3J3Q ("ERROR while computing the reduced surface" after its restarts);
  47 GiB and 21 minutes for the 1K4R capsid at 1 Å.
* NanoShaper: segmentation fault on 4V6X at 4 grid points/Å (also with
  five-fold acceleration-grid limits); error exit without a message on 3J3Q
  at 1 Å and on the one- and two-atom synthetic inputs.
* ESES: segmentation fault on coordinates 10⁴ Å from the origin; vertices
  1.3 Å off on exact duplicate atoms; 21 GiB for 1AON and 29 GiB for 1JJ2 at
  0.25 Å (4 threads), stopped at the 40 GiB cap on 4V6X at 0.25 Å, 42 GiB for
  3J3Q at 1 Å.
* ChimeraX: failed on the far-offset input.
* EDTSurf: aborted on both capsids (1K4R assembly, 3J3Q) at 1 Å.
* sesa: crashed on 3J3Q.
* Byte-identical output (comment lines ignored) for 1–16 threads and repeats
  on 4HHB and 1AON: sesmesh, NanoShaper, ChimeraX; MSMS and EDTSurf across
  repeats (single-threaded). ESES's 1AON mesh differed at every thread count.

## 7. Performance and x86 measurements

Complete runs on one idle 16-core Zen 5 host, multi-threaded programs with
16 threads (MSMS, EDTSurf and sesa are single-threaded); wall time in
seconds (peak memory in GiB), median of three runs where repeated. Full
table: [`validation/compare/summary.md`](validation/compare/summary.md)
("Performance"), data `validation/compare/performance.csv`.

| Entry | h (Å) | sesmesh | NanoShaper | ESES | MSMS | ChimeraX | EDTSurf |
|---|---|---|---|---|---|---|---|
| 1CRN | 0.25 | 0.02 (0.02) | 0.18 (0.10) | 6.3 (0.73) | 0.09 (0.01) | 1.5 (0.22) | 0.20 (0.05) |
| 4HHB | 0.25 | 0.25 (0.14) | 2.3 (0.42) | 36 (7.3) | 2.2 (0.06) | 7.5 (0.46) | 2.6 (0.36) |
| 1AON | 0.25 | 3.2 (1.3) | 42 (7.4) | 356 (21) | 66 (1.4) | 91 (3.7) | 2.8 (0.33) |
| 1JJ2 | 0.5 | 1.9 (0.64) | 11 (2.0) | 100 (16) | 34 (1.4) | 32 (1.4) | 3.5 (0.45) |
| 4V6X | 0.5 | 5.5 (1.8) | 41 (6.9) | 280 (26) | 507 (8.8) | 95 (3.7) | 6.0 (0.55) |
| 3J3Q | 1.0 | 28 (4.5) | fail | 770 (42) | fail | 369 (15) | fail |
| 1K4R asm. 1 | 1.0 | 5.6 (1.2) | 29 (4.8) | 153 (18) | 1292 (47) | 76 (3.4) | fail |

sesmesh is the fastest program on every entry at matched resolution:
NanoShaper takes 4–13× as long (median 7×), ChimeraX 13–74×, ESES 27–316×,
single-threaded MSMS 3–232×. EDTSurf is faster on 1AON at 0.25 Å but does
not reach the requested resolution on large molecules (0.9 M triangles
there, against 17.5 M for the other programs).

**x86-64 (issues #8, #10).** The same host is the homogeneous x86-64
platform the Apple Silicon measurements lacked:

* Thread scaling (complete runs, 1AON and 1JJ2 at 0.5 Å): 11.3× and 11.6×
  on 16 threads (71 % and 72 % parallel efficiency), 79 % at 8 threads,
  above the specification's 75 % target (#10).
* SIMD (#8): the explicit kernels shorten the meshing stage by 9–13 %
  against the auto-vectorized portable fallback (4HHB and 1AON at 0.25 Å),
  and the criterion field benchmark (arrangement, classification, kernel;
  `validation/compare/bench-field-zen5.txt`) gives 1.17–1.20× for AVX2 and
  1.18–1.21× for AVX-512: far below the 3× target, because the compiler
  vectorizes the fallback well and the validity tests, gathering and
  double-precision refinement do not use the explicit kernels (#13).

## 8. Reproduce

See [`tools/compare/README.md`](../tools/compare/README.md). Raw JSONL runs
are not committed (≈ 30 MB with per-component lists); `report.py` writes
everything in `docs/validation/compare/` and `tests/data/reference/` from
them.
