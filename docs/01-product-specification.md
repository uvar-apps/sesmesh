# sesmesh — Product Specification

| | |
|---|---|
| Document | 01 — Product Specification |
| Status | Draft v0.1 (for review) |
| Date | 2026-11-12 |
| Related | [02 — Design](02-design.md), [03 — Implementation Plan](03-implementation-plan.md) |

> Working name: **sesmesh** (library crate `sesmesh`, CLI crate `sesmesh-cli`, binary `sesmesh`).
> The name is free on crates.io as of the date above; see [Open questions](#10-open-questions-and-proposed-defaults).

---

## 1. Purpose

sesmesh is a reusable Rust library plus a command-line tool that computes the
**Solvent Excluded Surface (SES)** — also called the *Connolly surface*,
*molecular surface* or *Lee–Richards/Richards smooth molecular surface* — of a
molecule read from **PDB**, **PDBx/mmCIF** or **XYZ** files, and writes it as
a **triangle mesh with per-vertex normals** in **Wavefront OBJ** format.

The product must:

1. be fast on everything from small ligands to multi-million-atom assemblies,
   using **multi-core parallelism** and **SIMD vector instructions**;
2. give the user explicit control over **resolution / level of detail**;
3. produce meshes that are **watertight, consistently oriented and
   geometrically accurate** (vertices on the true SES, exact normals), so they
   are useful for visualization, 3D printing, and numerical work (area/volume,
   boundary-element and finite-element solvers);
4. expose **every feature through the library API**, so other Rust projects can
   embed it without the CLI; the CLI is a thin layer over the library.

---

## 2. Background

### 2.1 Molecular surface definitions

A molecule is modelled as a set of spheres: atom *i* has centre **c**ᵢ and
van der Waals (vdW) radius *r*ᵢ. Solvent is modelled as a spherical probe of
radius *r*ₚ (1.4 Å for water by convention).

| Surface | Definition | Origin |
|---|---|---|
| **vdW surface** | Boundary of the union of atom balls B(**c**ᵢ, *r*ᵢ). | — |
| **SAS** (solvent accessible surface) | Locus of the probe *centre* rolled over the vdW surface = boundary of the union of balls B(**c**ᵢ, *r*ᵢ + *r*ₚ). | Lee & Richards 1971 [R1] |
| **SES** (solvent excluded / molecular / Connolly surface) | Locus of the probe *surface* (inward-facing) as it rolls over the vdW surface; equivalently, the boundary of the union of all probe balls that do not intersect any atom ball. | Richards 1977 [R2]; Greer & Bush 1978 [R3]; Connolly 1983 [R4, R5] |

The SES is made of three kinds of analytic patches [R4]:

* **convex (contact) spherical patches** — the parts of atom spheres the probe
  can touch;
* **saddle (toroidal / reentrant) patches** — swept by the probe while touching
  two atoms;
* **concave (reentrant) spherical triangles** — probe positions touching three
  atoms.

The SES has **singularities**: self-intersecting tori (when the probe cannot
pass between two atoms) and intersecting concave patches (when probe positions
overlap). Handling these robustly is the historical difficulty of analytic
methods [R6, R9]. Quan & Stamm [R6] proved that the SES is exactly a level set
of the *signed distance function to the SAS* and that this distance is
computable from three cases (closest point on a sphere, on an intersection
circle arc, or at a three-sphere intersection point). This implicit
characterization handles all singularities automatically and is the
mathematical foundation of sesmesh (see Design §2).

Internal **cavities** — regions where a probe fits but cannot reach from the
outside — produce additional closed SES components. Tools differ in whether
they report them (MSMS reports only the external component by default [R9];
NanoShaper and EDTSurf detect and report cavities [R12, R13]).

### 2.2 Algorithm families (survey)

| Family | Representative work | How it works | Strengths | Weaknesses |
|---|---|---|---|---|
| Analytic patch construction + template triangulation | Connolly MSP [R4]; **MSMS** reduced surface [R9]; contour-buildup [R10, R11]; alpha shapes (NanoShaper's build-up) [R13]; sesa [R21] | Compute all probe placements and patch boundaries exactly, then triangulate each patch. | Exact geometry; compact, well-shaped meshes; analytic area. | Numerically fragile in singular/degenerate configurations; MSMS restarts with perturbed radii and is reported to fail on many structures > 10 000 atoms [R19]; complex to implement. |
| Analytic surface + grid ray casting | **NanoShaper** [R13] | Build analytic patches (CGAL alpha shapes), cast rays along grid lines, compute exact ray–patch intersections, triangulate with Marching Cubes tables ("analytical intersections MC"). | Vertices lie exactly on the surface; robust (exact predicates + parity checksum); multithreaded ray casting. | Depends on CGAL; analytic build-up still needed; heavy. |
| Volumetric / Euclidean distance transform | **EDTSurf** [R12]; LSMS level sets [R18]; ESES [R17]; Chimera distance grid [R19] | Voxelize the SAS, compute (approximate) distance from the SAS boundary, threshold at *r*ₚ, extract an isosurface. | Simple, robust, any size, resolution is a direct knob. | Accuracy tied to voxel size: EDTSurf reports 3.96 % area / 1.18 % volume error at 128³ and 1.99 % / 0.48 % at 256³ vs. MSMS analytic values [R12]; dense grids are memory hungry; normals approximate. |
| Narrow-band signed distance grids (GPU) | **Hermosilla et al.** [R15]; **QuickSES** [R16] | Classify grid points (outside / inside / boundary), compute distances only in a narrow band, organize points in 8³ bricks, refine progressively. | Highly parallel; interactive for 545 k atoms (1K4R, 512³ grid, ≈1.65 s on a GTX 970) [R15]. | Distances are grid-to-grid approximations; GPU-only; large memory (1.9 GB for 22 k atoms in [R15]). |
| Implicit analytic SES | Parulek & Viola [R14]; Quan & Stamm [R6, R7] | Evaluate an exact (or exact-near-surface) distance function to the SES. | Exact, handles singularities, embarrassingly parallel per sample. | Naive evaluation is expensive; needs good spatial culling. |

**Lessons taken into the product:**

* Grid-based pipelines are the robust, scalable choice and make resolution an
  explicit, intuitive knob (ChimeraX exposes `gridSpacing`, default 0.5 Å, as
  essentially its only quality setting [R20]; Chimera's study found a 0.5 Å
  distance grid visually comparable to MSMS density 2 [R19]).
* Accuracy of grid methods is limited by *approximate distances*, not by the
  grid itself. Using the **exact implicit SES distance** [R6] at grid samples and
  **exact edge crossings** (as NanoShaper does with ray casting [R13]) removes
  most of the discretization error in vertex positions and normals.
* Work must be restricted to a **narrow band** around the surface and organized
  in **bricks** (as in [R15, R30]) for memory efficiency and cache-friendly
  parallelism.

### 2.3 Positioning

sesmesh combines (a) the exact implicit SES characterization [R6], (b) sparse
narrow-band brick grids with Lipschitz-bound culling [R15, R30, R31], (c)
Marching Cubes with exact edge intersections and analytic normals
[R22, R13], and (d) CPU SIMD + multi-threading. Target: MSMS-class geometric
accuracy with EDTSurf/ChimeraX-class robustness and better speed than both, in
a dependency-light, embeddable Rust crate.

As of this writing no Rust crate computes an SES mesh; existing Rust crates in
the area compute SASA numerically (e.g. `rust-sasa`, `powersasa`) [R54].

---

## 3. Product scope

### 3.1 Goals

| ID | Goal |
|---|---|
| G1 | Compute the SES (and, cheaply, the SAS and vdW surfaces) of molecules of any size as a closed, oriented triangle mesh with normals. |
| G2 | Read PDB, PDBx/mmCIF and XYZ (incl. extended XYZ) — optionally gzip-compressed — with sensible atom selection and radius assignment. |
| G3 | Let the user control resolution / level of detail (grid spacing, presets, vertex density, decimation targets). |
| G4 | Exploit all cores (work-stealing parallelism) and SIMD (AVX2/AVX-512 on x86-64, NEON on AArch64) with runtime CPU dispatch and a portable scalar fallback. |
| G5 | Provide a stable, documented Rust library API usable without any file I/O (spheres in → mesh out). |
| G6 | Provide a CLI exposing every library feature. |
| G7 | Report quantitative results: area, enclosed volume, components, cavities, timings. |
| G8 | Be deterministic: identical input and options produce identical output regardless of thread count. |

### 3.2 Non-goals (v1)

* GPU computation (may be revisited; the design keeps the field kernel
  portable).
* Real-time rendering / a viewer.
* Gaussian, skin, ligand-excluded or minimal molecular surfaces (the design
  leaves room for additional implicit surface kinds).
* Electrostatics, curvature descriptors, docking.
* Analytic (MSMS-style) patch-by-patch triangulation.
* Reading MD trajectory formats (DCD, XTC); the library API supports repeated
  computation on new coordinates, so callers can feed frames themselves.
* Writing structure files.

### 3.3 Users and use cases

| ID | Actor | Use case |
|---|---|---|
| UC1 | Structural biologist | `sesmesh 1abc.cif -o 1abc.obj` and open the result in Blender, ChimeraX, MeshLab or a web viewer. |
| UC2 | 3D-printing / outreach | Produce a watertight, moderately decimated mesh of a large complex (e.g. a capsid). |
| UC3 | Computational chemist | Compute SES area/volume of many structures in batch scripts; JSON stats output. |
| UC4 | Method developer (Rust) | Embed the library: pass sphere centres and radii, receive vertices/normals/triangles plus per-vertex atom indices for colouring. |
| UC5 | Numerical solver (BEM/FEM) | High-resolution, manifold mesh with vertices exactly on the SES and accurate normals; cavities included or excluded on demand. |
| UC6 | MD analysis tool (Rust) | Recompute the surface for each frame, reusing allocations. |
| UC7 | Materials / small molecules | XYZ input with element radii; very small probe radii; SAS and vdW variants. |

---

## 4. Functional requirements

Priority uses MoSCoW: **M**ust, **S**hould, **C**ould, **W**on't (this release).

### 4.1 Input (FR-IN)

| ID | Requirement | Pri |
|---|---|---|
| FR-IN-1 | Read PDB files: `ATOM`/`HETATM` fixed-column records per wwPDB format v3.3 [R38]; `MODEL`/`ENDMDL`; `TER`; element symbol from columns 77–78 with fallback inference from the atom name; alternate location (col 17), occupancy, B-factor, insertion code; tolerate missing trailing columns, CRLF line endings, and hybrid-36 serial numbers [R41]. | M |
| FR-IN-2 | Read PDBx/mmCIF files [R39]: full CIF tokenizer (quoted strings, semicolon text fields, `?`/`.` nulls, multiple data blocks — first by default); `_atom_site` loop with arbitrary column order; `label_*` and `auth_*` identifiers; `pdbx_PDB_model_num`. Required for structures that have no PDB-format file (> 99 999 atoms, > 62 chains, long chain IDs, extended PDB IDs) [R40]. | M |
| FR-IN-3 | Read XYZ files: atom count line, comment line, `Element x y z` lines (element symbol or atomic number), extra columns ignored; multi-frame files with frame selection. | M |
| FR-IN-4 | Read extended XYZ (`Properties=species:S:1:pos:R:3...` in the comment line) [R43]; honour a per-atom radius column when the user names it. | S |
| FR-IN-5 | Read PQR files (whitespace-delimited, charge and radius columns) [R44]; use file radii by default. | S |
| FR-IN-6 | Transparent gzip decompression (`.gz`); read from stdin (`-`) with an explicit format. | M |
| FR-IN-7 | Automatic format detection from extension and content sniffing, overridable by option. | M |
| FR-IN-8 | Generate biological assemblies from mmCIF `pdbx_struct_assembly_gen` / `pdbx_struct_oper_list` (and PDB `REMARK 350 BIOMT`). Needed for virus capsids whose files contain only the asymmetric unit (e.g. 1K4R: 9 084 deposited atoms vs ≈ 545 k in the capsid [R15]). | S |
| FR-IN-9 | BinaryCIF input [R42]. | C |
| FR-IN-10 | Clear errors with file name, line number and offending text; never panic on malformed input. | M |

### 4.2 Atom selection (FR-SEL)

| ID | Requirement | Pri |
|---|---|---|
| FR-SEL-1 | Model selection: first model by default; specific model number; (C) all models as separate surfaces. | M |
| FR-SEL-2 | Alternate locations: default keep the highest-occupancy conformer per atom site (ties → first); options: a specific altloc letter, or all. | M |
| FR-SEL-3 | Hydrogens: excluded by default (united-atom convention, as in EDTSurf [R12]); option to include. | M |
| FR-SEL-4 | Waters (HOH, WAT, DOD, H2O, SOL, TIP3): excluded by default; option to include. | M |
| FR-SEL-5 | Option to exclude all HETATM / non-polymer atoms; option to select chains (by `auth_asym_id` by default, `label_asym_id` optionally) and to exclude residue names. | M |
| FR-SEL-6 | Library API accepts an arbitrary user predicate over atom records. | S |

### 4.3 Radii (FR-RAD)

| ID | Requirement | Pri |
|---|---|---|
| FR-RAD-1 | Element radius tables: **Bondi** 1964 [R34] completed with **Mantina et al.** 2009 for the remaining main-group elements [R35] (default); **Alvarez** 2013 [R36]. | M |
| FR-RAD-2 | Residue/atom-type united-atom radii **ProtOr** (Tsai et al. 1999 [R37]) for standard amino acids and nucleotides, falling back to element radii for other atoms. | S |
| FR-RAD-3 | Custom radius file (simple text/TOML: rules keyed by element, or residue name + atom name with wildcards) overriding or extending a built-in table. | M |
| FR-RAD-4 | Uniform radius option (all atoms same radius). | S |
| FR-RAD-5 | Per-file radii (PQR, extended XYZ column) when present. | S |
| FR-RAD-6 | Unknown elements: configurable fallback radius (default 1.80 Å) with a warning listing affected atoms; option to make it an error. | M |
| FR-RAD-7 | Global radius scale factor and additive offset (useful for sensitivity studies). | C |

### 4.4 Surface computation (FR-SURF)

| ID | Requirement | Pri |
|---|---|---|
| FR-SURF-1 | Compute the SES for probe radius *r*ₚ > 0 (default 1.4 Å). *r*ₚ = 0 yields the vdW surface. | M |
| FR-SURF-2 | Compute the SAS and the vdW surface with the same pipeline (`--surface ses|sas|vdw`). | M |
| FR-SURF-3 | Exact treatment of all SES singularities (self-intersecting tori, intersecting reentrant patches) — no radius perturbation restarts. | M |
| FR-SURF-4 | Cavity handling: `keep` (all components), `remove` (external components only), `only` (cavities only); minimum cavity volume / area filters. | M |
| FR-SURF-5 | Multiple disconnected molecules in one input produce multiple external components. | M |
| FR-SURF-6 | Robust to degenerate input: coincident atoms, symmetric/lattice configurations (e.g. fullerene, which breaks MSMS [R13]), collinear/coplanar atoms, one sphere inside another, very large or very small probes. | M |
| FR-SURF-7 | Per-vertex attributes: nearest atom index (input order), patch type (convex / saddle / concave). | M |
| FR-SURF-8 | Restrict the computed surface to a region (axis-aligned box or within distance *d* of a subset of atoms), while still taking all atoms into account for the geometry (e.g. "surface of the binding pocket"). | C |

### 4.5 Resolution and level of detail (FR-LOD)

| ID | Requirement | Pri |
|---|---|---|
| FR-LOD-1 | Grid spacing *h* in Å as the primary control (`--grid-spacing`, valid range 0.05–4 Å). | M |
| FR-LOD-2 | Quality presets: `draft` (1.0 Å), `low` (0.7 Å), `medium` (0.5 Å, default), `high` (0.25 Å), `ultra` (0.1 Å). | M |
| FR-LOD-3 | Target vertex density in vertices/Å² (MSMS-like `-density` semantics [R9]), converted to a grid spacing using a calibrated relation (Design §8.6). | S |
| FR-LOD-4 | Post-extraction decimation to a target triangle count or ratio with a geometric error bound (quadric error metrics [R28]); retained vertices stay on the SES with exact normals. | S |
| FR-LOD-5 | Optional mesh fairing (Taubin λ|μ smoothing [R29]) followed by exact re-projection of vertices onto the SES and normal recomputation. | S |
| FR-LOD-6 | Memory guard: estimate memory before computing; refuse (or, if requested, automatically coarsen) when above a user limit. | S |
| FR-LOD-7 | Adaptive (curvature-driven) resolution. | W |

### 4.6 Mesh properties (FR-MESH)

| ID | Requirement | Pri |
|---|---|---|
| FR-MESH-1 | Output is an indexed triangle mesh with shared vertices (no duplicated vertices at brick seams). | M |
| FR-MESH-2 | Each component is a closed, edge-manifold surface: every edge is shared by exactly two triangles; triangle winding is counter-clockwise seen from outside (solvent side). | M |
| FR-MESH-3 | Vertex normals are unit vectors pointing towards the solvent, computed analytically from the SES geometry (not from triangle averaging). Optional area-weighted face-averaged normals. | M |
| FR-MESH-4 | Vertices lie on the analytic SES within a configurable tolerance (default 10⁻³ Å) when exact edge refinement is enabled (default). | M |
| FR-MESH-5 | No zero-area triangles and no non-finite coordinates. | M |
| FR-MESH-6 | Component labelling with per-component area, signed volume, Euler characteristic and genus. | M |

### 4.7 Output (FR-OUT)

| ID | Requirement | Pri |
|---|---|---|
| FR-OUT-1 | Wavefront OBJ [R45]: `v`, `vn`, `f a//a b//b c//c` (1-based indices), header comments with tool version, input, all parameters and summary statistics; configurable decimal precision (default 4). | M |
| FR-OUT-2 | OBJ options: no normals; one `o`/`g` group per component; non-standard per-vertex colours (`v x y z r g b`) coloured by element, chain, patch type or component. | S |
| FR-OUT-3 | Binary little-endian PLY with optional per-vertex attributes (atom index, patch type, component). | S |
| FR-OUT-4 | MSMS-compatible `.vert`/`.face` output for tools that consume MSMS meshes [R9]. | S |
| FR-OUT-5 | Per-vertex attribute side file (CSV) for OBJ users. | S |
| FR-OUT-6 | Statistics report on stderr (human readable) and optionally as JSON. | M |
| FR-OUT-7 | Per-atom SES area attribution (CSV/JSON). | S |
| FR-OUT-8 | STL, glTF 2.0. | C |
| FR-OUT-9 | Write to stdout (`-o -`); gzip-compressed output when the file name ends with `.gz`. | S |

### 4.8 CLI (FR-CLI)

| ID | Requirement | Pri |
|---|---|---|
| FR-CLI-1 | Single binary `sesmesh` with subcommands `surface` (default when omitted), `info` (summarize an input: atoms, models, chains, bounding box, radii coverage), `radii` (print built-in tables). | M |
| FR-CLI-2 | Every library option is reachable from the CLI. | M |
| FR-CLI-3 | `--threads N` (default: all logical cores); `--simd auto|scalar|<level>` for diagnostics. | M |
| FR-CLI-4 | Progress reporting on TTY (`--progress`), quiet mode, verbosity levels. | S |
| FR-CLI-5 | Batch mode: multiple inputs → output directory, with per-file error isolation. | S |
| FR-CLI-6 | Distinct exit codes: 0 success, 1 runtime error, 2 usage error, 3 input parse error, 4 resource limit exceeded. | M |
| FR-CLI-7 | Shell completions and a man page generated from the CLI definition. | C |

### 4.9 Library (FR-LIB)

| ID | Requirement | Pri |
|---|---|---|
| FR-LIB-1 | Pure-geometry entry point: spheres (centres + radii) + parameters → mesh. No I/O required. | M |
| FR-LIB-2 | Structure readers, selection and radii assignment available as separate, composable functions. | M |
| FR-LIB-3 | Reusable engine object that keeps allocations between calls (trajectory use). | S |
| FR-LIB-4 | Caller-controlled parallelism: runs inside the caller's rayon thread pool if installed; optional single-threaded build (feature flag). | M |
| FR-LIB-5 | Progress callback and cooperative cancellation. | S |
| FR-LIB-6 | No third-party types in the public API (arrays/slices/own types only), so dependency upgrades do not break users. | M |
| FR-LIB-7 | Cargo features to trim dependencies: I/O formats, gzip, decimation, JSON, parallelism. | M |
| FR-LIB-8 | `#![forbid(unsafe_code)]` in all crates; SIMD goes through a safe abstraction crate (Design §11.2). | M |

### 4.10 Diagnostics (FR-DIAG)

| ID | Requirement | Pri |
|---|---|---|
| FR-DIAG-1 | Warnings (unknown elements, skipped records, altloc choices, empty selection) are collected and returned to the caller, not printed by the library. | M |
| FR-DIAG-2 | Optional per-stage timings and counters (atoms, pairs, probe positions, bricks, samples, vertices, triangles, peak memory estimate). | M |
| FR-DIAG-3 | Mesh validation utility (manifoldness, orientation, distance-to-SES check) exposed in the API and as `--validate`. | S |

---

## 5. Non-functional requirements

### 5.1 Performance (NFR-PERF)

Reference machines: (A) Apple Silicon, 8 performance cores, NEON; (B) x86-64
desktop, 8 cores / 16 threads, AVX2. Targets are wall-clock for the full CLI
run (parse → OBJ written) and were **provisional**: confirmed or revised
after the benchmarking phase (Plan, Phase 9).

*Revised for v1.0 (2026-11-13, issue #20):* the targets are the Phase 9
measurements (10-core Apple Silicon, 4 performance + 6 efficiency cores,
NEON, under background load), rounded up. Neither reference machine was
available; measurements on them and the x86 SIMD speed-ups (#8, #10, #13)
follow after 1.0, and the original provisional targets remain goals.

| Structure (PDB ID, atoms used) | Spacing | Target (v1.0) | Original provisional target |
|---|---|---|---|
| 1CRN crambin (327) | 0.25 Å | < 60 ms | < 60 ms |
| 4HHB haemoglobin (≈ 4.4 k heavy atoms) | 0.5 Å / 0.25 Å | < 250 ms / < 600 ms | < 100 ms / < 350 ms |
| 1AON GroEL–GroES (≈ 58 k) | 0.5 Å | < 3.5 s | < 1.5 s |
| 1JJ2 ribosomal subunit (≈ 98 k) | 0.5 Å | < 5 s | < 2.5 s (Chimera's unoptimized distance grid: ≈ 30 s [R19]) |
| 4V6X human 80S ribosome (≈ 238 k) | 0.5 Å | < 15 s | < 6 s |
| 3J3Q HIV-1 capsid (2 440 800) | 1.0 Å | < 75 s | < 60 s |

Additional targets:

* Parallel efficiency ≥ 75 % on 8 cores for ≥ 50 k atoms.
* SIMD field kernel ≥ 3× faster than the scalar kernel on AVX2 and ≥ 2× on
  NEON (lanes = 8 resp. 4 × f32).
* Parsing ≥ 200 MB/s for mmCIF and PDB per core on uncompressed input.
* OBJ writing ≥ 300 MB/s (parallel formatting).

*Phase 9 measurement (10-core Apple Silicon under background load,
`docs/validation-report.md` §6):* 1CRN meets its target; the others are
1.2–2.2× slower than the provisional targets, hence the revision above. The
additional targets for x86 (AVX2 kernel speed-up, 8-core efficiency) are
measured after 1.0 (#8, #10).

### 5.2 Scalability and memory (NFR-MEM)

* Memory proportional to surface area (narrow band), not to bounding-box
  volume.
* Peak memory target: ≤ 150 bytes per output triangle plus ≤ 800 bytes per
  atom (3J3Q at 1.0 Å within 16 GB). *Revised 2026-11-13 (issue #21):* the
  original 200 B per atom cannot hold the exact f64 SAS arrangement (about
  20 neighbour pairs, 2 exposed circles with ≈ 16 blockers each and 1.5
  triple points per atom; the circles alone take ≈ 200 B); 800 B per atom
  matches the measurements and the memory guard's estimate.
* No hard limits on atom count (32-bit indices up to 2³² − 1 atoms/vertices;
  error otherwise).
* *Phase 9 measurement:* 110–150 B per triangle at peak (met); ≈ 600–800 B
  per atom for the SAS arrangement and structure data (above 200 B; visible
  for the 1K4R capsid at 1 Å). 3J3Q at 1.0 Å: 5.3 GB.
* *v1.0 development measurement* (after compaction, validation report §5):
  ≈ 560 B per atom of field data plus ≈ 150 B of input structure and
  spheres; all large inputs within the revised budget (1K4R capsid at 1 Å
  1.74 GB of 1.92 GB, 3J3Q at 1 Å 7.16 GB of 9.25 GB).
* *After 1.0 (memory-guard fix, validation report §5):* peaks lowered
  further (capsid at 1 Å 1.54 GB, decimated 1.97 GB); the budget stays the
  NFR target, while the memory guard (FR-LOD-6) now estimates the whole
  process conservatively (16 MiB + 1 200 B per atom + 150 B per triangle,
  more with decimation, smoothing or a fine grid), not with the budget's
  figures.

### 5.3 Accuracy (NFR-ACC)

* For configurations with closed-form SES area and volume (single sphere; two
  spheres with toroidal saddle, Design §15) with atomic radii (1.2–1.8 Å):
  relative error ≤ 0.7 % for area and ≤ 1.5 % for volume at *h* = 0.25 Å,
  and observed convergence ≈ O(*h*²). *Revised 2026-11-13 (issue #4)* from
  ≤ 0.5 %: the Marching Cubes mesh is inscribed (its vertices lie on the
  surface), so it underestimates both by O(*h*²), most for small, highly
  curved spheres; proteins stay far within their bounds (validation
  report: area −0.20 %, volume −0.02 % at 0.25 Å).
* For proteins, compared with high-density analytic references (MSMS at
  vertex density ≥ 20, NanoShaper at ≥ 4 grids/Å, sesa analytic areas [R21]):
  area within 1.0 % and volume within 0.5 % at *h* = 0.25 Å; within 3 % / 1 %
  at *h* = 0.5 Å. (EDTSurf's published errors at 256³ are 1.99 % / 0.48 % [R12].)
  For v1.0 these bounds are verified against internal references (analytic
  surfaces and the *h* → 0 extrapolation of each entry's mesh, validation
  report §2–3); the cross-tool comparison follows after 1.0 (#5, #6).
* Vertex distance to the analytic SES ≤ 10⁻³ Å (with refinement enabled).
* *Note (Phase 4 measurement):* for single spheres and sphere pairs with
  atomic radii (1.2–1.8 Å) the Marching Cubes mesh, whose vertices lie on the
  surface, underestimates area by 0.4–0.65 % and volume by 0.8–1.35 % at
  *h* = 0.25 Å (O(*h*²), ≈ 0.2 % / 0.3 % at 0.125 Å). Decided after Phase 9
  (#4): the requirement was restated (first bullet), the method kept.
* Normal angular error vs analytic normal ≤ 0.1° away from SES singular curves.

### 5.4 Robustness (NFR-ROB)

* No panics or aborts on any input, including fuzzed files; all failures are
  `Result` errors.
* No silent wrong output: sanity checks (non-finite values, non-manifold
  edges) are reported as errors or warnings.
* Validation runs on the full PDB-derived test corpus (Plan, Phase 9) with
  zero failures.

### 5.5 Determinism (NFR-DET)

* Byte-identical output for identical input, options, binary and CPU SIMD
  level, independent of thread count and scheduling.
* Results from different SIMD levels (or scalar) agree within 10⁻⁵ Å per
  vertex; topology may only differ in exactly degenerate ties.

### 5.6 Portability (NFR-PORT)

* Tier 1: `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`,
  `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`.
* Builds on stable Rust only (no nightly features; `std::simd` is still
  nightly-only as of 2026 [R50]).
* Runtime SIMD dispatch: one binary runs optimally on any CPU of the target
  architecture.
* Library compiles for `wasm32-unknown-unknown` with the `parallel` feature
  disabled (C).

### 5.7 Quality, maintainability, licensing

* Rust edition 2024; MSRV declared in `Cargo.toml` and tested in CI.
* `cargo clippy -D warnings`, `rustfmt`, `cargo deny` (licenses, advisories),
  documentation for every public item (`#![deny(missing_docs)]`).
* Line coverage ≥ 85 % for the library (excluding SIMD duplicates).
* Licence: MIT OR Apache-2.0 (Rust ecosystem convention). No code derived from
  restrictively licensed tools (MSMS's licence is not open source [R9]).
* Test structures from the PDB are CC0 and may be vendored.

---

## 6. Acceptance criteria

### 6.1 MVP (v0.1)

1. `sesmesh input.{pdb,cif,xyz}[.gz] -o out.obj` produces a closed, oriented
   SES mesh with analytic normals for every structure in the small validation
   set (1CRN, 1UBQ, 3FX2, 2PTN, 4HHB, 1BRS).
2. Grid spacing and presets work; SAS and vdW variants work.
3. Analytic two-sphere and single-sphere tests pass the NFR-ACC bounds.
4. Multi-threaded and deterministic; scalar kernel only is acceptable.

### 6.2 v1.0

1. All **Must** and **Should** requirements implemented and tested.
2. NFR-PERF targets met on both reference machines (or revised with written
   justification after Phase 9).
3. NFR-ACC met on the validation corpus; results published in
   `docs/validation-report.md`.
4. Zero failures on the robustness corpus and after ≥ 24 CPU-hours of fuzzing
   per parser.
5. API documentation, examples, CLI manual, changelog; crates published.

*As applied to 1.0.0 (2026-11-13):* 1 met (NFR-MEM per-atom budget revised,
#21); 2 revised with justification (§5.1, #20); 3 met against internal
references (§5.3, analytic bound revised, #4), cross-tool references after
1.0 (#5, #6); 4 robustness sweep 200/200 and a 10-minute run of every fuzz
target (7 targets, ≈ 18 M inputs, no failures), the 24 CPU-hour campaign
per parser after 1.0 (#22); 5 met.

---

## 7. Constraints and assumptions

* Input coordinates are in Å. Output is in Å in the input coordinate frame.
* Radii are ≥ 0; atoms with radius 0 are ignored for SES/SAS (with a warning).
* Single precision (`f32`) is sufficient for output geometry; all
  topology-critical geometric constructions use `f64`.
* The SES is defined by the *selected* atoms only; selection happens before
  surface computation.

---

## 8. Risks (product level)

| Risk | Impact | Mitigation |
|---|---|---|
| Exact field evaluation too slow for very large inputs | Miss NFR-PERF | Lipschitz culling, cheap vdW/SAS bounds for bricks, SIMD, lower-bound pruning; fallback fast mode with linear interpolation only (Design §7, §8). |
| Marching Cubes topological artefacts at coarse spacing (thin necks, near-singular cusps) | Non-manifold or topologically wrong meshes | Consistent table, sign convention for zeros, manifold validation; MC33 option later [R23, R24]. |
| Cavity removal semantics differ from MSMS/NanoShaper in rare touching configurations | User confusion | Document; mesh-component classification by signed volume plus enclosure test (Design §9.2). |
| SIMD crate API churn | Maintenance | Thin internal abstraction; scalar reference kernel always available. |
| Large outputs (10⁷–10⁸ triangles) | Slow I/O, huge files | Parallel formatting, binary PLY, decimation, gzip. |

---

## 9. Out-of-scope clarifications

* "Optimized algorithms exploiting vector instructions" is interpreted as
  explicit SIMD kernels for the hot loops (field evaluation, bounds, MC
  classification) plus autovectorization-friendly data layout elsewhere.
* "Level of detail" is interpreted as user-selected uniform resolution plus
  optional decimation; adaptive/curvature-driven resolution is a future
  extension.

---

## 10. Open questions and proposed defaults

| # | Question | Proposed default |
|---|---|---|
| Q1 | Default cavity handling? MSMS reports only the external surface; the mathematical SES includes cavities. | `keep` (exact SES incl. cavities); `--cavities remove` for MSMS-like output. |
| Q2 | Default radius set? | Bondi + Mantina element radii (works for every format, including XYZ). ProtOr opt-in. |
| Q3 | Hydrogens/waters excluded by default? | Yes (both). |
| Q4 | Default resolution? | `medium` = 0.5 Å (ChimeraX default [R20]). |
| Q5 | Crate/binary name? | `sesmesh`. |
| Q6 | Licence? | MIT OR Apache-2.0. |
| Q7 | Biological assembly generation in v1? | Yes ("Should"), mmCIF first. |
| Q8 | Additional output formats priority after OBJ? | PLY binary, then MSMS vert/face. |

---

## 11. Glossary

| Term | Meaning |
|---|---|
| Probe | Sphere of radius *r*ₚ modelling a solvent molecule. |
| SAS / SES | Solvent accessible / solvent excluded surface (§2.1). |
| Probe-accessible region **P** | Set of possible probe centres: points outside every expanded atom ball. |
| SAS vertex / probe placement | Point where three expanded spheres meet on the SAS; a probe there touches three atoms. |
| Brick | Small cubic block of grid cells (default 8³) processed as one unit of work. |
| Narrow band | Grid region within a few cells of the surface. |
| Lipschitz culling | Skipping a region because a 1-Lipschitz field's value at its centre exceeds its radius. |
| MC | Marching Cubes isosurface extraction [R22]. |
| QEM | Quadric error metric decimation [R28]. |

---

## 12. References

Molecular surface definitions and theory

* **[R1]** B. Lee, F. M. Richards, "The interpretation of protein structures: estimation of static accessibility," *J. Mol. Biol.* 55(3):379–400, 1971. doi:[10.1016/0022-2836(71)90324-X](https://doi.org/10.1016/0022-2836(71)90324-X)
* **[R2]** F. M. Richards, "Areas, volumes, packing, and protein structure," *Annu. Rev. Biophys. Bioeng.* 6:151–176, 1977.
* **[R3]** J. Greer, B. L. Bush, "Macromolecular shape and surface maps by solvent exclusion," *PNAS* 75(1):303–307, 1978. doi:[10.1073/pnas.75.1.303](https://doi.org/10.1073/pnas.75.1.303)
* **[R4]** M. L. Connolly, "Analytical molecular surface calculation," *J. Appl. Cryst.* 16(5):548–558, 1983. doi:[10.1107/S0021889883010985](https://doi.org/10.1107/S0021889883010985)
* **[R5]** M. L. Connolly, "Solvent-accessible surfaces of proteins and nucleic acids," *Science* 221:709–713, 1983.
* **[R6]** C. Quan, B. Stamm, "Mathematical analysis and calculation of molecular surfaces," *J. Comput. Phys.* 322:760–782, 2016. doi:[10.1016/j.jcp.2016.07.007](https://doi.org/10.1016/j.jcp.2016.07.007); preprint [HAL hal-01205580](https://hal.sorbonne-universite.fr/hal-01205580)
* **[R7]** C. Quan, B. Stamm, "Meshing molecular surfaces based on analytical implicit representation," *J. Mol. Graph. Model.* 71:200–210, 2017. doi:[10.1016/j.jmgm.2016.11.008](https://doi.org/10.1016/j.jmgm.2016.11.008)
* **[R8]** B. Kozlíková et al., "Visualization of biomolecular structures: state of the art revisited," *Comput. Graph. Forum* 36(8):178–204, 2017. doi:[10.1111/cgf.13072](https://doi.org/10.1111/cgf.13072)

SES algorithms and tools

* **[R9]** M. F. Sanner, A. J. Olson, J.-C. Spehner, "Reduced surface: an efficient way to compute molecular surfaces," *Biopolymers* 38(3):305–320, 1996. MSMS manual: <https://ccsb.scripps.edu/msms/documentation/>
* **[R10]** M. Totrov, R. Abagyan, "The contour-buildup algorithm to calculate the analytical molecular surface," *J. Struct. Biol.* 116(1):138–143, 1996. doi:[10.1006/jsbi.1996.0022](https://doi.org/10.1006/jsbi.1996.0022)
* **[R11]** N. Lindow, D. Baum, S. Prohaska, H.-C. Hege, "Accelerated visualization of dynamic molecular surfaces," *Comput. Graph. Forum* 29(3):943–952, 2010. doi:[10.1111/j.1467-8659.2009.01693.x](https://doi.org/10.1111/j.1467-8659.2009.01693.x)
* **[R12]** D. Xu, Y. Zhang, "Generating triangulated macromolecular surfaces by Euclidean distance transform (EDTSurf)," *PLoS ONE* 4(12):e8140, 2009. doi:[10.1371/journal.pone.0008140](https://doi.org/10.1371/journal.pone.0008140)
* **[R13]** S. Decherchi, W. Rocchia, "A general and robust ray-casting-based algorithm for triangulating surfaces at the nanoscale (NanoShaper)," *PLoS ONE* 8(4):e59744, 2013. doi:[10.1371/journal.pone.0059744](https://doi.org/10.1371/journal.pone.0059744)
* **[R14]** J. Parulek, I. Viola, "Implicit representation of molecular surfaces," *IEEE PacificVis* 2012, 217–224. doi:[10.1109/PacificVis.2012.6183594](https://doi.org/10.1109/PacificVis.2012.6183594)
* **[R15]** P. Hermosilla, M. Krone, V. Guallar, P.-P. Vázquez, À. Vinacua, T. Ropinski, "Interactive GPU-based generation of solvent-excluded surfaces," *The Visual Computer* 33(6–8):869–881, 2017. doi:[10.1007/s00371-017-1397-2](https://doi.org/10.1007/s00371-017-1397-2)
* **[R16]** X. Martinez, M. Krone, M. Baaden, "QuickSES: a library for fast computation of solvent excluded surfaces," *Eurographics Workshop on Molecular Graphics and Visual Analysis (MolVA)*, 2019. <https://diglib.eg.org/handle/10.2312/molva20191095>
* **[R17]** B. Liu, B. Wang, R. Zhao, Y. Tong, G.-W. Wei, "ESES: software for Eulerian solvent excluded surface," *J. Comput. Chem.* 38(7):446–466, 2017. doi:[10.1002/jcc.24682](https://doi.org/10.1002/jcc.24682)
* **[R18]** T. Can, C.-I. Chen, Y.-F. Wang, "Efficient molecular surface generation using level-set methods (LSMS)," *J. Mol. Graph. Model.* 25(4):442–454, 2006.
* **[R19]** T. Goddard, "Molecular surface algorithms" (UCSF Chimera technical notes), 2013. <https://www.cgl.ucsf.edu/chimera/data/surface-oct2013/surface.html>
* **[R20]** UCSF ChimeraX User Guide, `surface` command. <https://rbvi.ucsf.edu/chimerax/docs/user/commands/surface.html>
* **[R21]** L. Wang, "sesa: a program for the analytic computation of solvent-excluded surface areas," *ChemistryOpen* 13(12):e202400172, 2024. doi:[10.1002/open.202400172](https://doi.org/10.1002/open.202400172)

Isosurface extraction, sparse volumes, mesh processing

* **[R22]** W. E. Lorensen, H. E. Cline, "Marching cubes: a high resolution 3D surface construction algorithm," *SIGGRAPH '87*, *Comput. Graph.* 21(4):163–169, 1987. doi:[10.1145/37402.37422](https://doi.org/10.1145/37402.37422)
* **[R23]** T. Lewiner, H. Lopes, A. W. Vieira, G. Tavares, "Efficient implementation of Marching Cubes' cases with topological guarantees," *J. Graphics Tools* 8(2):1–15, 2003. doi:[10.1080/10867651.2003.10487582](https://doi.org/10.1080/10867651.2003.10487582)
* **[R24]** L. Custódio, T. Etiene, S. Pesco, C. Silva, "Practical considerations on Marching Cubes 33 topological correctness," *Computers & Graphics* 37(7):840–850, 2013.
* **[R25]** W. Schroeder, R. Maynard, B. Geveci, "Flying edges: a high-performance scalable isocontouring algorithm," *IEEE LDAV* 2015. doi:[10.1109/LDAV.2015.7348069](https://doi.org/10.1109/LDAV.2015.7348069)
* **[R26]** S. F. F. Gibson, "Constrained elastic surface nets," *MICCAI '98*, LNCS 1496:888–898, 1998. doi:[10.1007/BFb0056277](https://doi.org/10.1007/BFb0056277)
* **[R27]** T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual contouring of Hermite data," *ACM TOG* 21(3):339–346, 2002.
* **[R28]** M. Garland, P. S. Heckbert, "Surface simplification using quadric error metrics," *SIGGRAPH '97*, 209–216. doi:[10.1145/258734.258849](https://doi.org/10.1145/258734.258849)
* **[R29]** G. Taubin, "A signal processing approach to fair surface design," *SIGGRAPH '95*, 351–358, 1995.
* **[R30]** K. Museth, "VDB: high-resolution sparse volumes with dynamic topology," *ACM TOG* 32(3):27, 2013. doi:[10.1145/2487228.2487235](https://doi.org/10.1145/2487228.2487235)
* **[R31]** S. F. Frisken, R. N. Perry, A. P. Rockwood, T. R. Jones, "Adaptively sampled distance fields," *SIGGRAPH 2000*, 249–254.

Distance transforms

* **[R32]** P. F. Felzenszwalb, D. P. Huttenlocher, "Distance transforms of sampled functions," *Theory of Computing* 8:415–428, 2012. doi:[10.4086/toc.2012.v008a019](https://doi.org/10.4086/toc.2012.v008a019)
* **[R33]** C. R. Maurer Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing exact Euclidean distance transforms of binary images in arbitrary dimensions," *IEEE TPAMI* 25(2):265–270, 2003. doi:[10.1109/TPAMI.2003.1177156](https://doi.org/10.1109/TPAMI.2003.1177156)

Atomic radii

* **[R34]** A. Bondi, "van der Waals volumes and radii," *J. Phys. Chem.* 68(3):441–451, 1964. doi:[10.1021/j100785a001](https://doi.org/10.1021/j100785a001)
* **[R35]** M. Mantina, A. C. Chamberlin, R. Valero, C. J. Cramer, D. G. Truhlar, "Consistent van der Waals radii for the whole main group," *J. Phys. Chem. A* 113(19):5806–5812, 2009. doi:[10.1021/jp8111556](https://doi.org/10.1021/jp8111556)
* **[R36]** S. Alvarez, "A cartography of the van der Waals territories," *Dalton Trans.* 42:8617–8636, 2013. doi:[10.1039/C3DT50599E](https://doi.org/10.1039/C3DT50599E)
* **[R37]** J. Tsai, R. Taylor, C. Chothia, M. Gerstein, "The packing density in proteins: standard radii and volumes (ProtOr)," *J. Mol. Biol.* 290:253–266, 1999. doi:[10.1006/jmbi.1999.2829](https://doi.org/10.1006/jmbi.1999.2829)

File formats

* **[R38]** wwPDB, *Atomic Coordinate Entry Format Version 3.3*. <https://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html>
* **[R39]** wwPDB, *PDBx/mmCIF dictionary resources*. <https://mmcif.wwpdb.org/>
* **[R40]** PDBj, *Large structures* (PDB-format limits: > 62 chains, > 99 999 atoms, etc.). <https://pdbj.org/help/large-structures>
* **[R41]** R. W. Grosse-Kunstleve, *Hybrid-36 encoding for PDB serial and residue numbers*. <https://cci.lbl.gov/hybrid_36/>
* **[R42]** D. Sehnal, S. Bittrich, S. Velankar, J. Koča, A. S. Rose, "BinaryCIF and CIFTools," *PLoS Comput. Biol.* 16(10):e1008247, 2020. doi:[10.1371/journal.pcbi.1008247](https://doi.org/10.1371/journal.pcbi.1008247)
* **[R43]** libAtoms, *Extended XYZ specification*. <https://github.com/libAtoms/extxyz>
* **[R44]** APBS documentation, *PQR molecular structure format*. <https://apbs.readthedocs.io/en/latest/formats/pqr.html>
* **[R45]** P. Bourke (ed.), *Wavefront OBJ file format* (Appendix B1 of the Advanced Visualizer manual). <https://paulbourke.net/dataformats/obj/>
* **[R46]** M. Wojdyr, "GEMMI: a library for structural biology," *JOSS* 7(73):4200, 2022. doi:[10.21105/joss.04200](https://doi.org/10.21105/joss.04200) (used as an independent reference parser in tests).

Rust ecosystem

* **[R47]** rayon — data-parallelism library. <https://crates.io/crates/rayon>
* **[R48]** fearless_simd — safe portable SIMD with runtime dispatch (SSE2/SSE4.2/AVX2/AVX-512, NEON, WASM). <https://docs.rs/fearless_simd>
* **[R49]** Alternatives: `pulp` <https://crates.io/crates/pulp>, `wide` <https://crates.io/crates/wide>, `multiversion` <https://crates.io/crates/multiversion>
* **[R50]** Rust `portable_simd` tracking issue (std::simd, nightly-only). <https://github.com/rust-lang/rust/issues/86656>
* **[R51]** flate2 with the zlib-rs backend. <https://github.com/rust-lang/flate2-rs>, <https://trifectatech.org/projects/zlib-rs/>
* **[R52]** meshoptimizer (A. Kapoulkine) and its Rust bindings `meshopt`. <https://github.com/zeux/meshoptimizer>, <https://crates.io/crates/meshopt>
* **[R53]** pdbtbx — Rust PDB/mmCIF library (evaluated, not adopted; see Design §5.1). <https://github.com/douweschulte/pdbtbx>
* **[R54]** rust-sasa — Rust SASA calculation (Shrake–Rupley). <https://crates.io/crates/rust-sasa>
