# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Mesh decimation (#1, FR-LOD-4): `mesh::decimate` with `mesh::Decimation`
  (`ratio`, `max_triangles`, `max_error`) and `mesh::DecimationStats`;
  `SurfaceParams::decimation` / `with_decimation`, `SurfaceStats::decimation`.
  Pure-Rust quadric-error half-edge collapses: retained vertices keep their
  exact surface position, analytic normal and attributes; the mesh stays a
  closed, oriented manifold; deterministic. CLI: `--decimate R`,
  `--max-triangles N`, `--decimate-error E`; the OBJ/PLY header and the
  statistics JSON record the decimation (`triangles_before_decimation`).
- Automatic coarsening under a memory limit (#3, FR-LOD-6):
  `SurfaceParams::auto_coarsen` / `with_auto_coarsen`, CLI `--auto-coarsen`
  (with `--max-memory`): the grid spacing grows until the estimate fits
  instead of failing; `SurfaceStats::spacing` reports the spacing used.
- gzip-compressed PLY and MSMS output (#15): `-o mesh.ply.gz`,
  `-o mesh.vert.gz` (writes `mesh.vert.gz` and `mesh.face.gz`);
  `export::write_msms_paths` takes explicit `.vert` / `.face` paths.
- Shell completions and manual page (#2): `sesmesh completions
  <bash|zsh|fish|powershell|elvish>` and `sesmesh man`; release archives
  contain `sesmesh.1` and the bash, zsh, fish and PowerShell scripts.
- Ctrl-C handling (#12): SIGINT cancels the computation through `Control`
  within about 100 ms; the CLI exits with code 130 ("interrupted") and starts
  no further batch input.
- Fuzz target `assembly` (#22): mmCIF and PDB parsing with biological
  assembly expansion.

### Changed

- Lower memory use (#21), output unchanged (byte-identical meshes): the
  field keeps neither the SAS arcs nor the neighbour lists of buried spheres,
  `Circle` no longer stores its second basis vector, the arrangement arrays
  have exact capacities, and the arrangement build, brick classification
  and per-brick meshes use flat per-chunk arrays instead of per-sphere and
  per-brick vectors. Field data per atom: ≈ 1 150 → 560 bytes; peak memory
  on the 1K4R capsid at 1 Å: 2.42 → 1.74 GB.

### Fixed

- Biological assemblies whose operator expressions expand to more than
  100 000 copies or 500 million atoms are refused with `Error::ResourceLimit`
  (exit code 4) instead of exhausting memory.

## [0.9.1] - 2026-11-13

### Fixed

- CLI: a reader closing standard output early (e.g. `sesmesh x.cif -o - |
  head`, `--stats - | head`) no longer reports "error: standard output:
  Broken pipe" with exit code 1; the CLI stops quietly with exit code 0.
  `--stats -` no longer panics in that case.

## [0.9.0] - 2026-11-13

First public release (release candidate for 1.0; the open v1.0 criteria are
listed in `docs/release.md` and GitHub issue #19).

### Added

- Product specification, design document and implementation plan (`docs/`).
- Phase 0 scaffold (T0.1–T0.5): Cargo workspace (`sesmesh`, `sesmesh-cli`,
  `sesmesh-testkit`, `xtask`), shared lints and build profiles, MSRV 1.89.
- CI workflows: rustfmt, clippy, tests on macOS arm64 / Linux x86-64 /
  Linux arm64 / Windows x86-64, rustdoc, cargo-deny, MSRV check; nightly
  workflow for long tests.
- `cargo xtask sets` and `cargo xtask fetch --set <name>`: download of RCSB
  test/benchmark structures with SHA-256 lock file (`tests/data/sets.lock`).
- Dual MIT / Apache-2.0 licence, README, CONTRIBUTING.
- Phase 1 structure layer (T1.1–T1.12):
  - Readers for PDB (fixed columns, hybrid-36 serials/residue numbers,
    MODEL/ENDMDL/END, CRLF, chunked parallel parsing above 8 MB), PDBx/mmCIF
    (CIF 1.1 lexer, `_atom_site` in any column order, auth/label IDs, data
    block choice), XYZ and extended XYZ (`Properties=`, radius column) and PQR;
    gzip detection by magic bytes; format detection by extension and content.
  - `Element` (all 118 elements, inference from PDB-style atom names),
    `Structure`/`Model`/`AtomRecord` with an interned `StringPool`,
    aggregated `Diagnostics` warnings, `Error` with file/line/column.
  - `Selection`: model choice, per-residue alt-loc policies, hydrogen,
    water, hetero, chain and residue filters, custom predicates.
  - Radii: Bondi (+ Mantina 2009), Alvarez 2013, ProtOr united-atom radii,
    uniform radius, TOML custom rules, file-provided radii; strict mode.
  - Vendored fixtures (1CRN, 1UBQ, 3FX2, 2PTN, 1L2Y, 1EJG in PDB and mmCIF;
    small PQR/XYZ/extXYZ files), parser benchmarks, six fuzz targets and a
    nightly fuzz job.
  - Decompressed-size limit for gzip input (`ReadOptions::max_decompressed_bytes`,
    `Error::ResourceLimit`).

- Phase 2 geometry (T2.1–T2.9):
  - Public `Sphere` input type (`Sphere::from_atoms`) and
    `geom::arrangement_stats`.
  - Internal SAS arrangement: Morton-ordered spheres, CSR cell lists,
    containment and duplicate removal, neighbour lists, intersection circles
    with exposed arcs and blocker lists (trig-free pseudo-angle interval
    union), exposed triple points (probe positions), sphere exposure flags.
  - `sesmesh-testkit`: brute-force SES/SAS/vdW oracle, two-sphere analytic
    area/volume references (closed form and quadrature, incl. the
    self-intersecting torus), deterministic random and symmetric clusters and
    proptest strategies.
  - Verification: arrangement matches the oracle and dense sampling on 10 000
    random clusters, symmetric lattices/cages and crambin; arrangement
    benchmark.

- Phase 3 field engine with multi-threading and SIMD built in (T3.1–T3.9,
  plus parts of T5 and T6 moved forward):
  - `SesField`: exact `f64` SES/SAS/vdW field at any point, with closest
    feature (`Patch`) and outward normal; `SurfaceKind`, `SimdLevel`.
  - Sparse brick lattice: touched bricks, conservative Lipschitz
    classification of bricks and 2×2×2 sub-blocks, per-sub-block candidate
    gathering, `fearless_simd` kernels (8 lanes, runtime dispatch:
    AVX-512/AVX2/SSE4.2/NEON/WASM/scalar), exact `f64` path for very large
    coordinates; `field::sample_surface_bricks` for benchmarks/diagnostics.
  - rayon facade (`parallel` feature); parallel arrangement construction
    (1AON 0.73 s → 0.16 s), classification and sampling, deterministic for
    any thread count.
  - Tests: point field and lattice samples vs the oracle (SES/SAS/vdW),
    classification soundness, brick-seam sign consistency, symmetric inputs,
    SIMD vs scalar agreement, 1 vs N threads; field benchmark.

- Phase 4 MVP (T4.1–T4.9): structure file → closed SES mesh → OBJ.
  - Marching Cubes table generated by a test-only generator with a uniform
    face rule (ambiguous faces separate inside corners) and loop
    triangulations chosen among all candidates (no diagonal inside a cube
    face, no inverted vertex, least area, best minimum angle); exhaustive
    tests of every case and of all 12 288 shared-face pairings.
  - Per-brick meshing over the sampled bricks with edge ownership, exact
    vertex placement (safeguarded Newton on the analytic field, 2.2–2.5
    evaluations per vertex), analytic normals, patch kind and nearest atom
    per vertex; parallel global assembly; deterministic for any thread count.
  - `Mesh`, `PatchKind`, `mesh::{validate, components, area_volume}`
    (closedness, orientation, degeneracy, duplicates; per-component area,
    signed volume, Euler characteristic, cavity flag).
  - `compute_surface`, `SurfaceParams` (SES/SAS/vdW, probe, spacing or
    quality presets, refinement mode, brick size, SIMD level),
    `SurfaceOutput`, `SurfaceStats` with stage timings; `Error::Internal`.
  - `export::obj`: OBJ writer with header comments, precision 0–9,
    optional normals, gzip output, parallel chunked formatting with a
    fixed-point formatter byte-identical to `format!`.
  - `sesmesh` CLI: `surface` (default), `info`, `radii`; input, selection,
    radii, surface, resolution, output and performance options; exit codes
    0/1/2/3/4.
  - `BuiltinRadii::atom_table`; testkit minimal OBJ reader.
  - Tests: analytic one/two-sphere accuracy and O(h²) convergence, brick
    sizes, robustness corpus, random-cluster proptest, thread-count
    byte-identity, OBJ round trip and snapshot, CLI end to end; `surface`
    benchmark (1AON at 0.25 Å: 17.5 M triangles in ≈ 8.8 s on 10 cores).

- Phase 5 parallelism (T5.5–T5.7, plus `SurfaceEngine` from T4.7):
  - `Control` (cancellation flag, progress callback), `Progress`, `Stage`,
    `compute_surface_with`, `Error::Cancelled`; cancellation checked per
    sphere, per brick and between stages (< 100 ms latency).
  - `SurfaceEngine` for repeated computations (keeps per-thread sampling
    buffers between calls).
  - CLI `--progress` (live line on a terminal, one line per stage
    otherwise).
  - Parallel connected components (lock-free union–find) and validation
    sharing one sort of packed edge keys: measure stage 0.30 s → 0.16 s on
    1AON at 0.5 Å.
  - Determinism suite (1–16 threads, repeated runs, byte-identical OBJ) and
    thread-scaling benchmark; scaling analysis in the design document.

- Phase 6 SIMD and tuning (T6.1–T6.3, T6.5–T6.7):
  - `SimdLevel::{Sse4_2, Avx2, Avx512, Neon}`, `SimdLevel::{ALL, name,
    from_name, effective}`; CLI `--simd auto|scalar|sse4.2|avx2|avx512|neon`
    (explicit levels capped at the detected one, other architectures are
    usage errors).
  - Vertex refinement evaluates exact `f64` candidate lists gathered once
    per sub-block, with blockers restricted to nearby spheres.
  - Default brick size 10 (was 8).
  - Tests: every available SIMD level against the fallback, coordinates
    offset by 10⁴ Å, 5 000 Å extents (automatic `f64` path); CI job summary
    with the SIMD level matrix; field benchmark at every level.

- Phase 7 mesh features (T7.1–T7.5):
  - Triangles grouped by component (`Component::first_triangle`,
    `Mesh::vertex_component`).
  - `CavityMode::{Keep, Remove, Only}`, `min_cavity_volume`,
    `min_component_area` (components inside a removed cavity go with it),
    `SurfaceStats::removed_components`.
  - `NormalMode::{Analytic, FaceAverage, None}`; meshes without normals
    validate and write without `vn`.
  - `mesh::atom_areas`; `color` module (`ElementPalette` from the bundled
    `atom-colors.json`, categorical and patch colours, `parse_hex`).
  - OBJ vertex colours (`v x y z r g b`) and groups (`ObjOptions::{vertex_colors,
    groups}`).
  - CLI: `--cavities`, `--min-cavity-volume`, `--min-component-area`,
    `--face-normals`, `--group-components`, `--vertex-colors
    element|chain|patch|component`, `--palette`, `--vertex-data`,
    `--atom-areas`, `--stats` (JSON).
  - `docs/04-output-formats.md`: OBJ header, colours, CSV columns, JSON
    statistics schema.

- Phase 8 level of detail and outputs (T8.1–T8.3, T8.5–T8.7):
  - `Resolution::Density` / `--density` (vertices per Å²).
  - `estimate_memory`, `SurfaceParams::max_memory` / `--max-memory`
    (`Error::ResourceLimit`, exit code 4).
  - Taubin smoothing with exact re-projection (`--smooth N`).
  - Binary PLY (`export::ply`) and MSMS `.vert`/`.face` (`export::msms`)
    writers; `--output-format obj|ply|msms`.
  - Biological assemblies from mmCIF and PDB (`ReadOptions::assembly`,
    `io::read_assemblies`, `--assembly`; `info` lists them).
  - Batch mode: several inputs, `--output-dir`, `--keep-going`.
  - Testkit binary PLY reader.

- Phase 9 validation (T9.1, T9.3–T9.5, T9.7; internal references):
  - `[sets.validation]` (53 entries) and `cargo xtask sample-sweep`
    (`[sets.sweep]`, 200 random entries, seed 2026).
  - `cargo xtask validate` (accuracy vs h → 0 extrapolation, density
    calibration) and `cargo xtask sweep` (robustness); results in
    `docs/validation/` and `docs/validation-report.md`.
  - `estimate_memory` takes the sphere count and uses measured costs
    (150 B/triangle + 800 B/sphere); density test tightened to ±5 %.

- Phase 10 documentation and release preparation (T10.1–T10.5):
  - Crate guide in the rustdoc (concepts, modules, features, versioning) and
    five examples (`file_to_obj`, `spheres_to_mesh`, `trajectory`,
    `custom_radii`, `field_queries`), run in CI.
  - `docs/cli.md` (CLI manual), `docs/api-review.md` (API Guidelines
    checklist, `#[non_exhaustive]` audit, semver policy),
    `docs/release.md` (process and v1.0 status).
  - Release workflow: tier-1 binaries (`dist` profile) with checksums, draft
    GitHub release on `v*` tags, manual crates.io publish; `sesmesh` and
    `sesmesh-cli` are publishable (`cargo publish --workspace --dry-run`
    passes), with repository and documentation metadata.
  - README: documentation links, method description and citations.
  - `Warning` is `#[non_exhaustive]`; `tests/api.rs` checks `Send + Sync`
    of the main types; CLI help mentions PLY and MSMS.

### Fixed

- Validation ignored vertex positions when a mesh had no normals.
- Meshes far from the origin (≈ 10⁴ Å) could contain zero-area triangles:
  vertices placed 10⁻⁴·h from a sample collapsed into one `f32` position.
  The edge clamp is now at least four `f32` ulps of the largest coordinate.
- XYZ reader no longer pre-allocates from an untrusted atom count (found by
  fuzzing: a count of 377 777 773 requested 36 GB).

[Unreleased]: https://github.com/uvar-apps/sesmesh/compare/v0.9.1...HEAD
[0.9.1]: https://github.com/uvar-apps/sesmesh/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/uvar-apps/sesmesh/releases/tag/v0.9.0
