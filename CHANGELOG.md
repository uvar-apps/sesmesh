# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

### Fixed

- XYZ reader no longer pre-allocates from an untrusted atom count (found by
  fuzzing: a count of 377 777 773 requested 36 GB).
