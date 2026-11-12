# Agent guide — sesmesh

Rust library + CLI computing Solvent Excluded Surfaces (SES) of molecules → OBJ meshes.
Read `docs/01-product-specification.md`, `docs/02-design.md`, `docs/03-implementation-plan.md` first;
task IDs (T4.1 …) and requirement IDs (FR-*, NFR-*) come from there. Conventions: `CONTRIBUTING.md`.

## Commands

```bash
cargo test --workspace                     # all tests (debug, ~1 min)
cargo test --workspace --release -- --include-ignored   # incl. 10k-cluster + 3J3Q tests
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all
cargo xtask fetch --set small --set bench  # test/benchmark structures (RCSB) into tests/data/cache
cargo bench -p sesmesh --bench field       # also: parse, geometry
```

MSRV 1.89, edition 2024, `unsafe_code = "forbid"`, the library never prints, output must not
depend on thread count. Keep docs and CHANGELOG in sync with any deviation from the plan.

## Layout (crates/sesmesh/src)

- `io/` readers (PDB, mmCIF, XYZ, PQR, gzip) · `structure/` atoms, selection · `radii/` tables
- `geom/` `Sphere`, cell lists, SAS arrangement (circles, exposed arcs, SAS vertices)
- `field/` `SesField` exact f64 field (`point.rs`), lattice + brick/sub-block classification
  (`lattice.rs`), per-sub-block gathering + `fearless_simd` kernels + `Engine::map_bricks`
  (`kernel.rs`), tests (`tests.rs`)
- `par.rs` rayon facade (feature `parallel`) · `crates/sesmesh-testkit` brute-force oracle,
  analytic two-sphere references, cluster generators

## Status

- Done: Phases 0–3 (scaffold/CI, structure layer, SAS geometry, multi-threaded SIMD field engine).
  Parts of Phases 5/6 (rayon, SIMD dispatch) were moved into Phase 3.
- Next — Phase 4 (MVP): per-brick Marching Cubes over `Engine::map_bricks` samples (values are
  exact or `±band` sign-only — only signs matter for case selection; refine crossing edges with
  `SesField` exact roots), edge ownership + global assembly (D§8), analytic normals from the
  closest point, OBJ writer, `SurfaceParams`/engine API, CLI `sesmesh surface|info|radii`,
  analytic one/two-sphere accuracy tests (testkit `analytic`).
- Performance notes and next optimizations: D§7.7.
- `out/` holds untracked meshes from a throwaway prototype (not part of the library).
