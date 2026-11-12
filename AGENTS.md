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
- `mesh/` `Mesh`, validation, components (`mod.rs`), generated MC table (`mc_table.rs`,
  generator + exhaustive tests in `tables.rs`; regenerate with `SESMESH_BLESS=1 cargo test -p
  sesmesh mc_table`), per-brick MC + exact refinement + assembly (`extract.rs`)
- `surface.rs` `SurfaceParams`, `compute_surface`, stats · `export/` OBJ writer, fast formatter
- `par.rs` rayon facade (feature `parallel`) · `crates/sesmesh-testkit` brute-force oracle,
  analytic two-sphere references, cluster generators, minimal OBJ reader
- `crates/sesmesh-cli/src/main.rs` CLI (`surface` default, `info`, `radii`), tests in `tests/cli.rs`

## Status

- Done: Phases 0–5 = v0.1 MVP (file → closed SES/SAS/vdW mesh → OBJ, library + CLI) plus
  `SurfaceEngine`, `Control` (cancel/progress), `--progress`, determinism suite, scaling study
  (D§11.1). Deviations are noted in the plan's per-phase status boxes. `control.rs` holds
  `Control`/`Progress`/`Stage`; `field::ScratchPool` backs `SurfaceEngine`.
- Next — Phase 6: batched SIMD vertex refinement with the brick's gathered candidates (largest
  perf lever, D§7.7), kernel tuning, x86 levels in CI. Phase 7: cavity modes, component filters,
  OBJ groups/colours, JSON stats.
- Accuracy: inscribed MC mesh underestimates area/volume (D§15 "Measured"; spec NFR-ACC note).
- Performance notes and next optimizations: D§7.7.
- `out/` holds untracked meshes from a throwaway prototype (not part of the library).
