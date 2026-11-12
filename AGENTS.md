# Agent guide — sesmesh

Rust library + CLI computing Solvent Excluded Surfaces (SES, also SAS / vdW) of molecules →
closed, oriented triangle meshes with analytic normals → OBJ. Read
`docs/01-product-specification.md`, `docs/02-design.md`, `docs/03-implementation-plan.md` first;
task IDs (T7.1 …) and requirement IDs (FR-*, NFR-*) come from there. Every plan phase has a
"Status" box listing what was built and every deviation. Conventions: `CONTRIBUTING.md`.

## Commands

```bash
cargo test --workspace                     # all tests (debug, ~1.5 min)
cargo test --workspace --release -- --include-ignored   # + 10k clusters, 3J3Q, 4HHB/1AON determinism
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy -p sesmesh -p sesmesh-cli --lib --bins --target x86_64-unknown-linux-gnu -- -D warnings  # x86-only cfg code
cargo fmt --all
cargo +1.89 check -p sesmesh -p sesmesh-cli  # MSRV
cargo xtask fetch --set small --set bench  # RCSB structures into tests/data/cache (untracked)
cargo bench -p sesmesh --bench surface     # end to end; also: field (per SIMD level), scaling, parse, geometry
SESMESH_BLESS=1 cargo test -p sesmesh mc_table   # regenerate mesh/mc_table.rs (then rerun tests)
target/release/sesmesh data/1crn.cif -q high --timings   # CLI smoke run (writes 1crn.obj)
```

MSRV 1.89, edition 2024, `unsafe_code = "forbid"`, `missing_docs = "deny"`, the library never
prints, output must not depend on thread count (byte-identical OBJ is tested), no third-party
types in the public API, public option structs are `#[non_exhaustive]` (outside the crate build
them with `Default` + field assignment). Keep docs, plan status boxes and CHANGELOG in sync with
any deviation from the plan.

## Working with this user

- One branch per phase (`phase-N-name`), created from `main`. Commit only when asked ("Commit
  the working tree changes with a sensible message"): imperative subject, bullet body, trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Then, when asked, fast-forward
  `main` (`git merge --ff-only`). Push only when asked: `origin` =
  https://github.com/uvar-apps/sesmesh (private); push `main` and the phase branches.
- Never commit `out/` (review meshes: prototype files plus `*-sesmesh.obj` from the real CLI).
- `atom-colors.json` (repo root, untracked, user-provided): 118 element colours,
  `{"format":"eigenvista-atom-colors","version":1,"colors":[{"z","symbol","color":"#rrggbb"}]}`.
  Use it as the default palette for `--vertex-colors element` (Phase 7, T7.5); ask before
  committing it (e.g. as `crates/sesmesh/data/atom-colors.json` + `include_str!`).
- The user reviews meshes in a viewer; the crambin SES was confirmed "perfect" (T4.10).

## Repository state at handoff (2026-11-13)

- `origin/main` = Phase 5 (`51ea399`); the first GitHub CI run on it passed on all jobs.
- Branch `phase-6-simd`: Phase 6 and this handoff, committed locally (all local checks pass).
  Not yet merged into `main` or pushed: fast-forward `main` and push when the user asks, then
  check the CI run (the new "SIMD level matrix" step runs there for the first time).
- `atom-colors.json` and `out/` are untracked on purpose.
- Phases 0–6 done. Next: Phase 7 on a new branch `phase-7-mesh` from `main`.

## Layout (crates/sesmesh/src)

- `io/` readers (PDB, mmCIF, XYZ, PQR, gzip) · `structure/` atoms, selection · `radii/` tables
- `geom/` `Sphere`, cell lists, SAS arrangement (circles, exposed arcs, SAS vertices), built in
  parallel with cancellation checks (`Arrangement::build_with`)
- `field/` `SesField` exact f64 field (`point.rs`: global `ses_eval`; per-sub-block
  `gather_local`/`eval_local` with locally restricted blockers, used by refinement), lattice +
  brick/sub-block classification (`lattice.rs`), gathering + `fearless_simd` kernels +
  `Engine::map_bricks_with` + `ScratchPool` + SIMD level selection (`kernel.rs`), tests
- `mesh/` `Mesh`, `validate`, parallel `components` (lock-free union–find), packed edge keys
  (`mod.rs`); generated MC table (`mc_table.rs`, generator + exhaustive tests in `tables.rs`);
  per-brick MC, exact refinement, ulp-aware `edge_clamp`, parallel assembly (`extract.rs`)
- `surface.rs` `SurfaceParams`, `compute_surface[_with]`, `SurfaceEngine`, stats ·
  `control.rs` `Control`/`Progress`/`Stage` · `export/` OBJ writer + fixed-point formatter
- `par.rs` rayon facade (feature `parallel`) · `crates/sesmesh-testkit` brute-force oracle,
  analytic two-sphere references, cluster generators, minimal OBJ reader
- `crates/sesmesh-cli/src/main.rs` CLI (`surface` default, `info`, `radii`; `--progress`,
  `--simd`, `-j`, `--timings`), end-to-end tests in `crates/sesmesh-cli/tests/cli.rs`
- Tests: unit tests next to the code; `crates/sesmesh/tests/{surface,determinism,structures}.rs`;
  data-dependent tests skip when `tests/data/cache` is missing (the nightly workflow fetches it)

## Status

- Phase 4 (v0.1 MVP): file → mesh → OBJ, library + CLI, exit codes 0/1/2/3/4.
- Phase 5: `Control` (cancel < 100 ms, progress), `SurfaceEngine`, `--progress`, determinism
  suite (1–16 threads), scaling study (D§11.1: ~80 % on 4 P-cores, 4.5–4.9× on 10 cores).
- Phase 6: `SimdLevel::{Sse4_2,Avx2,Avx512,Neon}` capped at the detected level, CI SIMD level
  matrix in the job summary, f64 refinement over per-sub-block candidate lists, default brick
  size 10, fix for f32 vertex collapse far from the origin, f64 path for > 2 000 Å extents.
  Profile and rejected experiments: D§7.5; x86 speed-ups still to measure (Phase 9).
- Accuracy: the inscribed MC mesh underestimates area (0.4–0.65 %) and volume (0.8–1.35 %) at
  0.25 Å for atom-sized spheres, O(h²) (D§15 "Measured", spec NFR-ACC note).
- Performance: D§7.7 and D§11.1 (1AON at 0.5 Å ≈ 3.1 s, 4HHB at 0.25 Å ≈ 0.54 s, 10 cores).

## Next: Phase 7 (plan §3, D§9, FR-SURF-4..7, FR-MESH-6, FR-OUT-2/5/6/7)

Already available: `mesh::components` (deterministic ids, area, signed volume, χ, genus,
`is_cavity`), `SurfaceOutput::components`, `Mesh::vertex_atom` (input sphere index) and
`vertex_patch`. To do:

1. T7.1: reorder triangles so each component is contiguous (store triangle ranges); genus tests
   (torus-shaped atom ring); multi-molecule inputs.
2. T7.2: `CavityMode { Keep, Remove, Only }` + `min_cavity_volume` / `min_component_area` in
   `SurfaceParams`; nested-component enclosure test (ray parity, D§9.2); CLI `--cavities`,
   `--min-cavity-volume`, `--min-component-area`; compact vertices after removal.
3. T7.3: per-atom area (each triangle's area split equally among its vertices' nearest atoms,
   fixed-order sums, input atom order); `--atom-areas <csv>`.
4. T7.4: normal modes analytic (default) / face-average / none (`--face-normals`).
5. T7.5: OBJ `--group-components` (`g component_N`), `--vertex-colors
   element|chain|patch|component` (`v x y z r g b`; element palette from `atom-colors.json`),
   `--vertex-data <csv>`, JSON stats behind a `json` feature (`--stats [<json>]`, schema in
   `docs/`). Element and chain colouring need atom records: the CLI maps `vertex_atom` through
   the `AtomSet`.

## Gotchas learned

- Timing on this machine is noisy (Defender and other apps; load average often 10–25): write
  outputs to `/dev/null`, check `uptime`, compare with criterion baselines
  (`--save-baseline` / `--baseline`), repeat runs. Single-thread runs are steadier.
- zsh does not word-split `$var`; use `${=var}` in loops. Foreground `sleep` is blocked; run
  long jobs in the background.
- Clippy here is strict (pedantic subset): `is_multiple_of`, method paths instead of closures,
  `;` after unit expressions, `clamp`. Code under `#[cfg(target_arch = "x86_64")]` is only
  linted by the x86 target command above.
- A surface brick may own no crossing edge yet have crossing cells: the cell pass must always
  run (Phase 4 hole bug). MC triangulations must not put a diagonal inside a cube face (the
  neighbouring cell duplicates it). `Mesh` positions are f32 in the input frame: keep vertex
  separations above a few f32 ulps (`edge_clamp`).
- `out/` holds `1crn-ses-*` prototype meshes (marching tetrahedra) and `*-sesmesh.obj` from the
  real CLI for comparison.
