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
SESMESH_GPU_NO_KERNELS=1 cargo clippy --workspace --all-targets --all-features -- -D warnings  # no ROCm here
cargo clippy -p sesmesh -p sesmesh-cli --lib --bins --target x86_64-unknown-linux-gnu -- -D warnings  # x86-only cfg code
cargo fmt --all
cargo +1.89 check -p sesmesh -p sesmesh-cli  # MSRV
cargo xtask fetch --set small --set bench  # RCSB structures into tests/data/cache (untracked)
cargo bench -p sesmesh --bench surface     # end to end; also: field (per SIMD level), scaling, parse, geometry
cargo xtask fetch --set validation --set sweep   # Phase 9 corpora (~70 MB)
cargo run --release -p xtask -- validate   # accuracy study → docs/validation/accuracy.csv (~3 min)
cargo run --release -p xtask -- sweep      # robustness sweep → docs/validation/sweep.csv (~2 min)
SESMESH_BLESS=1 cargo test -p sesmesh mc_table   # regenerate mesh/mc_table.rs (then rerun tests)
SESMESH_GPU_REQUIRE=1 cargo +stable test -p sesmesh-gpu --features hip   # on a ROCm machine (gfx1151)
SESMESH_GPU_REQUIRE=1 cargo +stable test --workspace --release --all-features -- --include-ignored  # ROCm: all + GPU
cargo +stable bench -p sesmesh --features hip --bench surface   # ROCm: adds …/hip-exact, …/hip-fast
tools/gpu/gate.sh /tmp/gate inputs.txt     # ROCm: CPU vs HIP exact/fast on a corpus (README there)
python3 -m unittest discover -s tools/ao   # ses_ao.py (end-to-end tests need aomesh >= 0.4.0)
target/release/sesmesh data/1crn.cif -q high --timings   # CLI smoke run (writes 1crn.obj)
```

Every source file starts with the header `// sesmesh — Solvent Excluded Surface meshes for
molecules` / `// Copyright (c) 2026 Ugo Varetto` / `// SPDX-License-Identifier: MIT OR
Apache-2.0` (comment syntax of the language; after a shebang, before `#![…]`; generators emit
it, see `CONTRIBUTING.md`). Author everywhere: Ugo Varetto (Cargo `authors`, licences, README,
site footer).

MSRV 1.89, edition 2024, `unsafe_code = "forbid"` (except `crates/sesmesh-gpu`, the GPU FFI:
`deny`, allowed in its `hip` module, `// SAFETY:` on every block), `missing_docs = "deny"`, the library never
prints, output must not depend on thread count (byte-identical OBJ is tested), no third-party
types in the public API, public option structs are `#[non_exhaustive]` (outside the crate build
them with `Default` + field assignment). Keep docs, plan status boxes and CHANGELOG in sync with
any deviation from the plan.

## Working with this user

- One branch per phase (`phase-N-name`), created from `main`. Commit only when asked ("Commit
  the working tree changes with a sensible message"): imperative subject, bullet body, trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Then, when asked, fast-forward
  `main` (`git merge --ff-only`). Push only when asked: `origin` =
  https://github.com/uvar-apps/sesmesh (public); push `main` and the phase branches.
- Validate locally before every push (full list: `docs/release.md`, *Local validation*: fmt,
  both clippy commands, debug + release `--include-ignored` tests, examples, rustdoc, MSRV,
  cargo-deny, publish dry run); never push and rely on GitHub CI, which is manual-only
  (`workflow_dispatch`). Release tags only on a locally validated commit.
- Never commit `out/` (review meshes: prototype files plus `*-sesmesh.obj` from the real CLI).
- `crates/sesmesh/data/atom-colors.json` (user-provided, 118 element colours,
  `{"format":"eigenvista-atom-colors","version":1,"colors":[{"z","symbol","color":"#rrggbb"}]}`)
  is the default element palette, turned into `color/element_table.rs` (a test keeps both in
  sync). The user's original root copy was removed on 2026-11-15 (identical to this one).
- The user reviews meshes in a viewer; the crambin SES was confirmed "perfect" (T4.10).

## Repository state at handoff (2026-11-15)

- Latest release **1.1.2** (2026-11-15; 1.1.1 and 1.1.0 on 2026-11-14; 1.0.0, 0.9.0 and
  0.9.1 on 2026-11-13): tags `v0.9.0`, `v0.9.1`, `v1.0.0`, `v1.1.0`, `v1.1.1`, `v1.1.2`,
  crates `sesmesh` and `sesmesh-cli` on crates.io (owner: the
  user's crates.io account), GitHub releases with tier-1 binaries. `main` = `origin/main` = the
  1.1.2 release commit (branch `release-1.1.2`). Semantic versioning from 1.0
  (`docs/api-review.md`); 1.x keeps default output byte-identical (new behaviour is opt-in),
  except correctness fixes that bring output to what the spec and design promise (user's
  decision, 2026-11-14): patch release, CHANGELOG names the meshes that change and by how much.
- Website: https://uvar-apps.github.io/sesmesh/ (GitHub Pages, source "GitHub Actions",
  deployed from `main` by `.github/workflows/pages.yml`; branch `website` holds its history).
- The crates.io token lives in the user's `~/.cargo/credentials.toml` (never read or print it).
  Next releases follow `docs/release.md` (bump `[workspace.package] version` and the `sesmesh`
  and `sesmesh-gpu` (`=x.y.z`) workspace dependencies, cut the changelog, `cargo publish --workspace --dry-run`, tag, publish).
- `out/` is untracked on purpose.
- Discoverability (2026-11-14): repo description + 13 topics set, `CITATION.cff` (bump
  `version`/`date-released` per release), crates' `homepage` + `science::bioinformatics`
  (on crates.io from the next publish). Left to the user: social preview upload, Google
  Search Console / Bing verification, announcements. Zenodo: record created by hand
  for 1.1.2 (version DOI 10.5281/zenodo.24238837, all versions 10.5281/zenodo.24238836); new releases are added
  there with *New version* (`docs/release.md` step 7), GitHub integration off.
- 1.0.0 contains the v1.0 development items (branch `v1-dev`): #1 decimation
  (`mesh/decimate.rs`, pure Rust), #3 `--auto-coarsen`, #15 gzip PLY/MSMS, #2
  `completions`/`man`, #12 Ctrl-C (exit 130, `ctrlc`), fuzz target `assembly` + assembly
  size limits, #21 memory (arrangement compaction, chunked build/meshing via `par::chunk_len`).
- Spec decisions for 1.0 (2026-11-13): NFR-MEM 150 B/triangle + 800 B/atom (#21), NFR-ACC
  analytic bound area ≤ 0.7 % / volume ≤ 1.5 % at 0.25 Å (#4), NFR-PERF = Phase 9
  measurements (#20); NFR-PERF reference machines = the measured 10-core M4 and 16-core
  Zen 5 (2026-11-15, no 8-P-core Apple machine). After 1.0 (tracking #19): cross-tool references #5/#6, 24 CPU-h fuzz
  campaign #22 (only 10 min per target before 1.0), x86 measurements #8/#9/#10, #13, #14,
  #18, #23 (#5/#6, #22 and #23 done 2026-11-14).

## Layout (crates/sesmesh/src)

- `io/` readers (PDB, mmCIF, XYZ, PQR, gzip), biological assemblies (`assembly.rs`:
  `ReadOptions::assembly`, `read_assemblies`) · `structure/` atoms, selection · `radii/` tables
- `geom/` `Sphere`, cell lists, SAS arrangement (circles, exposed arcs, SAS vertices), built in
  parallel with cancellation checks (`Arrangement::build_with`)
- `field/` `SesField` exact f64 field (`point.rs`: global `ses_eval`; per-sub-block
  `gather_local`/`eval_local` with locally restricted blockers, used by refinement), lattice +
  brick/sub-block classification (`lattice.rs`), gathering + `fearless_simd` kernels +
  `Engine::map_bricks_with` + `ScratchPool` + SIMD level selection (`kernel.rs`), tests
- `mesh/` `Mesh`, `validate`, parallel `components` (lock-free union–find), packed edge keys
  (`mod.rs`); generated MC table (`mc_table.rs`, generator + exhaustive tests in `tables.rs`);
  per-brick MC, exact refinement, ulp-aware `edge_clamp`, parallel assembly (`extract.rs`)
- `mesh/decimate.rs` QEM half-edge decimation (pass-based, deterministic)
- `mesh/post.rs` grouping by component, sub-grid fragments (`fragments`), cavity filtering +
  enclosure (ray parity), vertex compaction, face-averaged normals, `atom_areas`
- `backend.rs` `Backend::{Cpu, Auto, Hip}` (Phase 11), device cache, CPU fallback
- `surface.rs` `SurfaceParams` (incl. `CavityMode`, `NormalMode`, filters),
  `compute_surface[_with]`, `SurfaceEngine`, stats · `control.rs` `Control`/`Progress`/`Stage` ·
  `export/` OBJ (colours, groups), binary PLY, MSMS `.vert`/`.face` writers + fixed-point
  formatter · `color/` palettes · density/memory estimate (`Resolution::Density`,
  `estimate_memory`) and Taubin smoothing + re-projection (`mesh/post.rs`)
- `crates/sesmesh-gpu` (Phase 11, `docs/05-gpu-hip.md`, unpublished): GPU back ends, all the
  `unsafe`; feature `hip`: `hip/ffi.rs` (`libloading` of `libamdhip64`), `hip/mod.rs`
  (`Runtime`, `Device`, code-object selection), `memory.rs`, `stream.rs`, `selftest.rs`,
  `resident.rs` (`Resident`: arrangement uploaded once; `Batch`: pinned per-batch buffers;
  count → host scan → fill → sample passes; fast mode: `set_refinement` (f32 residuals of the
  frame-local copies) and `refine`: edge count → host scan → f32 refinement, each sub-block
  listing its crossing edges in the CPU's order); kernels in `kernels/hip/sesmesh.hip`, compiled by
  `build.rs` (README there). In `sesmesh`: `field/gpu.rs` (feature `hip`, Plan T11.3′:
  `Engine::resident`, `batch_params` (f32 radii + slack), `fold_brick_chunks_hip`: a device
  thread fills a ring of 3 `Batch` buffers (one ≤ 64-brick chunk per pool thread each; in fast
  mode ≥ 8 192 bricks, `Batching::mesh`, tuned in T11.6′), the
  pool takes chunks in order and meshes each once its batch is ready, no per-batch barrier;
  `Batching`), called from `surface::mesh_stage` when
  the back end is a HIP device and `Engine::device_ok` (f32 kernels, 8 sub-blocks per brick).
  `RefineMode::Fast` (T11.5′): the device also refines; `DeviceEdges` views a refined batch,
  `extract::mesh_brick_device` places vertices from it and refines unconverged edges in f64
  (`refine_cpu_edges`). Spikes S1 (`field/gpu_s1.rs`, ignored test `s1`,
  `docs/gpu-prior-art.md` §5.2) and S2 (`field/gpu_s2.rs`, `s2`, §5.4: fast vs exact vertices)
- `tools/gpu/` GPU gate (T11.6′): `gate.sh` (CPU vs `--backend hip` exact bytes and stats, fast
  faces, repeatability, area/volume; GNU `split`), `compare.py`, `summary.py`, README
- `par.rs` rayon facade (feature `parallel`) · `crates/sesmesh-testkit` brute-force oracle,
  analytic two-sphere references, cluster generators, minimal OBJ reader
- Website: `site/` (static HTML/CSS/JS, brutalist style, deployed by `.github/workflows/pages.yml`
  from `main`; Pages source must be "GitHub Actions"), helper scripts and instructions in
  `tools/site/README.md` (`ply2web.py` display meshes, `gallery-meshes.sh`, `render.html` +
  `capture.py` on port 8777, `og-card.py`), search-engine notes (canonical, Open Graph, JSON-LD,
  `sitemap.xml`). Library-page snippets were compiled against 1.1.0;
  re-check them when the API changes.
- `tools/ao/` ambient occlusion by post-processing (#27): `ses_ao.py` (stdlib Python) runs
  sesmesh to OBJ, then aomesh (separate program, not a dependency; ≥ 0.4.0: plain
  `map_Kd`, keeps OBJ header/objects/groups via objserde 0.3), `--bake-colors` writes a
  binary PLY with colours × AO (readable by `ply2web.py`); tests `test_ses_ao.py`, README,
  tutorial 04 on `site/tutorials.html`, figure `4hhb-chain-ao.webp` (`gallery-meshes.sh`).
  Interior cavities bake black (no outside viewpoint sees them).
- Docs: `docs/cli.md` (CLI manual), `docs/api-review.md` (API checklist, semver policy),
  `docs/release.md` (release process, v1.0 status), `.github/workflows/release.yml`
- `crates/sesmesh/examples/` five runnable examples (run in CI)
- `crates/sesmesh-cli/src/main.rs` CLI (`surface` default, `info`, `radii`; `--progress`,
  `--simd`, `-j`, `--timings`, `--cavities`, `--drop-fragments`, `--vertex-colors`, `--palette`, `--group-components`,
  `--face-normals`, `--vertex-data`, `--atom-areas`, `--stats`, `--density`, `--smooth`,
  `--max-memory`, `--auto-coarsen`, `--decimate`, `--max-triangles`, `--decimate-error`,
  `--output-format`, `--assembly`; batch mode with `--output-dir` / `--keep-going`;
  `completions`, `man` subcommands), end-to-end tests in `crates/sesmesh-cli/tests/cli.rs`. Output formats:
  `docs/04-output-formats.md`. Testkit has minimal OBJ and binary PLY readers.
- Tests: unit tests next to the code; `crates/sesmesh/tests/{surface,features,determinism,structures}.rs`;
  data-dependent tests skip when `tests/data/cache` is missing (the nightly workflow fetches it)

## Status

- Phase 4 (v0.1 MVP): file → mesh → OBJ, library + CLI, exit codes 0/1/2/3/4.
- Phase 5: `Control` (cancel < 100 ms, progress), `SurfaceEngine`, `--progress`, determinism
  suite (1–16 threads), scaling study (D§11.1: ~80 % on 4 P-cores, 4.5–4.9× on 10 cores).
- Phase 6: `SimdLevel::{Sse4_2,Avx2,Avx512,Neon}` capped at the detected level, CI SIMD level
  matrix in the job summary, f64 refinement over per-sub-block candidate lists, default brick
  size 10, fix for f32 vertex collapse far from the origin, f64 path for > 2 000 Å extents.
  Profile and rejected experiments: D§7.5; x86 speed-ups still to measure (Phase 9).
- Phase 7: components grouped (`Component::first_triangle`, `Mesh::vertex_component`), cavity
  modes + filters (components inside removed cavities removed too), normal modes, per-atom areas,
  OBJ groups + vertex colours, CSV side files, JSON stats (CLI-side `serde_json`, no library
  `json` feature). Deferred: 2PTN cavity comparison with reference tools (Phase 9).
- Phase 8: `--density` (h = √(1.5/d)), memory guard (exit 4), Taubin smoothing + exact
  re-projection, PLY/MSMS writers, mmCIF/PDB assemblies (1K4R capsid = 545 040 atoms, meshed at
  1 Å in ≈ 10 s), batch mode. Deferred: T8.4 decimation (`meshopt`, C++) and shell completions.
- Accuracy: the inscribed MC mesh underestimates area (0.4–0.65 %) and volume (0.8–1.35 %) at
  0.25 Å for atom-sized spheres, O(h²) (D§15 "Measured", spec NFR-ACC note).
- Performance: D§7.7 and D§11.1 (1AON at 0.5 Å ≈ 3.1 s, 4HHB at 0.25 Å ≈ 0.54 s, 10 cores).
- v1.0 development: decimation (crambin 0.25 Å 111 k → 28 k triangles in ≈ 120 ms; 1AON
  0.25 Å ≈ 6.5 s), auto-coarsen, gzip PLY/MSMS, completions + man page, Ctrl-C, memory
  (field ≈ 560 B/atom, 1K4R capsid 1 Å peak 2.42 → 1.74 GB).
- 1.1.0: memory guard fixed (Design §8.6/§11.3 "after 1.0"): triangles
  counted by MC cases on ≤ 512 sampled bricks, 0.5 Å pilot classification for finer grids,
  costs 16 MiB + 1 200 B/sphere + 150 B/triangle (+50 decimation, +50 smoothing) + 100 B per
  touched brick; lower transients (arrangement waves, packed touched keys, field freed before
  assembly, decimation corner lists). Measure memory with a counting global allocator in a
  scratch crate outside the workspace (`malloc_size` + `getrusage` per `Progress` stage).
- Phase 9 (docs/validation-report.md): 53-entry accuracy study (area −0.20 %, volume −0.02 % at
  0.25 Å vs h → 0 extrapolation), density constant 1.4999 ± 0.0025, 200/200 sweep ok (after
  1.1.1 also `sweep1000`: 1 000/1 000, #23), large
  inputs (3J3Q 1 Å: 48.7 M triangles, 70 s, 5.3 GB). Open: cross-tool references (#5, #6), NFR-ACC
  decision (#4), NFR-PERF targets missed ≤ 2× on this machine, per-atom memory ≈ 600–800 B.

- Phase 11 (GPU back end, HIP, #24; branch `phase-11-gpu-hip`, not merged or released yet):
  T11.0–T11.2, S1, T11.3′ (exact device sampling, byte-identical), S2, T11.5′ (`--refine fast`,
  f32 refinement on the device), T11.6′ (benchmarks, batch tuning, docs, `tools/gpu/`, release
  checklist) done 2026-11-21. Zen 5 + Radeon 8060S, SES: exact 1.23–1.44×, fast 1.98–2.47× the
  16 cores (1AON 0.25 Å 3.40 / 2.37 / 1.41 s; 3J3Q 1 Å 28.9 / 23.6 / 12.3 s); results
  `docs/05-gpu-hip.md` §12. Fast mode is the "quick preview" mode; its speed vs accuracy was
  reviewed by the user at the end of T11.5′ (kept as is: no accuracy given up). Planned
  release: 1.2.0 (new opt-in feature, minor), first publish of `sesmesh-gpu`.

## Next

Work after 1.0 is collected in the tracking issue #19 ("Roadmap after v1.0"). Labels: `type:`
(feature, performance, validation, ci, docs, decision, tracking), `area:` (io, field, mesh,
engine, export, cli, ci), `phase:` (8, 9, 10, backlog; deferred items carry `phase: backlog`),
`priority:` (must/should/could = spec MoSCoW). Close issues from commits/PRs (`Fixes #N`) and
tick them in #19. Summary:

1. Fuzzing campaign ≥ 24 CPU-hours per target (#22) done 2026-11-14 with `fuzz/campaign.sh`
   on the two comparison hosts (podman, toolchain and corpora in `~/sesmesh-fuzz` there):
   ≥ 24.9 CPU-h per target, 174.5 in total, results in `docs/validation/fuzz.csv`. It found
   one bug, an extended-XYZ `Properties=` column-count overflow (a panic), fixed after 1.1.1
   with a regression test in `io::xyz` and released in 1.1.2, the version the paper
   describes (1.1.1 measurements hold: only `io/xyz.rs` differs). The 1 000-entry sweep
   (#23) is done too (`xtask sweep --set sweep1000`, 1 000/1 000 ok).
2. Cross-tool comparison (#5, #6) done 2026-11-14, merged into `main`:
   `tools/compare/` (README there), `docs/comparison.md`, `docs/validation/compare/`,
   `tests/data/reference/*.toml`, `tests/references.rs`. Close #5/#6 (and report #8/#10
   numbers). Paper draft in `paper/` (`make`; README lists the open items).
3. Rare 1.1.0 refinement outliers on non-monotone edges, found by the comparison
   (`docs/validation/compare/runs.csv`, campaigns `acc110`/`sw110`, sesmesh `dist_max`):
   0.22 Å on 1RCL, 0.19 Å 2IEM, 0.17 Å 1GZM, 0.16 Å 8Y7M (all at 0.5 Å), 0.094 Å 1AON at
   0.5 Å, 0.083 Å 6VXX at 0.25 Å. Fixed in d358dcf + c97a0bb, released in 1.1.1. The sesmesh comparison runs
   were repeated with 1.1.1 on this Mac (campaigns `acc`/`sw`/`synth`/`det`/`large`; 1.1.0
   kept as `*110`, `tools/compare/README.md`); the paper describes 1.1.2 (§3.5, §7).
   Still to do: the site's accuracy wording ("≤ 1e-4·h" holds near the origin; the edge
   clamp is ≥ 4 f32 ulps farther out). arm64 and x86-64 builds write files that differ
   in the last bits (same counts and areas); determinism is per binary.
4. x86 kernel work (#9, #13), Ubuntu 26 CI image (#18), optional marching tetrahedra (#14).
   The 8-P-core Apple reference timings were dropped by decision (2026-11-15, Spec §5.1).
   #8 and #10 measured and closed (2026-11-15; results in Design §7.7, §11.1).
5. Ambient occlusion (#27) done 2026-11-15: `tools/ao/` plus tutorial 04, merged into
   `main`. Upstream fixes released the same day (user's repos `~/projects/objserde`,
   `~/projects/aomesh`): objserde 0.3.0 (keeps header comments, `o`, `g`) and aomesh 0.4.0
   (plain `map_Kd`, passes them through; AO values byte-identical to 0.3.1), both tagged
   with GitHub releases. Left open in #27: MeshLab check, optional gallery AO toggle.
   GPU back ends: #24 HIP (Phase 11, done on its branch; open after it: GPU buffers in the
   memory estimate, discrete GPUs, CPU/GPU edge split in fast mode, `docs/05-gpu-hip.md`
   §12), #25 CUDA, #26 Metal (f32 plan in the issues).

## Gotchas learned

- The comparison ran on two identical 16-core Zen 5 Linux machines with podman (see
  `tools/compare/README.md`; work dir `~/sesmesh-bench`, image `localhost/sesmesh-compare`).
  Ask the user before using them; never publish the image (MSMS and ChimeraX licences forbid
  redistribution).
- macOS `openrsync` silently transfers nothing with `-c`; use `rsync -a` and touch Rust sources
  before rebuilding on the hosts (rsync keeps older mtimes than the last build). Start
  detached remote jobs with `ssh -f host 'setsid nohup … &'`, else ssh never returns.
- Compared programs' quirks: ESES reads only `ATOM` records by fixed columns and aborts on atom
  names not starting with C/H/O/N/P/S; NanoShaper 0.7.5 needs `#include <memory>` (GCC 13),
  `-DENABLE_EXE=ON`, and writes a date into its OFF header; MSMS writes component k to
  `mesh_k.vert/.face`; ChimeraX reads no PQR (set radii on the atoms) and writes unshared
  vertices per patch.

- GPU (Phase 11): the gfx1151 machine's default Rust toolchain is 1.88 (below the MSRV): use
  `cargo +stable` there, with `~/.cargo/bin` on `PATH` in non-interactive ssh. `hipcc --genco`
  writes a clang offload bundle, not an ELF: inspect it with `/opt/rocm/lib/llvm/bin/
  clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--<target>`, then
  `llvm-readelf --notes` (`.wavefront_size`). Check GPU load (`gpu_busy_percent`) before timing.
- Timing on this machine is noisy (Defender and other apps; load average often 10–25): write
  outputs to `/dev/null`, check `uptime`, compare with criterion baselines
  (`--save-baseline` / `--baseline`), repeat runs. Single-thread runs are steadier.
- zsh does not word-split `$var`; use `${=var}` in loops. In zsh `"$h:path"` applies a `:`
  modifier to `$h` (scp then writes a local file): write `"${h}:path"`. `podman rm a b c`
  removes nothing when one name is missing; remove containers one by one. Foreground
  `sleep` is blocked; run long jobs in the background.
- The nightly fuzz jobs must pass `cargo fuzz run --target x86_64-unknown-linux-gnu`: the
  prebuilt cargo-fuzz from `taiki-e/install-action` is a musl build and defaults to the musl
  target, where AddressSanitizer fails ("sanitizer is incompatible with statically linked libc").
- CI's clippy job uses the latest stable Rust: keep the local `stable` toolchain current, or new
  lints break `main` (Phase 8: `needless_range_loop` on 1.100, unseen on 1.99).
- Clippy here is strict (pedantic subset): `is_multiple_of`, method paths instead of closures,
  `;` after unit expressions, `clamp`. Code under `#[cfg(target_arch = "x86_64")]` is only
  linted by the x86 target command above.
- A surface brick may own no crossing edge yet have crossing cells: the cell pass must always
  run (Phase 4 hole bug). MC triangulations must not put a diagonal inside a cube face (the
  neighbouring cell duplicates it). `Mesh` positions are f32 in the input frame: keep vertex
  separations above a few f32 ulps (`edge_clamp`).
- SES sheets thinner than h (two probe balls from opposite sides of an atom triangle almost
  meeting) give MC "sub-grid fragments": solid components without a convex vertex, mostly
  8-triangle octahedra around one node (4HHB 0.25 Å: one; validation corpus 0.25 Å: 1 358).
  Dropped on request (`remove_fragments`, `--drop-fragments`, Design §9.1; opt-in in 1.x to keep
  default output); real solid components always have convex patches (≥ 40 Å² in the corpus).
- `out/` holds `1crn-ses-*` prototype meshes (marching tetrahedra) and `*-sesmesh.obj` from the
  real CLI for comparison.
