# Release process (Plan T10.4)

Publishable crates: `sesmesh-gpu` (GPU back ends, an implementation
crate that `sesmesh` requires at the exact same version), `sesmesh`
(library) and `sesmesh-cli` (binary `sesmesh`), published in that order
(`cargo publish --workspace` orders them). `sesmesh-testkit` and `xtask` are
private (`publish = false`). `sesmesh-gpu` is new with Phase 11: check that
its name is still free on crates.io before its first release.

## v1.0 criteria (Spec §6.2) — status at 1.0.0 (2026-11-13)

| Criterion | Status |
|---|---|
| All Must and Should requirements implemented | Met: decimation (FR-LOD-4, #1), auto-coarsening (FR-LOD-6, #3); NFR-MEM met with the per-atom budget revised to 800 B (#21) |
| NFR-PERF met or revised with justification | Revised to the Phase 9 measurements on 10-core Apple Silicon (#20); x86 reference measurements after 1.0 (#8, #10; 16-core Zen 5 measured 2026-11-14, `docs/comparison.md`); reference machines revised to the two measured ones after 1.1.2 (2026-11-15, Spec §5.1) |
| NFR-ACC met on the validation corpus, report published | Met against internal references (`docs/validation-report.md`); analytic bound revised (#4); cross-tool references after 1.0 (#5, #6; done 2026-11-14, `docs/comparison.md`) |
| Zero robustness failures; ≥ 24 CPU-hours fuzzing per parser | Sweep 200/200; 10 minutes per fuzz target (7 targets, ≈ 18 M inputs, 0 failures); 24 CPU-hour campaign after 1.0 (#22; done after 1.1.1: ≥ 24.9 CPU-hours per target, one XYZ overflow fixed (released in 1.1.2), then 0 failures; 1 000-entry sweep #23 1 000/1 000, `docs/validation-report.md` §4, §7) |
| API docs, examples, CLI manual, changelog; crates published | Met |

## Local validation

Run before every push, not only releases; push only when all pass. Nothing
runs on GitHub: the workflows in `.github/workflows/` start only by hand and
are kept for reference (`ci.yml` lists the same steps):

```bash
cargo fmt --all --check
SESMESH_GPU_NO_KERNELS=1 cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo clippy -p sesmesh -p sesmesh-cli --lib --bins --target x86_64-unknown-linux-gnu --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --locked --release -- --include-ignored
for e in file_to_obj spheres_to_mesh trajectory custom_radii field_queries; do cargo run -q --release --locked -p sesmesh --example $e; done
SESMESH_GPU_NO_KERNELS=1 RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
cargo +1.89 check --package sesmesh --package sesmesh-cli --locked
cargo deny check
cargo publish --workspace --dry-run --locked
python3 -m unittest discover -s tools/ao
```

`SESMESH_GPU_NO_KERNELS=1` lets `--all-features` build `sesmesh-gpu`'s `hip`
and `cuda` features without ROCm or CUDA (no kernels are compiled;
docs/05-gpu-hip.md §9). The GPU tests skip without a device unless
`SESMESH_GPU_REQUIRE=1`.

### GPU hosts

Before a release that changes `sesmesh-gpu`, the field, meshing or the
GPU path, also run on a machine with an AMD GPU and ROCm (tested: Radeon
8060S, gfx1151, ROCm 7.16):

```bash
export PATH=$HOME/.cargo/bin:/opt/rocm/bin:$PATH
F=sesmesh-cli/hip,sesmesh-gpu/hip
cargo +stable clippy --workspace --all-targets --features $F --locked -- -D warnings
SESMESH_GPU_REQUIRE=1 cargo +stable test --workspace --release --features $F --locked -- --include-ignored
cargo +stable build --release -p sesmesh-cli --features hip --locked
tools/gpu/gate.sh /tmp/gate /tmp/gate-inputs.txt
tools/gpu/gate.sh /tmp/gate-sv /tmp/gate-inputs.txt sas:1 sas:0.5 vdw:1 vdw:0.5
```

and on a machine with an NVIDIA GPU (tested: GeForce RTX 2070 Max-Q,
sm_75, driver for CUDA 13.4), with `F=sesmesh-cli/cuda,sesmesh-gpu/cuda`,
the same clippy and test commands, a build with `--features cuda`, and
the gate with `BACKEND=cuda` (`docs/06-gpu-cuda.md` §3–§4: the build can
run in `tools/gpu/cuda-build.Containerfile` on another machine, and the
gate's CPU references too, with `MODE=cpu` / `MODE=gpu`). A host with
both toolkits can use `--all-features`.

The clippy runs compile the kernels with `hipcc` or `nvcc`. The tests
include the device tests (exact device sampling equals the CPU's, fast
meshes equal the exact triangles, batch sizes, cancellation) and the CLI's
`--backend hip|cuda` / `--refine fast` tests. `tools/gpu/gate.sh` compares
the CPU and GPU back ends on the test corpus (`tools/gpu/README.md`:
inputs, checks, run time). The GPU builds are not in the release binaries:
users build them with `cargo install sesmesh-cli --features hip` or
`--features cuda`.

The Mac covers macOS arm64, the GPU hosts Linux x86-64. Linux arm64 and
Windows x86-64 are tested only on such a machine, when one is available.

## Releases

| Version | Date | Notes |
|---|---|---|
| 1.3.1 | 2026-11-21 | README without the CI badge (crates.io pages); cites version 1.3.1 (Zenodo doi:10.5281/zenodo.24412232). Code and output unchanged from 1.3.0. |
| 1.3.0 | 2026-11-21 | GPU back end for NVIDIA GPUs (CUDA, feature `cuda`; same kernels and pipeline as HIP, batches in device memory; exact mode byte-identical to the CPU); `tools/gpu/gate.sh` can take its CPU references from another machine; default meshes unchanged. Binaries for macOS arm64 and Linux x86-64 only, without GPU features. |
| 1.2.0 | 2026-11-21 | GPU back end for AMD GPUs (HIP, feature `hip`, crate `sesmesh-gpu`; exact mode byte-identical to the CPU) and opt-in fast mode (`--refine fast`); default meshes unchanged. First release built and uploaded locally (no workflow runs); binaries for macOS arm64 and Linux x86-64 only. |
| 1.1.2 | 2026-11-15 | Extended XYZ: overflowing `Properties=` column counts are an input error instead of a panic (found by the 24 CPU-hour fuzzing campaign, #22); 1 000-entry robustness sweep (#23), `fuzz/campaign.sh`; default meshes unchanged. |
| 1.1.1 | 2026-11-14 | Exact refinement bisects edges where Newton/Illinois stalled (rare vertices up to 0.22 Å off the SES in 1.1.0; all others bit-identical), cross-tool comparison (#5, #6) and stored references, paper draft, crate homepage/category, `CITATION.cff`. |
| 1.1.0 | 2026-11-14 | Opt-in sub-grid fragment removal (`--drop-fragments`), memory guard fixed (estimate now an upper bound of the peak) and lower peak memory, CLI creates missing output directories, project website on GitHub Pages; default meshes unchanged. |
| 1.0.0 | 2026-11-13 | First stable release: decimation, auto-coarsening, gzip PLY/MSMS, completions and man page, Ctrl-C, lower memory; spec criteria revised (NFR-MEM, NFR-ACC, NFR-PERF). |
| 0.9.1 | 2026-11-13 | CLI: quiet exit when standard output is closed early (broken pipe). |
| 0.9.0 | 2026-11-13 | First public release (release candidate for 1.0): crates `sesmesh` and `sesmesh-cli` on crates.io, binaries on the GitHub release `v0.9.0`. |

## Checklist

Nothing in this list runs on GitHub: the binaries are built, tested and
packaged locally, and pushes and tags start no workflow.

1. Local validation passes on the commit to release, before anything is
   pushed (*Local validation* above, and the *GPU hosts* steps when they
   apply).
2. Set the version in `Cargo.toml` (`[workspace.package] version` and the
   `sesmesh` and `sesmesh-gpu` (`=x.y.z`) entries of
   `[workspace.dependencies]`), run
   `cargo update -w`. Also set `version` and `date-released` in
   `CITATION.cff` and the version shown on the website
   (`grep -n 'X.Y.Z' site/*.html` with the old version: the header badges, the
   home page label and `softwareVersion` in its JSON-LD block).
3. In `CHANGELOG.md`, rename *Unreleased* to `## [X.Y.Z] - YYYY-MM-DD`,
   open a new *Unreleased* section and update the comparison links at the
   end. Add a row to *Releases* above.
4. Commit, run the local validation again (it includes
   `cargo publish --workspace --dry-run --locked`), then tag:

   ```bash
   git tag -a vX.Y.Z -m "sesmesh X.Y.Z"
   ```

5. Build and package the binaries, one archive per target, on a machine
   of that kind (or a container with an older glibc for Linux, so that the
   binary runs on older distributions):

   ```bash
   tools/release/package.sh aarch64-apple-darwin dist
   ```

   The script builds with the `dist` profile, smoke-tests the binary
   (`--version`, 1CRN draft) and writes
   `dist/sesmesh-X.Y.Z-<target>.tar.gz` (`.zip` for Windows), with the
   binary, README, CHANGELOG, licences, `docs/cli.md`,
   `docs/04-output-formats.md`, the man page and the shell completions,
   and a `.sha256` file. Collect the archives in one directory and write
   `SHA256SUMS` (`cat *.sha256 > SHA256SUMS`).
6. Push the branch, `main` and the tag:

   ```bash
   git push origin main vX.Y.Z
   ```

7. crates.io (`sesmesh-gpu`, then `sesmesh`, then `sesmesh-cli`; the token
   is in `~/.cargo/credentials.toml`):

   ```bash
   cargo publish --workspace --locked
   ```

8. GitHub release, created through the API (no workflow):

   ```bash
   gh release create vX.Y.Z --verify-tag --title "sesmesh X.Y.Z" \
     --notes "See CHANGELOG.md for the changes in this release." \
     dist/*.tar.gz dist/*.zip dist/SHA256SUMS
   ```

9. Zenodo (by hand; keep Zenodo's GitHub integration off so that every
   version stays under the one record): on https://doi.org/10.5281/zenodo.24238836 choose **New version**,
   upload the release's source archive (GitHub release → *Source code
   (tar.gz)*), set the version and date, publish. In the next commit, cite
   the new version everywhere: `CITATION.cff` (`doi` and a new
   `identifiers` entry), the reference and BibTeX in `README.md` ("Citing
   sesmesh") and in `site/library.html` (`#cite`): version, year and version
   DOI. The all-versions DOI (10.5281/zenodo.24238836) stays.
10. Website: the `pages` workflow deploys `site/` from `main` to GitHub
    Pages. It starts only by hand (Actions → pages → Run workflow), at the
    maintainer's decision.

Binaries use the `dist` profile (fat LTO, one codegen unit, stripped) and
runtime SIMD dispatch, so one x86-64 binary uses AVX-512/AVX2/SSE4.2 as
available.
