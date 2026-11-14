# Release process (Plan T10.4)

Publishable crates: `sesmesh` (library) and `sesmesh-cli` (binary
`sesmesh`), published in that order. `sesmesh-testkit` and `xtask` are
private (`publish = false`).

## v1.0 criteria (Spec §6.2) — status at 1.0.0 (2026-11-13)

| Criterion | Status |
|---|---|
| All Must and Should requirements implemented | Met: decimation (FR-LOD-4, #1), auto-coarsening (FR-LOD-6, #3); NFR-MEM met with the per-atom budget revised to 800 B (#21) |
| NFR-PERF met or revised with justification | Revised to the Phase 9 measurements on 10-core Apple Silicon (#20); x86 reference measurements after 1.0 (#8, #10; 16-core Zen 5 measured 2026-11-14, `docs/comparison.md`) |
| NFR-ACC met on the validation corpus, report published | Met against internal references (`docs/validation-report.md`); analytic bound revised (#4); cross-tool references after 1.0 (#5, #6; done 2026-11-14, `docs/comparison.md`) |
| Zero robustness failures; ≥ 24 CPU-hours fuzzing per parser | Sweep 200/200; 10 minutes per fuzz target (7 targets, ≈ 18 M inputs, 0 failures); 24 CPU-hour campaign after 1.0 (#22) |
| API docs, examples, CLI manual, changelog; crates published | Met |

## Local validation

Run before every push, not only releases; push only when all pass. The same
steps as `.github/workflows/ci.yml`, which runs only by hand:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo clippy -p sesmesh -p sesmesh-cli --lib --bins --target x86_64-unknown-linux-gnu --locked -- -D warnings
cargo test --workspace --locked
cargo test --workspace --locked --release -- --include-ignored
for e in file_to_obj spheres_to_mesh trajectory custom_radii field_queries; do cargo run -q --release --locked -p sesmesh --example $e; done
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
cargo +1.89 check --package sesmesh --package sesmesh-cli --locked
cargo deny check
cargo publish --workspace --dry-run --locked
```

This covers the host only (macOS arm64 here). Linux x86-64, Linux arm64 and
Windows are covered by running the CI workflow by hand, or by testing on
such a machine.

## Releases

| Version | Date | Notes |
|---|---|---|
| 1.1.1 | 2026-11-14 | Exact refinement bisects edges where Newton/Illinois stalled (rare vertices up to 0.22 Å off the SES in 1.1.0; all others bit-identical), cross-tool comparison (#5, #6) and stored references, paper draft, crate homepage/category, `CITATION.cff`. |
| 1.1.0 | 2026-11-14 | Opt-in sub-grid fragment removal (`--drop-fragments`), memory guard fixed (estimate now an upper bound of the peak) and lower peak memory, CLI creates missing output directories, project website on GitHub Pages; default meshes unchanged. |
| 1.0.0 | 2026-11-13 | First stable release: decimation, auto-coarsening, gzip PLY/MSMS, completions and man page, Ctrl-C, lower memory; spec criteria revised (NFR-MEM, NFR-ACC, NFR-PERF). |
| 0.9.1 | 2026-11-13 | CLI: quiet exit when standard output is closed early (broken pipe). |
| 0.9.0 | 2026-11-13 | First public release (release candidate for 1.0): crates `sesmesh` and `sesmesh-cli` on crates.io, binaries on the GitHub release `v0.9.0`. |

## Checklist

1. Local validation passes on the commit to release, before anything is
   pushed (GitHub CI is manual-only; see *Local validation* below).
2. Set the version in `Cargo.toml` (`[workspace.package] version` and the
   `sesmesh` entry of `[workspace.dependencies]`), run
   `cargo update -w`. Also set `version` and `date-released` in
   `CITATION.cff` and the version shown on the website
   (`grep -n 'X.Y.Z' site/*.html` with the old version: the header badges, the
   home page label and `softwareVersion` in its JSON-LD block).
3. In `CHANGELOG.md`, rename *Unreleased* to `## [X.Y.Z] - YYYY-MM-DD` and
   open a new *Unreleased* section.
4. Dry runs:

   ```bash
   cargo publish --workspace --dry-run
   ```

   and the `release` workflow by hand (Actions → release → Run workflow,
   `publish-crates` unticked): builds and packages the binaries for the four
   tier-1 targets as artifacts.
5. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "sesmesh X.Y.Z"
   ```

   ```bash
   git push origin main vX.Y.Z
   ```

   The workflow creates a **draft** GitHub release with the archives
   (`sesmesh-X.Y.Z-<target>.tar.gz|zip`, each with the binary, README,
   CHANGELOG, licences and `docs/cli.md`) and `SHA256SUMS`. Review and
   publish the draft.
6. crates.io: add a `CARGO_REGISTRY_TOKEN` secret to the `crates-io`
   environment (repository settings), then run the workflow by hand with
   `publish-crates` ticked — or run `cargo publish --workspace` locally.

Binaries use the `dist` profile (fat LTO, one codegen unit, stripped) and
runtime SIMD dispatch, so one x86-64 binary uses AVX-512/AVX2/SSE4.2 as
available.
