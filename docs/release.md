# Release process (Plan T10.4)

Publishable crates: `sesmesh` (library) and `sesmesh-cli` (binary
`sesmesh`), published in that order. `sesmesh-testkit` and `xtask` are
private (`publish = false`).

## v1.0 criteria (Spec §6.2) — status at 1.0.0 (2026-11-13)

| Criterion | Status |
|---|---|
| All Must and Should requirements implemented | Met: decimation (FR-LOD-4, #1), auto-coarsening (FR-LOD-6, #3); NFR-MEM met with the per-atom budget revised to 800 B (#21) |
| NFR-PERF met or revised with justification | Revised to the Phase 9 measurements on 10-core Apple Silicon (#20); x86 reference measurements after 1.0 (#8, #10) |
| NFR-ACC met on the validation corpus, report published | Met against internal references (`docs/validation-report.md`); analytic bound revised (#4); cross-tool references after 1.0 (#5, #6) |
| Zero robustness failures; ≥ 24 CPU-hours fuzzing per parser | Sweep 200/200; 10 minutes per fuzz target (7 targets, ≈ 18 M inputs, 0 failures); 24 CPU-hour campaign after 1.0 (#22) |
| API docs, examples, CLI manual, changelog; crates published | Met |

## Releases

| Version | Date | Notes |
|---|---|---|
| 1.0.0 | 2026-11-13 | First stable release: decimation, auto-coarsening, gzip PLY/MSMS, completions and man page, Ctrl-C, lower memory; spec criteria revised (NFR-MEM, NFR-ACC, NFR-PERF). |
| 0.9.1 | 2026-11-13 | CLI: quiet exit when standard output is closed early (broken pipe). |
| 0.9.0 | 2026-11-13 | First public release (release candidate for 1.0): crates `sesmesh` and `sesmesh-cli` on crates.io, binaries on the GitHub release `v0.9.0`. |

## Checklist

1. Green CI on `main` (tests on all tier-1 targets, clippy on the latest
   stable, MSRV, rustdoc, cargo-deny).
2. Set the version in `Cargo.toml` (`[workspace.package] version` and the
   `sesmesh` entry of `[workspace.dependencies]`), run
   `cargo update -w`.
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
