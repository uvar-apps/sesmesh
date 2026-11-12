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
