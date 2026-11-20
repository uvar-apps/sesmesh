# Contributing to sesmesh

Work follows the [implementation plan](docs/03-implementation-plan.md). Each
change maps to one or more task IDs (for example `T1.3`) and the requirement
IDs it implements (`FR-*`, `NFR-*` from the
[specification](docs/01-product-specification.md)).

## Definition of done

A task is done when:

1. Code is formatted (`cargo fmt --all`) and `SESMESH_GPU_NO_KERNELS=1 cargo clippy --workspace --all-targets --all-features -- -D warnings` is clean (the variable builds the `hip` feature of `sesmesh-gpu` without ROCm).
2. There is no `unsafe` code (enforced by `unsafe_code = "forbid"`), except the GPU FFI in `crates/sesmesh-gpu` (`unsafe_code = "deny"`, allowed per module, every block with a `// SAFETY:` comment).
3. Every public item is documented (`missing_docs = "deny"`), with an example
   when it is user-facing.
4. Unit tests (and property tests where the plan asks for them) pass on all
   CI targets.
5. Hot-path changes come with a criterion benchmark, and the PR states the
   results.
6. `CHANGELOG.md` has an entry under *Unreleased*.
7. The PR description lists the task and requirement IDs.

## Local checks

Run these before opening a PR (CI runs the same):

```bash
cargo fmt --all --check
```

```bash
SESMESH_GPU_NO_KERNELS=1 cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
```

```bash
cargo test --workspace --locked
```

```bash
SESMESH_GPU_NO_KERNELS=1 RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
```

```bash
cargo deny check
```

## Conventions

* **MSRV**: `rust-version` in the workspace `Cargo.toml` (currently 1.89) is
  the highest MSRV among the planned dependencies. Raise it only when a
  dependency requires it, and update the `msrv` CI job in the same change.
* **Dependencies** follow the table in Plan §2. Declare them in
  `[workspace.dependencies]` and add them only in the phase that first uses
  them. No third-party types in the public API (Design §1).
* **The library never prints.** It returns diagnostics and logs through `log`.
  Only the CLI writes to stdout or stderr.
* **Determinism**: output must not depend on thread count or scheduling
  (Spec NFR-DET). Collect parallel results in index order, and sum floats in a
  fixed order.
* **Test data**: see [tests/data/README.md](tests/data/README.md). Tests that
  need downloaded files must skip cleanly when the cache is empty.
* Licensing: contributions are dual-licensed MIT OR Apache-2.0 (see README).
* **File headers**: every source file (Rust, Python, shell, JavaScript, CSS,
  HTML, TOML, workflow YAML) starts with the project header, after a shebang
  if there is one and before any `#![…]` attribute:

  ```rust
  // sesmesh — Solvent Excluded Surface meshes for molecules
  // Copyright (c) 2026 Ugo Varetto
  // SPDX-License-Identifier: MIT OR Apache-2.0
  ```

  Generated files get it from their generator (`mesh/tables.rs` writes it
  into `mc_table.rs`).
