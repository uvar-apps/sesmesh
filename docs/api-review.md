# API review (Plan T10.3)

Review of the `sesmesh` public API against the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/checklist.html),
2026-11-13, before the first release.

| Guideline | Status | Notes |
|---|---|---|
| C-CASE, C-CONV, C-GETTER, C-ITER naming | ✓ | `snake_case` functions, `with_*` builders, getters without `get_` (`SesField::kind`, `SurfaceEngine::params`). |
| C-COMMON-TRAITS | ✓ | Every public type implements `Debug` (checked with `-W missing-debug-implementations`); value types derive `Clone`/`PartialEq` (and `Copy`/`Eq`/`Hash` where meaningful); options implement `Default`. |
| C-SEND-SYNC | ✓ | `tests/api.rs` asserts `Send + Sync` for the main types (`Sphere`, `SurfaceParams`, `SurfaceEngine`, `SurfaceOutput`, `Mesh`, `SesField`, `Structure`, `Diagnostics`, `Error`, `Control`, option structs). |
| C-GOOD-ERR | ✓ | One `Error` enum (`thiserror`), `std::error::Error + Send + Sync + 'static`, `Display` with file/line/column for parse errors, `source()` for I/O; `#[non_exhaustive]`. |
| C-CRATE-DOC, C-EXAMPLE | ✓ | Crate guide (concepts, modules, features, versioning) with runnable examples; five `examples/` run in CI; most user-facing items have doctests. |
| C-FAILURE | ✓ | Every fallible public function has `# Errors`; possible panics are documented under `# Panics` (checked with `clippy::missing_errors_doc`/`missing_panics_doc`). |
| C-LINK | ✓ | Intra-doc links; rustdoc builds with `-D warnings`. |
| C-METADATA | ✓ | `description`, `license`, `repository`, `keywords`, `categories`, `readme`, `rust-version`. |
| C-CTOR, C-BUILDER | ✓ | `SurfaceParams::{ses, sas, vdw}` + `#[must_use]` `with_*` builders; `Control::new().with_cancel().with_progress()`. |
| C-GENERIC, C-OBJECT | ✓ | Readers take `impl Read`/`AsRef<Path>`/`&[u8]`; writers take `impl Write`; the progress callback is `&dyn Fn(Progress) + Sync`. |
| C-NEWTYPE, C-CUSTOM-TYPE | ✓ | `Element`, `Sym`, enums for modes (`CavityMode`, `NormalMode`, `RefineMode`, `SurfaceKind`, `SimdLevel`, `Quality`). |
| C-VALIDATE | ✓ | `SurfaceParams::validate`; invalid input returns `Error::InvalidInput` (no panics on input; fuzzed parsers). |
| C-STRUCT-PRIVATE, C-STRUCT-BOUNDS | ✓ | Engine and field types have private fields; no trait bounds on struct definitions. |
| C-SEALED | n/a | No public traits. |
| C-STABLE | ✓ | No third-party type in the public API (Spec FR-LIB-6); pre-1.0 dependencies (`fast-float2`, `toml`) are internal. |
| C-PERMISSIVE | ✓ | MIT OR Apache-2.0; dependency licenses checked by `cargo deny`. |

## `#[non_exhaustive]` audit

* **Non-exhaustive** (may grow without a breaking change): all option and
  result structs with public fields (`SurfaceParams`, `SurfaceStats`,
  `SurfaceOutput`, `Timings`, `Mesh`, `Component`, `Validation`, `ObjOptions`,
  `PlyOptions`, `MsmsHeader`, `ReadOptions`, `RadiusOptions`, `Selection`,
  `AtomRecord`, `Model`, `Structure`, `Source`, `FieldSample`, `Progress`,
  `AssemblySummary`, `Warning`, `ArrangementStats`) and all enums except
  `PatchKind` (`Error`, `WarningKind`, `Format`, `SurfaceKind`, `SimdLevel`,
  `Quality`, `Resolution`, `RefineMode`, `CavityMode`, `NormalMode`,
  `Stage`, `Patch`, `ModelChoice`, `AltLocPolicy`, `ChainIdKind`,
  `RadiusSet`, `BuiltinRadii`).
* **Exhaustive by design:** `Sphere` (centre + radius, built literally),
  `PatchKind` (the three SES patch types are fixed by the geometry).
* **Private fields** (cannot be built by users, so they can grow freely):
  `SesField`, `SurfaceEngine`, `Control`, `Diagnostics`, `RadiusRules`,
  `ElementPalette`, `StringPool`, `Sym`, `Element`, `AtomSet`.
* Changed in this review: `Warning` became `#[non_exhaustive]`.

## Not adopted

* `clippy::must_use_candidate` (pedantic): 246 getters would gain
  `#[must_use]`; builders and pure constructors already have it where
  dropping the result is a likely mistake.

## Semver policy

* `0.x` (0.9.0, 0.9.1): minor versions could break the API; patch versions
  did not.
* From `1.0` (1.0.0, 2026-11-13): semantic versioning. Adding fields to non-exhaustive structs,
  variants to non-exhaustive enums, functions and features is a minor change;
  removing or renaming public items, changing signatures or default outputs
  (mesh geometry for the same parameters) is a major change.
* MSRV (currently 1.89) may rise in minor releases, announced in the
  changelog; it is checked in CI.
* Output determinism (identical OBJ bytes for any thread count) is part of
  the contract; output changes between versions are listed in the changelog.
