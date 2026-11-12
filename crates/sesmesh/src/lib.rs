//! # sesmesh
//!
//! Solvent Excluded Surface (SES, also called the Connolly or molecular
//! surface) meshing for molecular structures read from PDB, PDBx/mmCIF and XYZ
//! files.
//!
//! **Status:** project scaffold (Phase 0). No functionality is implemented
//! yet. The specification, design and implementation plan live in the
//! repository's `docs/` directory.

// The library reports through return values and `log`, never by printing
// (Spec FR-DIAG-1).
#![deny(clippy::print_stdout, clippy::print_stderr)]

/// Version of the `sesmesh` library, taken from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
