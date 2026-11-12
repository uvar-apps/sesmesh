//! # sesmesh
//!
//! Solvent Excluded Surface (SES, also called the Connolly or molecular
//! surface) meshing for molecular structures read from PDB, PDBx/mmCIF and XYZ
//! files.
//!
//! **Status:** Phases 1–2 — readers ([`io`]), atom selection
//! ([`Selection`]), radius assignment ([`radii`]) and the sphere geometry
//! behind the surface ([`Sphere`], [`geom`]) are implemented. The surface
//! field and meshing follow in later phases; see the repository's `docs/`
//! directory.
//!
//! ```
//! use sesmesh::{io, radii, Diagnostics, Selection};
//!
//! let cif = "data_x\nloop_\n_atom_site.type_symbol\n_atom_site.Cartn_x\n\
//!            _atom_site.Cartn_y\n_atom_site.Cartn_z\nC 0 0 0\nO 1.2 0 0\n";
//! let structure = io::read_bytes(cif.as_bytes(), &io::ReadOptions::default())?;
//! let mut diags = Diagnostics::new();
//! let atoms = structure.select(&Selection::default(), &mut diags)?;
//! let radii = radii::assign(&atoms, &radii::RadiusOptions::default(), &mut diags)?;
//! assert_eq!(radii, [1.70, 1.52]);
//! # Ok::<(), sesmesh::Error>(())
//! ```

// The library reports through return values, never by printing
// (Spec FR-DIAG-1).
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod diag;
pub mod error;
pub mod geom;
pub mod io;
pub mod radii;
pub mod structure;

pub use diag::{Diagnostics, Warning, WarningKind};
pub use error::{Error, Result};
pub use geom::Sphere;
pub use structure::{AtomRecord, AtomSet, Element, Model, Selection, Structure};

/// Version of the `sesmesh` library, taken from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Entry points for fuzz targets (`cargo fuzz` builds with `--cfg fuzzing`).
#[cfg(fuzzing)]
#[doc(hidden)]
pub mod fuzzing {
    /// Tokenizes `data` as CIF and returns the number of tokens read before
    /// the end or the first error.
    pub fn cif_tokens(data: &[u8]) -> usize {
        crate::io::cif::lexer::Lexer::new(data)
            .take_while(Result::is_ok)
            .count()
    }
}
