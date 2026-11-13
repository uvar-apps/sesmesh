// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # sesmesh
//!
//! Solvent Excluded Surface (SES, also called the Connolly or molecular
//! surface) meshing for molecular structures read from PDB, PDBx/mmCIF, XYZ
//! and PQR files: closed, consistently oriented triangle meshes with exact
//! analytic normals, multi-threaded and SIMD-accelerated.
//!
//! ## Quick start
//!
//! ```
//! use sesmesh::{compute_surface, mesh, Sphere, SurfaceParams};
//!
//! let atoms = [Sphere::new([0.0, 0.0, 0.0], 1.7), Sphere::new([3.0, 0.0, 0.0], 1.7)];
//! let out = compute_surface(&atoms, &SurfaceParams::ses(1.4).with_spacing(0.3))?;
//! assert!(mesh::validate(&out.mesh).is_ok()); // closed, oriented, manifold
//! assert_eq!(out.stats.components, 1);
//! # Ok::<(), sesmesh::Error>(())
//! ```
//!
//! From a file (readers, selection, radii):
//!
//! ```
//! use sesmesh::{io, radii, Diagnostics, Selection, Sphere, SurfaceParams, compute_surface};
//!
//! let cif = "data_x\nloop_\n_atom_site.type_symbol\n_atom_site.Cartn_x\n\
//!            _atom_site.Cartn_y\n_atom_site.Cartn_z\nC 0 0 0\nO 1.2 0 0\n";
//! let structure = io::read_bytes(cif.as_bytes(), &io::ReadOptions::default())?;
//! let mut diags = Diagnostics::new();
//! let atoms = structure.select(&Selection::default(), &mut diags)?;
//! let radii = radii::assign(&atoms, &radii::RadiusOptions::default(), &mut diags)?;
//! assert_eq!(radii, [1.70, 1.52]);
//! let out = compute_surface(&Sphere::from_atoms(&atoms, &radii)?, &SurfaceParams::default())?;
//! assert!(out.stats.area > 0.0);
//! # Ok::<(), sesmesh::Error>(())
//! ```
//!
//! More in the [examples](https://github.com/uvar-apps/sesmesh/tree/main/crates/sesmesh/examples)
//! (file → OBJ, spheres → mesh, trajectories with [`SurfaceEngine`], custom
//! radii, field queries) and the
//! [CLI manual](https://github.com/uvar-apps/sesmesh/blob/main/docs/cli.md).
//!
//! ## Concepts
//!
//! * **Surfaces** ([`SurfaceKind`]): the **SES** (default) is the boundary of
//!   the region a probe sphere of radius `r_p` (1.4 Å for water) cannot
//!   enter; the **SAS** is traced by the probe centre (atom radii + `r_p`);
//!   the **vdW** surface is the union of the atom spheres. sesmesh evaluates
//!   the exact SES field `F(x) = r_p − d(x, P)` (`P`: positions the probe
//!   centre can reach) on a sparse grid and extracts its zero level set with
//!   Marching Cubes; every vertex is then moved onto the exact surface, so
//!   positions are exact and normals are analytic. Each vertex records its
//!   patch ([`PatchKind`]: convex, saddle, concave) and nearest atom.
//! * **Resolution** ([`Resolution`]): the grid spacing `h` in Å
//!   ([`SurfaceParams::with_spacing`]), a preset ([`Quality`]: draft 1.0,
//!   low 0.7, medium 0.5 (default), high 0.25, ultra 0.1 Å), or a target
//!   vertex density ([`SurfaceParams::with_density`]: about `1.5 / h²`
//!   vertices per Å²). Error decreases as `h²`: area −0.2 % and volume
//!   −0.02 % on average at 0.25 Å on a 53-entry protein corpus
//!   ([validation report](https://github.com/uvar-apps/sesmesh/blob/main/docs/validation-report.md)). [`SurfaceParams::with_smoothing`]
//!   improves triangle shapes while keeping vertices on the surface.
//! * **Components and cavities**: [`SurfaceOutput::components`] lists the
//!   closed components (area, signed volume, Euler characteristic, genus);
//!   internal cavities have negative volume. [`CavityMode`] keeps them (the
//!   exact SES), removes them with their content, or keeps only them;
//!   triangles are grouped by component.
//! * **Determinism**: the output is bit-identical for any number of threads
//!   and on repeated runs. Work runs in the current rayon pool (feature
//!   `parallel`, default); [`Control`] adds cancellation and progress.
//! * **SIMD**: the field kernel runs at the best instruction set of the CPU
//!   (AVX-512, AVX2, SSE4.2, NEON, WASM SIMD) or a portable fallback
//!   ([`SimdLevel`]); all levels agree within 2·10⁻⁵ Å.
//! * **Errors and warnings**: every fallible call returns [`Result`] with
//!   [`Error`]; non-fatal problems (unknown elements, missing radii,
//!   discarded alt-locs) are collected in [`Diagnostics`]. The library never
//!   prints.
//!
//! ## Modules
//!
//! | Module | Content |
//! |---|---|
//! | [`io`] | PDB, mmCIF, XYZ/extended XYZ, PQR readers (gzip), biological assemblies |
//! | [`structure`] | Atoms, models, [`Selection`] |
//! | [`radii`] | Bondi, Alvarez, ProtOr, uniform and custom radius rules |
//! | [`geom`] | [`Sphere`], SAS arrangement statistics |
//! | [`field`] | Exact field queries ([`SesField`]), [`SurfaceKind`], [`SimdLevel`] |
//! | [`mesh`] | [`Mesh`], validation, components, per-atom areas |
//! | [`export`] | OBJ, binary PLY, MSMS writers |
//! | [`color`] | Vertex colour palettes |
//!
//! ## Features
//!
//! `parallel` (rayon, default), `gzip` (compressed input and output,
//! default), `radii-file` (TOML radius rules, default). Without default
//! features the crate is single-threaded and dependency-light (WASM).
//!
//! ## Versioning
//!
//! Semantic versioning. Public structs and enums that may grow are
//! `#[non_exhaustive]`: build option structs from their constructors or
//! `Default` and assign fields; match enums with a wildcard arm.
//!
//! ## Licence
//!
//! Copyright (c) 2026 Ugo Varetto. Licensed under either of the Apache
//! License, Version 2.0 or the MIT licence, at your option.

// The library reports through return values, never by printing
// (Spec FR-DIAG-1).
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod color;
mod control;
pub mod diag;
pub mod error;
pub mod export;
pub mod field;
pub mod geom;
pub mod io;
pub mod mesh;
mod par;
pub mod radii;
pub mod structure;
mod surface;

pub use control::{Control, Progress, Stage};
pub use diag::{Diagnostics, Warning, WarningKind};
pub use error::{Error, Result};
pub use field::SimdLevel;
pub use field::{SesField, SurfaceKind};
pub use geom::Sphere;
pub use mesh::{Mesh, PatchKind};
pub use structure::{AtomRecord, AtomSet, Element, Model, Selection, Structure};
pub use surface::{
    CavityMode, NormalMode, Quality, RefineMode, Resolution, SurfaceEngine, SurfaceOutput,
    SurfaceParams, SurfaceStats, Timings, VERTICES_PER_AREA_H2, compute_surface,
    compute_surface_with, estimate_memory,
};

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
