// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sphere-set geometry: the public [`Sphere`] input type and the internal
//! SAS arrangement used by the surface engine (Design §6).

// Arrangement data is read by the field engine (Plan Phase 3); until then
// parts are only exercised by tests.
#![allow(dead_code)]

mod arrangement;
mod cells;
mod vec;

pub use arrangement::ArrangementStats;
#[allow(unused_imports)]
pub(crate) use arrangement::{Arc, Arrangement, Circle, Csr, EPS, SasVertex, Status};
#[allow(unused_imports)]
pub(crate) use cells::CellGrid;
#[allow(unused_imports)]
pub(crate) use vec::Vec3;

use crate::error::{Error, Result};
use crate::structure::AtomSet;

/// An atom as the surface engine sees it: a centre (Å) and a van der Waals
/// radius (Å). Radius 0 means "ignore".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    /// Centre in Å.
    pub center: [f64; 3],
    /// Radius in Å.
    pub radius: f64,
}

impl Sphere {
    /// A sphere at `center` with `radius`.
    pub fn new(center: [f64; 3], radius: f64) -> Self {
        Self { center, radius }
    }

    /// Pairs selected atoms with their radii (e.g. from
    /// [`radii::assign`](crate::radii::assign)).
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the lengths differ.
    pub fn from_atoms(atoms: &AtomSet<'_>, radii: &[f64]) -> Result<Vec<Sphere>> {
        if atoms.len() != radii.len() {
            return Err(Error::InvalidInput(format!(
                "{} atoms but {} radii",
                atoms.len(),
                radii.len()
            )));
        }
        Ok(atoms
            .iter()
            .zip(radii)
            .map(|(a, &radius)| Sphere::new(a.position, radius))
            .collect())
    }
}

/// Builds the SAS arrangement and returns its counts. Exposed for
/// benchmarks and diagnostics; the arrangement itself is internal.
///
/// # Errors
/// [`Error::InvalidInput`] for invalid spheres or probe radius.
pub fn arrangement_stats(spheres: &[Sphere], probe_radius: f64) -> Result<ArrangementStats> {
    Arrangement::build(spheres, probe_radius).map(|a| a.stats)
}
