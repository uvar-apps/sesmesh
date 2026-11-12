//! The surface field (Design §2) and its sampling on a sparse brick lattice
//! with SIMD kernels and multi-threading (Design §7).
//!
//! * [`SesField`] evaluates the exact SES, SAS or vdW field at any point in
//!   `f64`, with the closest surface feature and the normal.
//! * Internally, the engine finds the bricks the surface can pass through
//!   (conservative Lipschitz tests) and samples them with a kernel that
//!   processes 4–16 samples per instruction (NEON / SSE / AVX2 / AVX-512,
//!   chosen at runtime), one brick per task across all cores.

mod kernel;
mod lattice;
mod point;

pub use point::{FieldSample, Patch, SesField};

pub(crate) use kernel::Engine;
pub(crate) use lattice::pack;
pub(crate) use point::{Feature, NearEval};

use crate::error::Result;
use crate::geom::Sphere;

/// Which molecular surface to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SurfaceKind {
    /// Solvent excluded (Connolly / molecular) surface. Default.
    #[default]
    Ses,
    /// Solvent accessible surface (atom radii expanded by the probe radius).
    Sas,
    /// Van der Waals surface (union of atom spheres).
    Vdw,
}

impl SurfaceKind {
    /// Parses `ses`, `sas` or `vdw` (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "ses" => Some(Self::Ses),
            "sas" => Some(Self::Sas),
            "vdw" => Some(Self::Vdw),
            _ => None,
        }
    }
}

/// Instruction set selection for the sampling kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SimdLevel {
    /// Best instruction set of the running CPU (detected once). Default.
    #[default]
    Auto,
    /// Portable scalar code (testing and diagnostics).
    Scalar,
}

/// Counts from [`sample_surface_bricks`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SamplingStats {
    /// Bricks overlapping some sphere.
    pub touched_bricks: usize,
    /// Bricks the surface can pass through (sampled).
    pub surface_bricks: usize,
    /// Samples evaluated.
    pub samples: usize,
    /// Samples with a negative field value (inside).
    pub inside_samples: usize,
    /// Instruction set used.
    pub simd: &'static str,
    /// Order-dependent hash of all sample values (determinism checks).
    pub checksum: u64,
}

/// Samples the field on every surface brick and returns counts and a
/// checksum. Exposed for benchmarks and diagnostics; the mesher (Phase 4)
/// consumes the samples directly.
///
/// # Errors
/// [`Error::InvalidInput`](crate::Error::InvalidInput) for invalid spheres,
/// probe radius, spacing or brick size.
pub fn sample_surface_bricks(
    spheres: &[Sphere],
    probe_radius: f64,
    kind: SurfaceKind,
    spacing: f64,
    brick_cells: usize,
    simd: SimdLevel,
) -> Result<SamplingStats> {
    let field = SesField::new(spheres, probe_radius, kind)?;
    let engine = Engine::new(field, spacing, brick_cells, simd)?;
    let (touched, bricks) = engine.surface_bricks()?;
    let per_brick = engine.map_bricks(&bricks, |_, values| {
        let inside = values.iter().filter(|v| **v < 0.0).count();
        let hash = values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
            (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
        });
        (inside, hash)
    });
    let checksum = per_brick
        .iter()
        .fold(0u64, |h, &(_, b)| h.rotate_left(5) ^ b);
    Ok(SamplingStats {
        touched_bricks: touched,
        surface_bricks: bricks.len(),
        samples: bricks.len() * engine.samples_per_brick(),
        inside_samples: per_brick.iter().map(|p| p.0).sum(),
        simd: engine.simd_name(),
        checksum,
    })
}

#[cfg(test)]
mod tests;
