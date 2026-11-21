// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch descriptions shared by the GPU back ends: the arrangement and grid
//! views uploaded to the device, the batch and refinement parameters, the
//! sub-block record and edge result encodings, and the pass times
//! (`docs/gpu-prior-art.md` §3.2, §5.2, §5.4). The kernels
//! (`kernels/sesmesh.cuh`) are the same for every back end.

use crate::Result;

/// Words per sub-block record: state, list base, then the sizes of the
/// inside (spheres: near), S, C, V and (SES, with [`BatchParams::near`])
/// near lists, stored in that order from the base.
pub const REC_WORDS: usize = 7;
/// Record state: the sub-block is sampled.
pub const STATE_SAMPLE: u32 = 0;
/// Record state: uniformly outside.
pub const STATE_OUT: u32 = 1;
/// Record state: uniformly inside.
pub const STATE_IN: u32 = 2;

/// Bit position of the feature kind in a batch's edge features; the low
/// bits hold the sphere, circle or vertex index.
pub const FEATURE_SHIFT: u32 = 29;
/// Feature kind: the point is probe-accessible.
pub const FEATURE_SOLVENT: u32 = 0;
/// Feature kind: nothing within the cutoff.
pub const FEATURE_BEYOND: u32 = 1;
/// Feature kind: a sphere (SES: its expanded sphere).
pub const FEATURE_SPHERE: u32 = 2;
/// Feature kind: a circle (SES).
pub const FEATURE_CIRCLE: u32 = 3;
/// Feature kind: a SAS vertex (SES).
pub const FEATURE_VERTEX: u32 = 4;
/// Flag in a batch's edge evaluation counts: Newton / Illinois converged (else the
/// edge was bisected).
pub const EDGE_CONVERGED: u32 = 1 << 31;

/// A cell grid: items of cell `c` are `items[starts[c]..starts[c + 1]]`,
/// cells x-fastest, origin in frame-local coordinates.
#[derive(Debug, Clone, Copy)]
pub struct GridRef<'a> {
    /// Origin (frame-local).
    pub origin: [f32; 3],
    /// Cell edge.
    pub cell: f32,
    /// Cells per axis.
    pub dims: [u32; 3],
    /// Cell starts (`dims` product + 1).
    pub starts: &'a [u32],
    /// Items, cell by cell.
    pub items: &'a [u32],
}

/// The arrangement in frame-local f32, as the CPU kernels see it.
#[derive(Debug, Clone, Copy)]
pub struct ArrangementRef<'a> {
    /// Spheres: x, y, z, radius, radius², (radius − tol)², vdW radius.
    pub sphere: [&'a [f32]; 7],
    /// Sphere status: 0 exposed, 1 buried, 2 contained.
    pub status: &'a [u32],
    /// Neighbour CSR (offsets, items) of the class-S blockers.
    pub neighbors: (&'a [u32], &'a [u32]),
    /// Circles: centre x, y, z, axis x, y, z, e1 x, y, z, radius.
    pub circle: [&'a [f32]; 10],
    /// Circle blocker CSR (offsets, items).
    pub circle_blockers: (&'a [u32], &'a [u32]),
    /// SAS vertices: x, y, z.
    pub vertex: [&'a [f32]; 3],
    /// Grid over sphere centres.
    pub sphere_grid: GridRef<'a>,
    /// Grid over circle centres.
    pub circle_grid: GridRef<'a>,
    /// Grid over vertices.
    pub vertex_grid: GridRef<'a>,
    /// Largest sphere radius.
    pub max_expanded: f32,
    /// Largest circle radius.
    pub rho_max: f32,
}

/// Field and lattice parameters of a batch.
#[derive(Debug, Clone, Copy)]
pub struct BatchParams {
    /// SES (else SAS/vdW, union of spheres).
    pub ses: bool,
    /// Probe radius.
    pub probe: f32,
    /// Band (clamp) of the sampled field.
    pub band: f32,
    /// SES: `(probe + band)²` as the CPU computes it.
    pub cutoff2: f32,
    /// Sub-block circumradius.
    pub rho: f32,
    /// Gather radius (SES `probe + band + rho`, spheres `band + rho`).
    pub reach: f32,
    /// Classification margin `rho + eps`.
    pub margin: f32,
    /// Slack on every f32 decision, above the f32 error.
    pub slack: f32,
    /// SES classification cutoff squared, `(probe + margin + eps)²`.
    pub cutoffc2: f32,
    /// Lattice spacing.
    pub h: f32,
    /// Brick edge in cells.
    pub b: u32,
    /// Sub-block edge in cells (`b / 2`).
    pub sub: u32,
    /// SES: also list the non-contained spheres within reach (the
    /// refinement blockers' superset; S1 measurements only).
    pub near: bool,
}

/// Wall times of one batch, in seconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchTimes {
    /// Count pass.
    pub count: f64,
    /// Host scan of the counts.
    pub scan: f64,
    /// Fill pass with the Lipschitz test.
    pub fill: f64,
    /// Sampling pass.
    pub sample: f64,
}

/// Residuals of the frame-local `f32` positions (`f64` minus `f32`, as
/// `f32`), the lattice spacing as an `f32` pair, and the iteration settings,
/// for the back ends' `Resident::refine`.
#[derive(Debug, Clone, Copy)]
pub struct RefineRef<'a> {
    /// Sphere centres: x, y, z residuals.
    pub sphere_lo: [&'a [f32]; 3],
    /// Circle centres: x, y, z residuals.
    pub circle_lo: [&'a [f32]; 3],
    /// SAS vertices: x, y, z residuals.
    pub vertex_lo: [&'a [f32]; 3],
    /// Lattice spacing, high part.
    pub h_hi: f32,
    /// Lattice spacing, low part (`h − h_hi`).
    pub h_lo: f32,
    /// Stop where `|F| ≤ tol`.
    pub tol: f32,
    /// Newton / Illinois evaluations per edge, at most.
    pub max_iter: u32,
    /// Bisection steps after an unconverged Newton / Illinois phase, at most.
    pub max_bisect: u32,
}

/// Wall times of the back ends' `Resident::refine`, in seconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct RefineTimes {
    /// Count pass.
    pub count: f64,
    /// Host scan of the counts.
    pub scan: f64,
    /// Refinement pass.
    pub refine: f64,
}

/// A back end's resident arrangement, for code generic over back ends: the
/// methods of `hip::Resident` and `cuda::Resident`, which implement it.
pub trait DeviceResident: Sync {
    /// The back end's batch buffers.
    type Batch: DeviceBatch + Send + Sync;

    /// Back end name: `hip` or `cuda`.
    const BACKEND: &'static str;

    /// Empty batch buffers on the resident's device.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation fails.
    fn new_batch(&self) -> Result<Self::Batch>;

    /// Uploads what refinement needs (`Resident::set_refinement`).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or copy fails.
    fn set_refinement(&mut self, r: &RefineRef<'_>) -> Result<()>;

    /// Classifies, gathers and samples a batch of bricks (`Resident::run`).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation, copy or launch fails.
    fn run(
        &self,
        p: &BatchParams,
        bricks: &[[i32; 3]],
        batch: &mut Self::Batch,
    ) -> Result<BatchTimes>;

    /// Refines the crossing edges of the last run (`Resident::refine`).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation, copy or launch fails.
    fn refine(&self, p: &BatchParams, batch: &mut Self::Batch) -> Result<RefineTimes>;

    /// Bytes of the resident arrangement.
    fn resident_bytes(&self) -> usize;
}

/// A back end's batch buffers as the host reads them, for code generic over
/// back ends: the accessors of `hip::Batch` and `cuda::Batch`.
pub trait DeviceBatch {
    /// Bricks of the last run.
    fn bricks(&self) -> usize;
    /// Sample values of brick `i` of the last run.
    fn brick_values(&self, i: usize) -> &[f32];
    /// Sub-block records of the last run.
    fn records(&self) -> &[u32];
    /// Whether the last run was refined.
    fn refined(&self) -> bool;
    /// First refined edge of every sub-block, then the total.
    fn edge_offsets(&self) -> &[u32];
    /// Edge parameter per refined edge.
    fn edge_params(&self) -> &[f32];
    /// Feature per refined edge.
    fn edge_features(&self) -> &[u32];
    /// Closest point per refined edge, 3 each.
    fn edge_closest(&self) -> &[f32];
    /// Evaluations per refined edge, with [`EDGE_CONVERGED`].
    fn edge_evals(&self) -> &[u32];
    /// Device and host bytes held.
    fn bytes(&self) -> usize;
}

/// Implements [`DeviceResident`] and [`DeviceBatch`] for a back end's
/// `Resident` and `Batch` by forwarding to their inherent methods.
macro_rules! forward_batch_traits {
    ($name:literal) => {
        impl<'d> $crate::batch::DeviceResident for Resident<'d> {
            type Batch = Batch<'d>;

            const BACKEND: &'static str = $name;

            fn new_batch(&self) -> $crate::Result<Batch<'d>> {
                Batch::new(self.device)
            }

            fn set_refinement(&mut self, r: &$crate::batch::RefineRef<'_>) -> $crate::Result<()> {
                Resident::set_refinement(self, r)
            }

            fn run(
                &self,
                p: &$crate::batch::BatchParams,
                bricks: &[[i32; 3]],
                batch: &mut Batch<'d>,
            ) -> $crate::Result<$crate::batch::BatchTimes> {
                Resident::run(self, p, bricks, batch)
            }

            fn refine(
                &self,
                p: &$crate::batch::BatchParams,
                batch: &mut Batch<'d>,
            ) -> $crate::Result<$crate::batch::RefineTimes> {
                Resident::refine(self, p, batch)
            }

            fn resident_bytes(&self) -> usize {
                Resident::resident_bytes(self)
            }
        }

        impl $crate::batch::DeviceBatch for Batch<'_> {
            fn bricks(&self) -> usize {
                Batch::bricks(self)
            }
            fn brick_values(&self, i: usize) -> &[f32] {
                Batch::brick_values(self, i)
            }
            fn records(&self) -> &[u32] {
                Batch::records(self)
            }
            fn refined(&self) -> bool {
                Batch::refined(self)
            }
            fn edge_offsets(&self) -> &[u32] {
                Batch::edge_offsets(self)
            }
            fn edge_params(&self) -> &[f32] {
                Batch::edge_params(self)
            }
            fn edge_features(&self) -> &[u32] {
                Batch::edge_features(self)
            }
            fn edge_closest(&self) -> &[f32] {
                Batch::edge_closest(self)
            }
            fn edge_evals(&self) -> &[u32] {
                Batch::edge_evals(self)
            }
            fn bytes(&self) -> usize {
                Batch::bytes(self)
            }
        }
    };
}
pub(crate) use forward_batch_traits;

/// Whether `off` is a valid CSR offset table of `items`: starts at 0, never
/// decreases, ends at `items.len()`.
pub(crate) fn csr_rows(off: &[u32], items: &[u32]) -> bool {
    off.first() == Some(&0)
        && off.windows(2).all(|w| w[0] <= w[1])
        && off.last().is_some_and(|&l| l as usize == items.len())
}

/// Checks the shapes the back ends' `Resident::new` rely on (the kernels
/// read rows `off[k]..off[k + 1]` of every candidate).
///
/// # Panics
///
/// If array lengths disagree or a CSR table is malformed.
pub(crate) fn validate(a: &ArrangementRef<'_>) {
    let n = a.sphere[0].len();
    assert!(a.sphere.iter().all(|v| v.len() == n) && a.status.len() == n);
    let nc = a.circle[0].len();
    assert!(a.circle.iter().all(|v| v.len() == nc));
    assert!(a.vertex.iter().all(|v| v.len() == a.vertex[0].len()));
    assert!(a.neighbors.0.len() == n + 1 && csr_rows(a.neighbors.0, a.neighbors.1));
    assert!(
        a.circle_blockers.0.len() == nc + 1 && csr_rows(a.circle_blockers.0, a.circle_blockers.1)
    );
    for g in [&a.sphere_grid, &a.circle_grid, &a.vertex_grid] {
        let cells = g.dims.iter().map(|&d| d as usize).product::<usize>();
        assert!(g.dims.iter().all(|&d| d > 0) && g.starts.len() == cells + 1);
        assert!(csr_rows(g.starts, g.items));
    }
}

/// Bytes of `a` once uploaded (4-byte words).
pub(crate) fn resident_bytes(a: &ArrangementRef<'_>) -> usize {
    let words = 8 * a.sphere[0].len()
        + 10 * a.circle[0].len()
        + 3 * a.vertex[0].len()
        + a.neighbors.0.len()
        + a.neighbors.1.len()
        + a.circle_blockers.0.len()
        + a.circle_blockers.1.len()
        + [&a.sphere_grid, &a.circle_grid, &a.vertex_grid]
            .iter()
            .map(|g| g.starts.len() + g.items.len())
            .sum::<usize>();
    4 * words
}
