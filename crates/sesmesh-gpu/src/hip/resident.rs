// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classification, candidate gathering and field sampling on the device
//! (Plan T11.3′, `docs/gpu-prior-art.md` §3.2 and §5.2), from an arrangement
//! uploaded once ([`Resident`]) and the CPU's own cell grids, batch by batch
//! ([`Batch`]).
//!
//! The device repeats the CPU's f64 list and classification tests in f32,
//! with `slack` added on the inclusive side. Its lists are therefore
//! supersets of the CPU's, and its uniform sub-blocks a subset. A batch holds
//! all 8 sub-blocks of each brick (sub-block `8 · brick + sb`). Passes:
//!
//! 1. count: cheap classification and list sizes;
//! 2. host scan: list bases, so the id pool is sized from the counts;
//! 3. fill: the lists, then the Lipschitz test at the sub-block centre (SES);
//! 4. sample: every brick sample, `±band` where no sub-block samples it.
//!
//! Device code is f32 and integer only.

use std::ffi::c_void;
use std::time::Instant;

use super::{Device, DeviceBuffer, HostBuffer};
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

/// Lanes per workgroup (`LANES` in the kernels).
const BLOCK: u32 = 64;

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

#[repr(C)]
struct GridParams {
    starts: *const c_void,
    items: *const c_void,
    ox: f32,
    oy: f32,
    oz: f32,
    cell: f32,
    dx: u32,
    dy: u32,
    dz: u32,
    n_items: u32,
}

/// `ResidentParams` in `kernels/hip/sesmesh.hip`.
#[repr(C)]
struct Params {
    sphere: [*const c_void; 7],
    status: *const c_void,
    nb_off: *const c_void,
    nb_items: *const c_void,
    circle: [*const c_void; 10],
    cb_off: *const c_void,
    cb_items: *const c_void,
    vertex: [*const c_void; 3],
    gs: GridParams,
    gc: GridParams,
    gv: GridParams,
    bricks: *const c_void,
    rec: *mut c_void,
    ids: *mut c_void,
    values: *mut c_void,
    ids_len: u64,
    values_len: u64,
    n_sub: u32,
    kind: u32,
    n_spheres: u32,
    n_circles: u32,
    n_vertices: u32,
    b: u32,
    sub: u32,
    near: u32,
    probe: f32,
    band: f32,
    cutoff2: f32,
    rho: f32,
    reach: f32,
    margin: f32,
    slack: f32,
    cutoffc2: f32,
    h: f32,
    max_expanded: f32,
    rho_max: f32,
    pad2: f32,
}

struct GridBuf {
    starts: DeviceBuffer<u32>,
    items: DeviceBuffer<u32>,
    origin: [f32; 3],
    cell: f32,
    dims: [u32; 3],
}

impl GridBuf {
    fn new(device: &Device, g: &GridRef<'_>) -> Result<Self> {
        Ok(Self {
            starts: device.upload(g.starts)?,
            items: device.upload(g.items)?,
            origin: g.origin,
            cell: g.cell,
            dims: g.dims,
        })
    }

    fn params(&self) -> GridParams {
        GridParams {
            starts: self.starts.as_device_ptr().cast_const(),
            items: self.items.as_device_ptr().cast_const(),
            ox: self.origin[0],
            oy: self.origin[1],
            oz: self.origin[2],
            cell: self.cell,
            dx: self.dims[0],
            dy: self.dims[1],
            dz: self.dims[2],
            n_items: u32::try_from(self.items.len()).expect("fewer than 2^32 items"),
        }
    }
}

/// The arrangement resident on the device. Batches run on it from any one
/// thread at a time, each into its own [`Batch`].
pub struct Resident<'d> {
    device: &'d Device,
    sphere: Vec<DeviceBuffer<f32>>,
    status: DeviceBuffer<u32>,
    nb: (DeviceBuffer<u32>, DeviceBuffer<u32>),
    circle: Vec<DeviceBuffer<f32>>,
    cb: (DeviceBuffer<u32>, DeviceBuffer<u32>),
    vertex: Vec<DeviceBuffer<f32>>,
    grids: [GridBuf; 3],
    max_expanded: f32,
    rho_max: f32,
    resident_bytes: usize,
}

/// The pinned host buffers of one batch (the device reads and writes them
/// directly), grown to the exact need of each batch.
pub struct Batch<'d> {
    device: &'d Device,
    bricks: HostBuffer<i32>,
    rec: HostBuffer<u32>,
    ids: HostBuffer<u32>,
    values: HostBuffer<f32>,
    n_bricks: usize,
    spb: usize,
    n_ids: usize,
}

impl std::fmt::Debug for Batch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Batch")
            .field("bricks", &self.n_bricks)
            .field("ids", &self.n_ids)
            .finish_non_exhaustive()
    }
}

impl<'d> Batch<'d> {
    /// Empty buffers on `device`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation fails.
    pub fn new(device: &'d Device) -> Result<Self> {
        Ok(Self {
            device,
            bricks: device.alloc_host(0)?,
            rec: device.alloc_host(0)?,
            ids: device.alloc_host(0)?,
            values: device.alloc_host(0)?,
            n_bricks: 0,
            spb: 0,
            n_ids: 0,
        })
    }

    /// Bricks of the last run.
    #[must_use]
    pub fn bricks(&self) -> usize {
        self.n_bricks
    }

    /// Ids in the lists of the last run.
    #[must_use]
    pub fn ids(&self) -> usize {
        self.n_ids
    }

    /// Candidate lists of the last run; a record's lists start at its base.
    #[must_use]
    pub fn id_lists(&self) -> &[u32] {
        &self.ids[..self.n_ids]
    }

    /// Sample values of the last run, `(b + 1)³` per brick, x fastest.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values[..self.spb * self.n_bricks]
    }

    /// Sample values of brick `i` of the last run.
    ///
    /// # Panics
    ///
    /// If `i` is not a brick of the last run.
    #[must_use]
    pub fn brick_values(&self, i: usize) -> &[f32] {
        assert!(i < self.n_bricks, "brick {i} of {}", self.n_bricks);
        &self.values[i * self.spb..(i + 1) * self.spb]
    }

    /// Sub-block records of the last run ([`REC_WORDS`] each, 8 per brick).
    #[must_use]
    pub fn records(&self) -> &[u32] {
        &self.rec[..REC_WORDS * 8 * self.n_bricks]
    }

    /// Host bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        4 * (self.bricks.len() + self.rec.len() + self.ids.len() + self.values.len())
    }
}

impl std::fmt::Debug for Resident<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resident")
            .field("resident_bytes", &self.resident_bytes)
            .finish_non_exhaustive()
    }
}

fn csr_rows(off: &[u32], items: &[u32]) -> bool {
    off.first() == Some(&0)
        && off.windows(2).all(|w| w[0] <= w[1])
        && off.last().is_some_and(|&l| l as usize == items.len())
}

impl<'d> Resident<'d> {
    /// Uploads the arrangement.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or copy fails.
    ///
    /// # Panics
    ///
    /// If array lengths disagree or a CSR table is malformed (the kernels
    /// read rows `off[k]..off[k + 1]` of every candidate).
    pub fn new(device: &'d Device, a: &ArrangementRef<'_>) -> Result<Self> {
        let n = a.sphere[0].len();
        assert!(a.sphere.iter().all(|v| v.len() == n) && a.status.len() == n);
        let nc = a.circle[0].len();
        assert!(a.circle.iter().all(|v| v.len() == nc));
        assert!(a.vertex.iter().all(|v| v.len() == a.vertex[0].len()));
        assert!(a.neighbors.0.len() == n + 1 && csr_rows(a.neighbors.0, a.neighbors.1));
        assert!(
            a.circle_blockers.0.len() == nc + 1
                && csr_rows(a.circle_blockers.0, a.circle_blockers.1)
        );
        for g in [&a.sphere_grid, &a.circle_grid, &a.vertex_grid] {
            let cells = g.dims.iter().map(|&d| d as usize).product::<usize>();
            assert!(g.dims.iter().all(|&d| d > 0) && g.starts.len() == cells + 1);
            assert!(csr_rows(g.starts, g.items));
        }
        let up = |v: &[f32]| device.upload(v);
        let sphere = a.sphere.iter().map(|v| up(v)).collect::<Result<Vec<_>>>()?;
        let circle = a.circle.iter().map(|v| up(v)).collect::<Result<Vec<_>>>()?;
        let vertex = a.vertex.iter().map(|v| up(v)).collect::<Result<Vec<_>>>()?;
        let grids = [
            GridBuf::new(device, &a.sphere_grid)?,
            GridBuf::new(device, &a.circle_grid)?,
            GridBuf::new(device, &a.vertex_grid)?,
        ];
        let words = 8 * n
            + 10 * nc
            + 3 * a.vertex[0].len()
            + a.neighbors.0.len()
            + a.neighbors.1.len()
            + a.circle_blockers.0.len()
            + a.circle_blockers.1.len()
            + [&a.sphere_grid, &a.circle_grid, &a.vertex_grid]
                .iter()
                .map(|g| g.starts.len() + g.items.len())
                .sum::<usize>();
        Ok(Self {
            device,
            sphere,
            status: device.upload(a.status)?,
            nb: (device.upload(a.neighbors.0)?, device.upload(a.neighbors.1)?),
            circle,
            cb: (
                device.upload(a.circle_blockers.0)?,
                device.upload(a.circle_blockers.1)?,
            ),
            vertex,
            grids,
            max_expanded: a.max_expanded,
            rho_max: a.rho_max,
            resident_bytes: 4 * words,
        })
    }

    /// Bytes of the resident arrangement.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Classifies, gathers and samples the bricks whose origins (minus the
    /// frame origin, in lattice units) are `bricks`, into `batch`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or launch fails.
    ///
    /// # Panics
    ///
    /// If a count exceeds `u32`, `p.sub` is not `p.b / 2` with `p.b` even
    /// and at least 4, or `batch` belongs to another device.
    pub fn run(
        &self,
        p: &BatchParams,
        bricks: &[[i32; 3]],
        batch: &mut Batch<'_>,
    ) -> Result<BatchTimes> {
        assert!(
            std::ptr::eq(self.device, batch.device),
            "batch of another device"
        );
        assert!(
            p.b >= 4 && p.b.is_multiple_of(2) && p.sub == p.b / 2,
            "8 sub-blocks per brick"
        );
        let mut times = BatchTimes::default();
        let n_sub = 8 * bricks.len();
        let side = p.b as usize + 1;
        let spb = side * side * side;
        if batch.bricks.len() < 3 * bricks.len() {
            batch.bricks = self.device.alloc_host(3 * bricks.len())?;
        }
        for (d, s) in batch.bricks.as_chunks_mut::<3>().0.iter_mut().zip(bricks) {
            d.copy_from_slice(s);
        }
        if batch.rec.len() < REC_WORDS * n_sub {
            batch.rec = self.device.alloc_host(REC_WORDS * n_sub)?;
        }
        if batch.values.len() < spb * bricks.len() {
            batch.values = self.device.alloc_host(spb * bricks.len())?;
        }
        batch.n_bricks = bricks.len();
        batch.spb = spb;
        if bricks.is_empty() {
            return Ok(times);
        }
        let grid = u32::try_from(n_sub).expect("fewer than 2^32 sub-blocks");

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        self.launch(self.device.kernels.resident_count, grid, &mut params)?;
        times.count = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total = 0u64;
        for r in batch.rec[..REC_WORDS * n_sub]
            .as_chunks_mut::<REC_WORDS>()
            .0
        {
            r[1] = u32::try_from(total).expect("fewer than 2^32 list ids per batch");
            total += r[2..7].iter().map(|&c| u64::from(c)).sum::<u64>();
        }
        let need = usize::try_from(total).expect("ids fit in memory");
        if batch.ids.len() < need {
            batch.ids = self.device.alloc_host(need)?;
        }
        batch.n_ids = need;
        times.scan = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        self.launch(self.device.kernels.resident_fill, grid, &mut params)?;
        times.fill = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        self.launch(self.device.kernels.resident_sample, grid, &mut params)?;
        times.sample = t.elapsed().as_secs_f64();
        Ok(times)
    }

    fn params(&self, p: &BatchParams, n_sub: u32, batch: &mut Batch<'_>) -> Params {
        let len32 = |n: usize| u32::try_from(n).expect("fewer than 2^32 elements");
        let ptr = |b: &DeviceBuffer<f32>| b.as_device_ptr().cast_const();
        Params {
            sphere: std::array::from_fn(|i| ptr(&self.sphere[i])),
            status: self.status.as_device_ptr().cast_const(),
            nb_off: self.nb.0.as_device_ptr().cast_const(),
            nb_items: self.nb.1.as_device_ptr().cast_const(),
            circle: std::array::from_fn(|i| ptr(&self.circle[i])),
            cb_off: self.cb.0.as_device_ptr().cast_const(),
            cb_items: self.cb.1.as_device_ptr().cast_const(),
            vertex: std::array::from_fn(|i| ptr(&self.vertex[i])),
            gs: self.grids[0].params(),
            gc: self.grids[1].params(),
            gv: self.grids[2].params(),
            bricks: batch.bricks.as_device_ptr().cast_const(),
            rec: batch.rec.as_device_ptr(),
            ids: batch.ids.as_device_ptr(),
            values: batch.values.as_device_ptr(),
            ids_len: batch.ids.len() as u64,
            values_len: batch.values.len() as u64,
            n_sub,
            kind: u32::from(!p.ses),
            n_spheres: len32(self.sphere[0].len()),
            n_circles: len32(self.circle[0].len()),
            n_vertices: len32(self.vertex[0].len()),
            b: p.b,
            sub: p.sub,
            near: u32::from(p.near),
            probe: p.probe,
            band: p.band,
            cutoff2: p.cutoff2,
            rho: p.rho,
            reach: p.reach,
            margin: p.margin,
            slack: p.slack,
            cutoffc2: p.cutoffc2,
            h: p.h,
            max_expanded: self.max_expanded,
            rho_max: self.rho_max,
            pad2: 0.0,
        }
    }

    /// Refines the crossing edges `edges` of the last run in `batch` in f32
    /// (spike S2), into `refiner`'s outputs. `groups` holds, per group of
    /// edges of one sub-block: brick index in the batch, sub-block, first
    /// edge, edge count; an edge is `x | y << 4 | z << 8 | axis << 12`
    /// (its lower sample in the brick).
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or launch fails.
    ///
    /// # Panics
    ///
    /// If a count exceeds `u32`, or `batch` or `refiner` belongs to another
    /// device.
    #[allow(clippy::too_many_arguments)]
    pub fn refine(
        &self,
        p: &BatchParams,
        batch: &mut Batch<'_>,
        refiner: &mut Refiner<'_>,
        groups: &[[u32; 4]],
        edges: &[u32],
        tol: f32,
        max_iter: u32,
        max_bisect: u32,
    ) -> Result<()> {
        assert!(
            std::ptr::eq(self.device, batch.device) && std::ptr::eq(self.device, refiner.device),
            "buffers of another device"
        );
        let len32 = |n: usize| u32::try_from(n).expect("fewer than 2^32 elements");
        let r = refiner;
        if r.groups.len() < 4 * groups.len() {
            r.groups = self.device.alloc_host(4 * groups.len())?;
        }
        for (d, s) in r.groups.as_chunks_mut::<4>().0.iter_mut().zip(groups) {
            d.copy_from_slice(s);
        }
        let n = edges.len();
        if r.edges.len() < n {
            r.edges = self.device.alloc_host(n)?;
            r.t = self.device.alloc_host(n)?;
            r.feat = self.device.alloc_host(n)?;
            r.q = self.device.alloc_host(3 * n)?;
            r.evals = self.device.alloc_host(n)?;
        }
        r.edges[..n].copy_from_slice(edges);
        r.n_edges = n;
        if groups.is_empty() {
            return Ok(());
        }
        let grid = len32(groups.len());
        let mut params = self.params(p, len32(batch.n_bricks * 8), batch);
        let ptr = |b: &DeviceBuffer<f32>| b.as_device_ptr().cast_const();
        let mut rp = RefineParams {
            sphere_lo: std::array::from_fn(|i| ptr(&r.sphere_lo[i])),
            circle_lo: std::array::from_fn(|i| ptr(&r.circle_lo[i])),
            vertex_lo: std::array::from_fn(|i| ptr(&r.vertex_lo[i])),
            groups: r.groups.as_device_ptr().cast_const(),
            edges: r.edges.as_device_ptr().cast_const(),
            t_out: r.t.as_device_ptr(),
            feat_out: r.feat.as_device_ptr(),
            q_out: r.q.as_device_ptr(),
            evals_out: r.evals.as_device_ptr(),
            n_groups: grid,
            n_edges: len32(n),
            max_iter,
            max_bisect,
            h_hi: r.h_hi,
            h_lo: r.h_lo,
            tol,
            pad: 0.0,
        };
        let mut args: [*mut c_void; 2] = [
            std::ptr::from_mut(&mut params).cast(),
            std::ptr::from_mut(&mut rp).cast(),
        ];
        // SAFETY: `Params` and `RefineParams` by value, laid out as
        // `ResidentParams` and `RefineParams`. Every pointer is a live buffer
        // of this device or pinned host memory. The residual arrays have the
        // resident arrays' lengths (`Refiner::new`), and the kernel bounds
        // every element index by those counts; groups and edges are bounded
        // by `n_groups` and `n_edges`, outputs have `n_edges` entries, list
        // and value reads are bounded as in `run`. The host touches no
        // buffer until the synchronous launch returns.
        unsafe {
            self.device
                .launch_sync(self.device.kernels.resident_refine, grid, BLOCK, &mut args)
        }
    }

    fn launch(&self, function: super::ffi::Handle, grid: u32, params: &mut Params) -> Result<()> {
        let mut args: [*mut c_void; 1] = [std::ptr::from_mut(params).cast()];
        // SAFETY: `Params` by value, laid out as `ResidentParams`. Every
        // pointer is a live buffer of this device (or pinned host memory)
        // with the length passed alongside: the kernels bound element
        // indices by the element counts, CSR rows by the validated offsets
        // (`new`), list writes by the counts and `ids_len`, and value writes
        // by `values_len`. The host does not touch the buffers until the
        // synchronous launch returns (`&mut Batch` in `run`; the resident
        // buffers are only read).
        unsafe { self.device.launch_sync(function, grid, BLOCK, &mut args) }
    }
}

/// Residuals of the frame-local `f32` positions (`f64` minus `f32`, as
/// `f32`) and the lattice spacing as an `f32` pair, for [`Resident::refine`].
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
}

/// `RefineParams` in `kernels/hip/sesmesh.hip`.
#[repr(C)]
struct RefineParams {
    sphere_lo: [*const c_void; 3],
    circle_lo: [*const c_void; 3],
    vertex_lo: [*const c_void; 3],
    groups: *const c_void,
    edges: *const c_void,
    t_out: *mut c_void,
    feat_out: *mut c_void,
    q_out: *mut c_void,
    evals_out: *mut c_void,
    n_groups: u32,
    n_edges: u32,
    max_iter: u32,
    max_bisect: u32,
    h_hi: f32,
    h_lo: f32,
    tol: f32,
    pad: f32,
}

/// Residual arrays on the device and the per-edge buffers of
/// [`Resident::refine`] (spike S2).
pub struct Refiner<'d> {
    device: &'d Device,
    sphere_lo: Vec<DeviceBuffer<f32>>,
    circle_lo: Vec<DeviceBuffer<f32>>,
    vertex_lo: Vec<DeviceBuffer<f32>>,
    h_hi: f32,
    h_lo: f32,
    groups: HostBuffer<u32>,
    edges: HostBuffer<u32>,
    t: HostBuffer<f32>,
    feat: HostBuffer<u32>,
    q: HostBuffer<f32>,
    evals: HostBuffer<u32>,
    n_edges: usize,
}

impl std::fmt::Debug for Refiner<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Refiner")
            .field("edges", &self.n_edges)
            .finish_non_exhaustive()
    }
}

impl<'d> Refiner<'d> {
    /// Uploads the residuals for `resident`.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or copy fails.
    ///
    /// # Panics
    ///
    /// If the residual arrays do not have `resident`'s element counts.
    pub fn new(resident: &Resident<'d>, r: &RefineRef<'_>) -> Result<Self> {
        let device = resident.device;
        let (ns, nc, nv) = (
            resident.sphere[0].len(),
            resident.circle[0].len(),
            resident.vertex[0].len(),
        );
        assert!(r.sphere_lo.iter().all(|v| v.len() == ns));
        assert!(r.circle_lo.iter().all(|v| v.len() == nc));
        assert!(r.vertex_lo.iter().all(|v| v.len() == nv));
        let up = |v: &[&[f32]; 3]| {
            v.iter()
                .map(|x| device.upload(x))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            device,
            sphere_lo: up(&r.sphere_lo)?,
            circle_lo: up(&r.circle_lo)?,
            vertex_lo: up(&r.vertex_lo)?,
            h_hi: r.h_hi,
            h_lo: r.h_lo,
            groups: device.alloc_host(0)?,
            edges: device.alloc_host(0)?,
            t: device.alloc_host(0)?,
            feat: device.alloc_host(0)?,
            q: device.alloc_host(0)?,
            evals: device.alloc_host(0)?,
            n_edges: 0,
        })
    }

    /// Edge parameters of the last run (the last evaluated point).
    #[must_use]
    pub fn t(&self) -> &[f32] {
        &self.t[..self.n_edges]
    }

    /// Features of the last evaluation per edge: `kind << 29 | id`, kinds
    /// 0 solvent, 1 beyond, 2 sphere, 3 circle, 4 vertex.
    #[must_use]
    pub fn features(&self) -> &[u32] {
        &self.feat[..self.n_edges]
    }

    /// Closest points of the last evaluation per edge, local to the brick
    /// origin (3 per edge).
    #[must_use]
    pub fn closest(&self) -> &[f32] {
        &self.q[..3 * self.n_edges]
    }

    /// Field evaluations per edge; bit 31 set where Newton / Illinois
    /// converged.
    #[must_use]
    pub fn evals(&self) -> &[u32] {
        &self.evals[..self.n_edges]
    }
}
