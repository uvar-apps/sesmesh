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
//! [`Resident::refine`] then refines the batch's crossing edges in f32 (fast
//! mode, Plan T11.5′, `docs/gpu-prior-art.md` §5.4): a count pass per
//! sub-block, a host scan, and the refinement, which lists each sub-block's
//! edges in the CPU's order.
//!
//! Device code is f32 and integer only.

use std::ffi::c_void;
use std::time::Instant;

use super::{Device, DeviceBuffer, HostBuffer};
use crate::Result;
use crate::batch::{
    ArrangementRef, BatchParams, BatchTimes, GridRef, REC_WORDS, RefineRef, RefineTimes,
};

/// Lanes per workgroup (`LANES` in the kernels).
const BLOCK: u32 = 64;

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

/// `ResidentParams` in `kernels/sesmesh.cuh`.
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
    refine: Option<RefineData>,
}

/// Residuals and settings of [`Resident::refine`] on the device.
struct RefineData {
    sphere_lo: Vec<DeviceBuffer<f32>>,
    circle_lo: Vec<DeviceBuffer<f32>>,
    vertex_lo: Vec<DeviceBuffer<f32>>,
    h_hi: f32,
    h_lo: f32,
    tol: f32,
    max_iter: u32,
    max_bisect: u32,
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
    edge_count: HostBuffer<u32>,
    edge_off: HostBuffer<u32>,
    t: HostBuffer<f32>,
    feat: HostBuffer<u32>,
    q: HostBuffer<f32>,
    evals: HostBuffer<u32>,
    /// Crossing edges refined after the last run (`None`: not refined).
    n_edges: Option<usize>,
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
            edge_count: device.alloc_host(0)?,
            edge_off: device.alloc_host(0)?,
            t: device.alloc_host(0)?,
            feat: device.alloc_host(0)?,
            q: device.alloc_host(0)?,
            evals: device.alloc_host(0)?,
            n_edges: None,
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

    /// Whether [`Resident::refine`] ran on the last run.
    #[must_use]
    pub fn refined(&self) -> bool {
        self.n_edges.is_some()
    }

    fn n_edges(&self) -> usize {
        self.n_edges.unwrap_or(0)
    }

    /// First refined edge of every sub-block (`8 · brick + sb`), then the
    /// total: `8 · bricks + 1` entries; empty unless [`Batch::refined`].
    /// The edges of a sub-block are those whose lower sample lies in it
    /// (sub-block `(v / sub).min(1)` per axis, `v < b`), in z, y, x, axis
    /// order.
    #[must_use]
    pub fn edge_offsets(&self) -> &[u32] {
        if self.refined() {
            &self.edge_off[..8 * self.n_bricks + 1]
        } else {
            &[]
        }
    }

    /// Per refined edge: the edge parameter of the last evaluation (from the
    /// lower sample, not clamped).
    #[must_use]
    pub fn edge_params(&self) -> &[f32] {
        &self.t[..self.n_edges()]
    }

    /// Per refined edge: the feature of the last evaluation,
    /// `kind << FEATURE_SHIFT | index` ([`crate::batch::FEATURE_SOLVENT`] …).
    #[must_use]
    pub fn edge_features(&self) -> &[u32] {
        &self.feat[..self.n_edges()]
    }

    /// Per refined edge: the closest point of the last evaluation (SES; the
    /// evaluated point for spheres), relative to the brick origin, 3 each.
    #[must_use]
    pub fn edge_closest(&self) -> &[f32] {
        &self.q[..3 * self.n_edges()]
    }

    /// Per refined edge: field evaluations, with [`crate::batch::EDGE_CONVERGED`] set where
    /// Newton / Illinois converged.
    #[must_use]
    pub fn edge_evals(&self) -> &[u32] {
        &self.evals[..self.n_edges()]
    }

    /// Host bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        4 * (self.bricks.len()
            + self.rec.len()
            + self.ids.len()
            + self.values.len()
            + self.edge_count.len()
            + self.edge_off.len()
            + self.t.len()
            + self.feat.len()
            + self.q.len()
            + self.evals.len())
    }
}

impl std::fmt::Debug for Resident<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resident")
            .field("resident_bytes", &self.resident_bytes)
            .finish_non_exhaustive()
    }
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
        crate::batch::validate(a);
        let up = |v: &[f32]| device.upload(v);
        let sphere = a.sphere.iter().map(|v| up(v)).collect::<Result<Vec<_>>>()?;
        let circle = a.circle.iter().map(|v| up(v)).collect::<Result<Vec<_>>>()?;
        let vertex = a.vertex.iter().map(|v| up(v)).collect::<Result<Vec<_>>>()?;
        let grids = [
            GridBuf::new(device, &a.sphere_grid)?,
            GridBuf::new(device, &a.circle_grid)?,
            GridBuf::new(device, &a.vertex_grid)?,
        ];
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
            resident_bytes: crate::batch::resident_bytes(a),
            refine: None,
        })
    }

    /// Uploads the residuals and settings [`Resident::refine`] needs.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or copy fails.
    ///
    /// # Panics
    ///
    /// If the residual arrays do not have the arrangement's element counts.
    pub fn set_refinement(&mut self, r: &RefineRef<'_>) -> Result<()> {
        let (ns, nc, nv) = (
            self.sphere[0].len(),
            self.circle[0].len(),
            self.vertex[0].len(),
        );
        assert!(r.sphere_lo.iter().all(|v| v.len() == ns));
        assert!(r.circle_lo.iter().all(|v| v.len() == nc));
        assert!(r.vertex_lo.iter().all(|v| v.len() == nv));
        let device = self.device;
        let up = |v: &[&[f32]; 3]| {
            v.iter()
                .map(|x| device.upload(x))
                .collect::<Result<Vec<_>>>()
        };
        let data = RefineData {
            sphere_lo: up(&r.sphere_lo)?,
            circle_lo: up(&r.circle_lo)?,
            vertex_lo: up(&r.vertex_lo)?,
            h_hi: r.h_hi,
            h_lo: r.h_lo,
            tol: r.tol,
            max_iter: r.max_iter,
            max_bisect: r.max_bisect,
        };
        self.resident_bytes += 12 * (ns + nc + nv);
        self.refine = Some(data);
        Ok(())
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
        batch.n_edges = None;
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

    /// Refines the crossing edges of the last run in `batch` in f32, into
    /// `batch` ([`Batch::edge_offsets`] …): the CPU's safeguarded Newton /
    /// Illinois iteration, then bisection, against the lists of each edge's
    /// sub-block, in brick-local coordinates. `p` must be the run's.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation or launch fails.
    ///
    /// # Panics
    ///
    /// Without [`Resident::set_refinement`], if `p.b` exceeds 32 (edge
    /// codes), if a count exceeds `u32`, or if `batch` belongs to another
    /// device.
    pub fn refine(&self, p: &BatchParams, batch: &mut Batch<'_>) -> Result<RefineTimes> {
        assert!(
            std::ptr::eq(self.device, batch.device),
            "batch of another device"
        );
        assert!(p.b <= 32, "brick edges up to 32 cells");
        let r = self
            .refine
            .as_ref()
            .expect("Resident::set_refinement before refine");
        let mut times = RefineTimes::default();
        let n_sub = 8 * batch.n_bricks;
        if batch.edge_count.len() < n_sub {
            batch.edge_count = self.device.alloc_host(n_sub)?;
        }
        if batch.edge_off.len() < n_sub + 1 {
            batch.edge_off = self.device.alloc_host(n_sub + 1)?;
        }
        batch.edge_off[0] = 0;
        if n_sub == 0 {
            batch.n_edges = Some(0);
            return Ok(times);
        }
        let grid = u32::try_from(n_sub).expect("fewer than 2^32 sub-blocks");

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        let mut rp = Self::refine_params(r, batch, 0);
        self.launch_refine(
            self.device.kernels.resident_edge_count,
            grid,
            &mut params,
            &mut rp,
        )?;
        times.count = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut total = 0u32;
        for j in 0..n_sub {
            total = total
                .checked_add(batch.edge_count[j])
                .expect("fewer than 2^32 crossing edges per batch");
            batch.edge_off[j + 1] = total;
        }
        let n = total as usize;
        if batch.t.len() < n {
            batch.t = self.device.alloc_host(n)?;
            batch.feat = self.device.alloc_host(n)?;
            batch.q = self.device.alloc_host(3 * n)?;
            batch.evals = self.device.alloc_host(n)?;
        }
        times.scan = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        let mut rp = Self::refine_params(r, batch, total);
        self.launch_refine(
            self.device.kernels.resident_refine,
            grid,
            &mut params,
            &mut rp,
        )?;
        times.refine = t.elapsed().as_secs_f64();
        batch.n_edges = Some(n);
        Ok(times)
    }

    fn refine_params(r: &RefineData, batch: &mut Batch<'_>, n_edges: u32) -> RefineParams {
        let ptr = |b: &DeviceBuffer<f32>| b.as_device_ptr().cast_const();
        RefineParams {
            sphere_lo: std::array::from_fn(|i| ptr(&r.sphere_lo[i])),
            circle_lo: std::array::from_fn(|i| ptr(&r.circle_lo[i])),
            vertex_lo: std::array::from_fn(|i| ptr(&r.vertex_lo[i])),
            edge_count: batch.edge_count.as_device_ptr(),
            edge_off: batch.edge_off.as_device_ptr().cast_const(),
            t_out: batch.t.as_device_ptr(),
            feat_out: batch.feat.as_device_ptr(),
            q_out: batch.q.as_device_ptr(),
            evals_out: batch.evals.as_device_ptr(),
            n_edges,
            max_iter: r.max_iter,
            max_bisect: r.max_bisect,
            pad0: 0,
            h_hi: r.h_hi,
            h_lo: r.h_lo,
            tol: r.tol,
            pad: 0.0,
        }
    }

    fn launch_refine(
        &self,
        function: super::ffi::Handle,
        grid: u32,
        params: &mut Params,
        rp: &mut RefineParams,
    ) -> Result<()> {
        let mut args: [*mut c_void; 2] = [
            std::ptr::from_mut(params).cast(),
            std::ptr::from_mut(rp).cast(),
        ];
        // SAFETY: `Params` and `RefineParams` by value, laid out as
        // `ResidentParams` and `RefineParams`. Every pointer is a live buffer
        // of this device or pinned host memory: the residuals have the
        // arrangement's lengths (`set_refinement`) and the kernels bound
        // element indices by its counts; `edge_count` has `n_sub` entries
        // and `edge_off` `n_sub + 1` (the count pass only writes the
        // former, the refine pass reads the latter, written by the host
        // scan); the outputs have `n_edges` entries (3 for `q`), and the
        // refine pass writes slot ids only below `edge_off[j + 1] ≤
        // n_edges`. List and value reads are bounded as in `run`. The host
        // touches no buffer until the synchronous launch returns.
        unsafe { self.device.launch_sync(function, grid, BLOCK, &mut args) }
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

/// `RefineParams` in `kernels/sesmesh.cuh`.
#[repr(C)]
struct RefineParams {
    sphere_lo: [*const c_void; 3],
    circle_lo: [*const c_void; 3],
    vertex_lo: [*const c_void; 3],
    edge_count: *mut c_void,
    edge_off: *const c_void,
    t_out: *mut c_void,
    feat_out: *mut c_void,
    q_out: *mut c_void,
    evals_out: *mut c_void,
    n_edges: u32,
    max_iter: u32,
    max_bisect: u32,
    pad0: u32,
    h_hi: f32,
    h_lo: f32,
    tol: f32,
    pad: f32,
}

crate::batch::forward_batch_traits!("hip");
