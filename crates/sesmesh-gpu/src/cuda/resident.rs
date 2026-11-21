// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classification, candidate gathering, field sampling and fast-mode edge
//! refinement on a CUDA device: the passes of `crate::hip::Resident`, with
//! the same kernels and descriptors (`kernels/sesmesh.cuh`).
//!
//! The difference is where a batch lives. On a discrete GPU the kernels
//! cannot afford to read lists and samples over the bus, so a [`Batch`]
//! keeps its working buffers in device memory. The host copies the sub-block
//! records down for the scans between passes (a few bytes per sub-block) and
//! the results it meshes from (sample values; in fast mode the edge
//! results) into pinned host buffers, which the accessors return.
//!
//! Device code is f32 and integer only.

use std::ffi::c_void;
use std::time::Instant;

use super::ffi::{DevicePtr, Handle};
use super::{Device, DeviceBuffer, HostBuffer};
use crate::batch::{
    ArrangementRef, BatchParams, BatchTimes, GridRef, REC_WORDS, RefineRef, RefineTimes,
};
use crate::{DeviceCopy, Result};

/// Threads per block (`LANES` in the kernels).
const BLOCK: u32 = 64;

/// `Grid` in `kernels/sesmesh.cuh`.
#[repr(C)]
struct GridParams {
    starts: DevicePtr,
    items: DevicePtr,
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
    sphere: [DevicePtr; 7],
    status: DevicePtr,
    nb_off: DevicePtr,
    nb_items: DevicePtr,
    circle: [DevicePtr; 10],
    cb_off: DevicePtr,
    cb_items: DevicePtr,
    vertex: [DevicePtr; 3],
    gs: GridParams,
    gc: GridParams,
    gv: GridParams,
    bricks: DevicePtr,
    rec: DevicePtr,
    ids: DevicePtr,
    values: DevicePtr,
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

/// `RefineParams` in `kernels/sesmesh.cuh`.
#[repr(C)]
struct RefineParams {
    sphere_lo: [DevicePtr; 3],
    circle_lo: [DevicePtr; 3],
    vertex_lo: [DevicePtr; 3],
    edge_count: DevicePtr,
    edge_off: DevicePtr,
    t_out: DevicePtr,
    feat_out: DevicePtr,
    q_out: DevicePtr,
    evals_out: DevicePtr,
    n_edges: u32,
    max_iter: u32,
    max_bisect: u32,
    pad0: u32,
    h_hi: f32,
    h_lo: f32,
    tol: f32,
    pad: f32,
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
            starts: self.starts.address(),
            items: self.items.address(),
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

impl std::fmt::Debug for Resident<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resident")
            .field("resident_bytes", &self.resident_bytes)
            .finish_non_exhaustive()
    }
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

/// A device buffer and the pinned host buffer it is copied to or from,
/// grown to the need of each batch.
struct Mirrored<T: DeviceCopy> {
    device: DeviceBuffer<T>,
    host: HostBuffer<T>,
}

impl<T: DeviceCopy> Mirrored<T> {
    fn new(d: &Device) -> Result<Self> {
        Ok(Self {
            device: d.alloc(0)?,
            host: d.alloc_host(0)?,
        })
    }

    /// Makes both hold at least `n` values (contents are lost on growth).
    fn reserve(&mut self, d: &Device, n: usize) -> Result<()> {
        if self.device.len() < n {
            self.device = d.alloc(0)?;
            self.device = d.alloc(n)?;
        }
        if self.host.len() < n {
            self.host = d.alloc_host(0)?;
            self.host = d.alloc_host(n)?;
        }
        Ok(())
    }

    /// Copies the first `n` host values to the device.
    fn upload(&mut self, n: usize) -> Result<()> {
        self.device.write_prefix(&self.host[..n])
    }

    /// Copies the first `n` device values to the host.
    fn download(&mut self, n: usize) -> Result<()> {
        self.device.read_prefix(&mut self.host[..n])
    }

    fn bytes(&self) -> usize {
        size_of::<T>() * (self.device.len() + self.host.len())
    }
}

/// The buffers of one batch: working buffers in device memory, and pinned
/// host copies of what the host reads.
pub struct Batch<'d> {
    device: &'d Device,
    bricks: Mirrored<i32>,
    rec: Mirrored<u32>,
    ids: DeviceBuffer<u32>,
    values: Mirrored<f32>,
    n_bricks: usize,
    spb: usize,
    n_ids: usize,
    edge_count: Mirrored<u32>,
    edge_off: Mirrored<u32>,
    t: Mirrored<f32>,
    feat: Mirrored<u32>,
    q: Mirrored<f32>,
    evals: Mirrored<u32>,
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
            bricks: Mirrored::new(device)?,
            rec: Mirrored::new(device)?,
            ids: device.alloc(0)?,
            values: Mirrored::new(device)?,
            n_bricks: 0,
            spb: 0,
            n_ids: 0,
            edge_count: Mirrored::new(device)?,
            edge_off: Mirrored::new(device)?,
            t: Mirrored::new(device)?,
            feat: Mirrored::new(device)?,
            q: Mirrored::new(device)?,
            evals: Mirrored::new(device)?,
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

    /// Candidate lists of the last run, copied from the device; a record's
    /// lists start at its base.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if the copy fails.
    pub fn id_lists(&self) -> Result<Vec<u32>> {
        let mut v = vec![0; self.n_ids];
        self.ids.read_prefix(&mut v)?;
        Ok(v)
    }

    /// Sample values of the last run, `(b + 1)³` per brick, x fastest.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values.host[..self.spb * self.n_bricks]
    }

    /// Sample values of brick `i` of the last run.
    ///
    /// # Panics
    ///
    /// If `i` is not a brick of the last run.
    #[must_use]
    pub fn brick_values(&self, i: usize) -> &[f32] {
        assert!(i < self.n_bricks, "brick {i} of {}", self.n_bricks);
        &self.values.host[i * self.spb..(i + 1) * self.spb]
    }

    /// Sub-block records of the last run ([`REC_WORDS`] each, 8 per brick).
    #[must_use]
    pub fn records(&self) -> &[u32] {
        &self.rec.host[..REC_WORDS * 8 * self.n_bricks]
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
            &self.edge_off.host[..8 * self.n_bricks + 1]
        } else {
            &[]
        }
    }

    /// Per refined edge: the edge parameter of the last evaluation (from the
    /// lower sample, not clamped).
    #[must_use]
    pub fn edge_params(&self) -> &[f32] {
        &self.t.host[..self.n_edges()]
    }

    /// Per refined edge: the feature of the last evaluation,
    /// `kind << FEATURE_SHIFT | index` (`FEATURE_SOLVENT` …).
    #[must_use]
    pub fn edge_features(&self) -> &[u32] {
        &self.feat.host[..self.n_edges()]
    }

    /// Per refined edge: the closest point of the last evaluation (SES; the
    /// evaluated point for spheres), relative to the brick origin, 3 each.
    #[must_use]
    pub fn edge_closest(&self) -> &[f32] {
        &self.q.host[..3 * self.n_edges()]
    }

    /// Per refined edge: field evaluations, with `EDGE_CONVERGED` set where
    /// Newton / Illinois converged.
    #[must_use]
    pub fn edge_evals(&self) -> &[u32] {
        &self.evals.host[..self.n_edges()]
    }

    /// Device and pinned host bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bricks.bytes()
            + self.rec.bytes()
            + 4 * self.ids.len()
            + self.values.bytes()
            + self.edge_count.bytes()
            + self.edge_off.bytes()
            + self.t.bytes()
            + self.feat.bytes()
            + self.q.bytes()
            + self.evals.bytes()
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
    /// frame origin, in lattice units) are `bricks`, into `batch`, and copies
    /// the records and sample values to its host buffers.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Call`] if an allocation, copy or launch fails.
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
        let d = self.device;
        let mut times = BatchTimes::default();
        let n_sub = 8 * bricks.len();
        let side = p.b as usize + 1;
        let spb = side * side * side;
        batch.bricks.reserve(d, 3 * bricks.len())?;
        for (dst, s) in batch
            .bricks
            .host
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(bricks)
        {
            dst.copy_from_slice(s);
        }
        batch.bricks.upload(3 * bricks.len())?;
        batch.rec.reserve(d, REC_WORDS * n_sub)?;
        batch.values.reserve(d, spb * bricks.len())?;
        batch.n_bricks = bricks.len();
        batch.spb = spb;
        batch.n_ids = 0;
        batch.n_edges = None;
        if bricks.is_empty() {
            return Ok(times);
        }
        let grid = u32::try_from(n_sub).expect("fewer than 2^32 sub-blocks");

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        self.launch(d.kernels().resident_count, grid, &mut params)?;
        times.count = t.elapsed().as_secs_f64();

        let t = Instant::now();
        batch.rec.download(REC_WORDS * n_sub)?;
        let mut total = 0u64;
        for r in batch.rec.host[..REC_WORDS * n_sub]
            .as_chunks_mut::<REC_WORDS>()
            .0
        {
            r[1] = u32::try_from(total).expect("fewer than 2^32 list ids per batch");
            total += r[2..7].iter().map(|&c| u64::from(c)).sum::<u64>();
        }
        batch.rec.upload(REC_WORDS * n_sub)?;
        let need = usize::try_from(total).expect("ids fit in memory");
        if batch.ids.len() < need {
            batch.ids = d.alloc(0)?;
            batch.ids = d.alloc(need)?;
        }
        batch.n_ids = need;
        times.scan = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        self.launch(d.kernels().resident_fill, grid, &mut params)?;
        times.fill = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        self.launch(d.kernels().resident_sample, grid, &mut params)?;
        batch.values.download(spb * bricks.len())?;
        batch.rec.download(REC_WORDS * n_sub)?;
        times.sample = t.elapsed().as_secs_f64();
        Ok(times)
    }

    fn params(&self, p: &BatchParams, n_sub: u32, batch: &Batch<'_>) -> Params {
        let len32 = |n: usize| u32::try_from(n).expect("fewer than 2^32 elements");
        let addr = |b: &DeviceBuffer<f32>| b.address();
        Params {
            sphere: std::array::from_fn(|i| addr(&self.sphere[i])),
            status: self.status.address(),
            nb_off: self.nb.0.address(),
            nb_items: self.nb.1.address(),
            circle: std::array::from_fn(|i| addr(&self.circle[i])),
            cb_off: self.cb.0.address(),
            cb_items: self.cb.1.address(),
            vertex: std::array::from_fn(|i| addr(&self.vertex[i])),
            gs: self.grids[0].params(),
            gc: self.grids[1].params(),
            gv: self.grids[2].params(),
            bricks: batch.bricks.device.address(),
            rec: batch.rec.device.address(),
            ids: batch.ids.address(),
            values: batch.values.device.address(),
            ids_len: batch.ids.len() as u64,
            values_len: batch.values.device.len() as u64,
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
    /// [`crate::Error::Call`] if an allocation, copy or launch fails.
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
        let d = self.device;
        let mut times = RefineTimes::default();
        let n_sub = 8 * batch.n_bricks;
        batch.edge_count.reserve(d, n_sub)?;
        batch.edge_off.reserve(d, n_sub + 1)?;
        batch.edge_off.host[0] = 0;
        if n_sub == 0 {
            batch.n_edges = Some(0);
            return Ok(times);
        }
        let grid = u32::try_from(n_sub).expect("fewer than 2^32 sub-blocks");

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        let mut rp = Self::refine_params(r, batch, 0);
        self.launch_refine(d.kernels().resident_edge_count, grid, &mut params, &mut rp)?;
        times.count = t.elapsed().as_secs_f64();

        let t = Instant::now();
        batch.edge_count.download(n_sub)?;
        let mut total = 0u32;
        for j in 0..n_sub {
            total = total
                .checked_add(batch.edge_count.host[j])
                .expect("fewer than 2^32 crossing edges per batch");
            batch.edge_off.host[j + 1] = total;
        }
        batch.edge_off.upload(n_sub + 1)?;
        let n = total as usize;
        batch.t.reserve(d, n)?;
        batch.feat.reserve(d, n)?;
        batch.q.reserve(d, 3 * n)?;
        batch.evals.reserve(d, n)?;
        times.scan = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut params = self.params(p, grid, batch);
        let mut rp = Self::refine_params(r, batch, total);
        self.launch_refine(d.kernels().resident_refine, grid, &mut params, &mut rp)?;
        batch.t.download(n)?;
        batch.feat.download(n)?;
        batch.q.download(3 * n)?;
        batch.evals.download(n)?;
        times.refine = t.elapsed().as_secs_f64();
        batch.n_edges = Some(n);
        Ok(times)
    }

    fn refine_params(r: &RefineData, batch: &Batch<'_>, n_edges: u32) -> RefineParams {
        let addr = |b: &DeviceBuffer<f32>| b.address();
        RefineParams {
            sphere_lo: std::array::from_fn(|i| addr(&r.sphere_lo[i])),
            circle_lo: std::array::from_fn(|i| addr(&r.circle_lo[i])),
            vertex_lo: std::array::from_fn(|i| addr(&r.vertex_lo[i])),
            edge_count: batch.edge_count.device.address(),
            edge_off: batch.edge_off.device.address(),
            t_out: batch.t.device.address(),
            feat_out: batch.feat.device.address(),
            q_out: batch.q.device.address(),
            evals_out: batch.evals.device.address(),
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
        function: Handle,
        grid: u32,
        params: &mut Params,
        rp: &mut RefineParams,
    ) -> Result<()> {
        let mut args: [*mut c_void; 2] = [
            std::ptr::from_mut(params).cast(),
            std::ptr::from_mut(rp).cast(),
        ];
        // SAFETY: `Params` and `RefineParams` by value, laid out as
        // `ResidentParams` and `RefineParams`. Every address is a live device
        // buffer of this device: the residuals have the arrangement's lengths
        // (`set_refinement`) and the kernels bound element indices by its
        // counts; `edge_count` has at least `n_sub` entries and `edge_off`
        // `n_sub + 1` (the count pass only writes the former, the refine
        // pass reads the latter, uploaded after the host scan); the outputs
        // have at least `n_edges` entries (3 for `q`), and the refine pass
        // writes slot ids only below `edge_off[j + 1] ≤ n_edges`. List and
        // value reads are bounded as in `run`. Nothing else uses the buffers
        // until the synchronous launch returns (`&mut Batch`).
        unsafe { self.device.launch_sync(function, grid, BLOCK, &mut args) }
    }

    fn launch(&self, function: Handle, grid: u32, params: &mut Params) -> Result<()> {
        let mut args: [*mut c_void; 1] = [std::ptr::from_mut(params).cast()];
        // SAFETY: `Params` by value, laid out as `ResidentParams`. Every
        // address is a live device buffer of this device with the length
        // passed alongside: the kernels bound element indices by the element
        // counts, CSR rows by the validated offsets (`new`), list writes by
        // the counts and `ids_len` (the buffer's length), and value writes by
        // `values_len`. Nothing else uses the buffers until the synchronous
        // launch returns (`&mut Batch` in `run`; the resident buffers are
        // only read).
        unsafe { self.device.launch_sync(function, grid, BLOCK, &mut args) }
    }
}

crate::batch::forward_batch_traits!("cuda");
