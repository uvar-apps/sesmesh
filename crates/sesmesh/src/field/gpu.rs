// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The mesh stage with classification, gathering and field sampling on a
//! HIP device (Plan T11.3′, docs/gpu-prior-art.md §3.2 and §5.2).
//!
//! The arrangement is uploaded once ([`Resident`]): the frame-local `f32`
//! copies the CPU kernels compute, the blocker tables and the CPU's three
//! cell grids. Bricks then go through in batches of whole mesh chunks. For
//! each batch the device classifies the sub-blocks, gathers their candidates
//! and samples every brick sample; the CPU threads mesh the chunks from those
//! values, refining vertices in `f64` as on the CPU. The device fills a
//! ring of [`Batch`] buffers ahead of the meshing threads, which take chunks
//! in order and wait only for their own batch.
//!
//! The device repeats the CPU's `f64` classification and gathering tests in
//! `f32` with a slack above the `f32` error on the inclusive side: its
//! uniform sub-blocks are a subset of the CPU's and its lists supersets.
//! Every sample the CPU evaluates gets the CPU's value bit for bit; samples
//! in sub-blocks that only the device evaluates get their value instead of
//! `±band`, with the same sign. Meshing reads the signs and the values at
//! crossing-edge ends, so the mesh equals the CPU back end's.

use std::sync::atomic::{AtomicU64, Ordering};

use sesmesh_gpu::hip::{ArrangementRef, Batch, BatchParams, Device, GridRef, Resident};

use super::{Engine, INSIDE_TOL};
use crate::control::{Control, Stage};
use crate::error::{Error, Result};
use crate::field::SurfaceKind;
use crate::field::lattice::CLASSIFY_EPS;
use crate::geom::{CellGrid, Status, Vec3};
use crate::par;

/// Bricks per mesh chunk on the device path (at most).
///
/// A batch is a number of chunks, and the device's first batch is not
/// overlapped by meshing: short chunks keep it short. The mesh
/// does not depend on the chunk length (the assembly numbers vertices and
/// triangles in brick order).
const DEVICE_CHUNK: usize = 64;

/// Chunk length of the device path for `bricks` bricks.
pub(crate) fn device_chunk(bricks: usize) -> usize {
    par::chunk_len(bricks).min(DEVICE_CHUNK)
}

/// Chunks per batch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Batching {
    pub(crate) chunks: usize,
}

impl Default for Batching {
    /// One chunk per thread: small batches keep the device's first batch
    /// (which no meshing overlaps) short, and the pipeline has no per-batch
    /// barrier (1AON at 0.25 Å, 16 threads: 1.46 s mesh stage; 1.48 s with
    /// 4 chunks per thread, 1.51 s with 16).
    fn default() -> Self {
        Self {
            chunks: par::threads(),
        }
    }
}

fn device_error(e: &sesmesh_gpu::Error) -> Error {
    Error::Backend(format!("hip: {e}"))
}

/// The arrangement as the CPU kernels see it: frame-local `f32` copies,
/// laid out for [`ArrangementRef`].
pub(super) struct HostArrangement {
    sphere: [Vec<f32>; 7],
    status: Vec<u32>,
    circle: [Vec<f32>; 10],
    vertex: [Vec<f32>; 3],
}

fn grid(g: &CellGrid, frame: Vec3) -> GridRef<'_> {
    let (origin, cell, dims, starts, items) = g.parts();
    let o = origin - frame;
    GridRef {
        origin: [o.x as f32, o.y as f32, o.z as f32],
        cell: cell as f32,
        dims: dims.map(|d| u32::try_from(d).expect("fewer than 2^32 cells per axis")),
        starts,
        items,
    }
}

impl Engine {
    /// Whether the device path can sample this engine's field: the `f32`
    /// kernels are in use and bricks split into 2×2×2 sub-blocks.
    pub(crate) fn device_ok(&self) -> bool {
        self.f32_ok && self.sub_blocks() == 8
    }

    /// The frame-local `f32` arrangement ([`Engine::device_ok`] engines).
    pub(super) fn host_arrangement(&self) -> HostArrangement {
        let a = &self.field.arr;
        let frame = self.lattice.frame_origin();
        let mut sphere: [Vec<f32>; 7] = Default::default();
        for v in &mut sphere {
            v.reserve_exact(a.centers.len());
        }
        for k in 0..a.centers.len() {
            let c = a.centers[k] - frame;
            let r32 = a.expanded[k] as f32;
            let rt = (r32 - INSIDE_TOL).max(0.0);
            let row = [
                c.x as f32,
                c.y as f32,
                c.z as f32,
                r32,
                r32 * r32,
                rt * rt,
                a.radii[k] as f32,
            ];
            for (v, x) in sphere.iter_mut().zip(row) {
                v.push(x);
            }
        }
        let status = a
            .status
            .iter()
            .map(|s| match s {
                Status::Exposed => 0,
                Status::Buried => 1,
                Status::Contained => 2,
            })
            .collect();
        let mut circle: [Vec<f32>; 10] = Default::default();
        for c in &a.circles {
            let m = c.center - frame;
            let row = [
                m.x as f32,
                m.y as f32,
                m.z as f32,
                c.axis.x as f32,
                c.axis.y as f32,
                c.axis.z as f32,
                c.e1.x as f32,
                c.e1.y as f32,
                c.e1.z as f32,
                c.radius as f32,
            ];
            for (v, x) in circle.iter_mut().zip(row) {
                v.push(x);
            }
        }
        let mut vertex: [Vec<f32>; 3] = Default::default();
        for v in &a.vertices {
            let q = v.position - frame;
            for (d, x) in vertex.iter_mut().zip([q.x as f32, q.y as f32, q.z as f32]) {
                d.push(x);
            }
        }
        HostArrangement {
            sphere,
            status,
            circle,
            vertex,
        }
    }

    /// Uploads the arrangement to `device`.
    ///
    /// # Errors
    /// [`Error::Backend`] if an allocation or copy fails.
    pub(super) fn resident<'d>(&self, device: &'d Device) -> Result<Resident<'d>> {
        let host = self.host_arrangement();
        let a = &self.field.arr;
        let fo = self.lattice.frame_origin();
        let aref = ArrangementRef {
            sphere: std::array::from_fn(|i| host.sphere[i].as_slice()),
            status: &host.status,
            neighbors: a.neighbors.parts(),
            circle: std::array::from_fn(|i| host.circle[i].as_slice()),
            circle_blockers: a.circle_blockers.parts(),
            vertex: std::array::from_fn(|i| host.vertex[i].as_slice()),
            sphere_grid: grid(&a.cells, fo),
            circle_grid: grid(&self.field.circle_cells, fo),
            vertex_grid: grid(&self.field.vertex_cells, fo),
            max_expanded: a.max_expanded as f32,
            rho_max: self.field.rho_max as f32,
        };
        Resident::new(device, &aref).map_err(|e| device_error(&e))
    }

    /// The device's batch parameters: the CPU's radii and margins
    /// (`lattice::uniform_sign`, `gather_ses`, `gather_spheres`) in `f32`,
    /// and the slack added to every `f32` decision.
    pub(super) fn batch_params(&self) -> BatchParams {
        let b = u32::try_from(self.lattice.b).expect("brick edge fits u32");
        let sub = b / 2;
        let h = self.lattice.h;
        let probe = self.field.probe;
        let band = self.band;
        let rho = 0.5 * 3f64.sqrt() * f64::from(sub) * h;
        let ses = self.field.kind == SurfaceKind::Ses;
        let reach = if ses { probe + band + rho } else { band + rho };
        let margin = rho + CLASSIFY_EPS;
        let cutoffc = (probe + margin + CLASSIFY_EPS) as f32;
        let cutoff = (probe + band) as f32;
        // f32 error of frame-local distances: a few ulps of the largest
        // coordinate, far below the slack (docs/gpu-prior-art.md §3.2).
        let m = self.coord_max.clamp(1.0, 1e6) as f32;
        let ulp = f32::from_bits(m.to_bits() + 1) - m;
        BatchParams {
            ses,
            probe: probe as f32,
            band: band as f32,
            cutoff2: cutoff * cutoff,
            rho: rho as f32,
            reach: reach as f32,
            margin: margin as f32,
            slack: 1e-3 + 16.0 * ulp,
            cutoffc2: cutoffc * cutoffc,
            h: h as f32,
            b,
            sub,
            near: false,
        }
    }

    /// Brick origins relative to the `f32` frame, in lattice units.
    pub(super) fn frame_origins(&self, bricks: &[[i32; 3]]) -> Vec<[i32; 3]> {
        let frame = self.lattice.frame();
        bricks
            .iter()
            .map(|&k| {
                let o = self.lattice.brick_origin(k);
                [0, 1, 2].map(|a| {
                    i32::try_from(o[a] - frame[a]).expect("frame-local brick origins fit i32")
                })
            })
            .collect()
    }
}

/// Progress over the bricks of a stage.
struct Progress<'a, 'c> {
    control: &'a Control<'c>,
    stage: Stage,
    done: AtomicU64,
    step: u64,
    total: u64,
}

impl Progress<'_, '_> {
    fn brick(&self) {
        if self.control.has_progress() {
            let d = self.done.fetch_add(1, Ordering::Relaxed) + 1;
            if d.is_multiple_of(self.step) && d < self.total {
                self.control.report(self.stage, d, self.total);
            }
        }
    }
}

/// Batch buffers in the ring: the device fills up to `RING - 1` batches
/// ahead of the oldest batch still being meshed.
const RING: usize = 3;

/// Pipeline state shared by the device thread and the meshing threads.
#[derive(Default)]
struct Pipeline {
    /// Batches the device has finished.
    ready: usize,
    /// Chunks meshed, per batch.
    meshed: Vec<usize>,
    /// The device thread has stopped (error or cancellation).
    stopped: bool,
    error: Option<Error>,
}

/// Waits on `cv` with a short timeout, so cancellation is seen promptly
/// even without a notification.
fn wait<'a, T>(
    cv: &std::sync::Condvar,
    g: std::sync::MutexGuard<'a, T>,
) -> std::sync::MutexGuard<'a, T> {
    cv.wait_timeout(g, std::time::Duration::from_millis(5))
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .0
}

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Engine {
    /// [`Engine::fold_brick_chunks_with`] with the field sampled on
    /// `device` ([`Engine::device_ok`] engines): same arguments, the same
    /// signs and crossing-edge values, hence the same meshes.
    ///
    /// A device thread classifies, gathers and samples batches of
    /// `batching.chunks` chunks into a ring of [`RING`] buffers, in order.
    /// The pool's threads take chunks in order and mesh each as soon as its
    /// batch is ready; a buffer is reused once all its chunks are meshed. No
    /// thread waits for a whole batch to be meshed.
    ///
    /// # Errors
    /// [`Error::Cancelled`]; [`Error::Backend`] if a device operation fails.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fold_brick_chunks_hip<A, I, F>(
        &self,
        device: &Device,
        batching: Batching,
        bricks: &[[i32; 3]],
        chunk: usize,
        control: &Control<'_>,
        stage: Stage,
        init: I,
        f: F,
    ) -> Result<Vec<A>>
    where
        A: Send,
        I: Fn() -> A + Sync + Send,
        F: Fn(&mut A, [i32; 3], &[f32]) + Sync + Send,
    {
        use std::sync::atomic::AtomicUsize;
        use std::sync::{Condvar, Mutex, RwLock};

        debug_assert!(self.device_ok(), "no device path for this engine");
        let total = bricks.len() as u64;
        let progress = Progress {
            control,
            stage,
            done: AtomicU64::new(0),
            step: (total / 100).max(1),
            total,
        };
        control.report(stage, 0, total);
        let chunk = chunk.max(1);
        let chunks = bricks.len().div_ceil(chunk);
        if chunks == 0 {
            control.report(stage, total, total);
            return Ok(Vec::new());
        }
        let resident = self.resident(device)?;
        let p = self.batch_params();
        control.check()?;
        let per_batch = batching.chunks.clamp(1, chunks);
        let batches = chunks.div_ceil(per_batch);
        let chunks_of = |k: usize| (k * per_batch)..chunks.min((k + 1) * per_batch);
        let bricks_of = |k: usize| {
            let c = chunks_of(k);
            (c.start * chunk)..bricks.len().min(c.end * chunk)
        };
        let ring = (0..RING.min(batches))
            .map(|_| Batch::new(device).map(RwLock::new))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| device_error(&e))?;
        let state = Mutex::new(Pipeline {
            meshed: vec![0; batches],
            ..Pipeline::default()
        });
        let cv = Condvar::new();
        let next = AtomicUsize::new(0);

        // Device thread: batch k into ring slot k % RING, once every chunk of
        // batch k − RING has been meshed.
        let produce = || {
            for k in 0..batches {
                let mut g = lock(&state);
                while k >= RING
                    && g.meshed[k - RING] < chunks_of(k - RING).len()
                    && !g.stopped
                    && !control.is_cancelled()
                {
                    g = wait(&cv, g);
                }
                if g.stopped || control.is_cancelled() {
                    break;
                }
                drop(g);
                let origins = self.frame_origins(&bricks[bricks_of(k)]);
                let mut slot = ring[k % RING]
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let ran = resident.run(&p, &origins, &mut slot);
                drop(slot);
                let mut g = lock(&state);
                match ran {
                    Ok(_) => g.ready = k + 1,
                    Err(e) => {
                        g.error = Some(device_error(&e));
                        break;
                    }
                }
                drop(g);
                cv.notify_all();
            }
            lock(&state).stopped = true;
            cv.notify_all();
        };

        // Meshing threads: chunks in order, each from its batch's slot.
        let consume = |_: usize| {
            let mut mine = Vec::new();
            let n = self.samples_per_brick();
            'chunks: loop {
                if control.is_cancelled() {
                    break;
                }
                let c = next.fetch_add(1, Ordering::Relaxed);
                if c >= chunks {
                    break;
                }
                let k = c / per_batch;
                let mut g = lock(&state);
                while g.ready <= k {
                    if g.stopped || control.is_cancelled() {
                        break 'chunks;
                    }
                    g = wait(&cv, g);
                }
                drop(g);
                let slot = ring[k % RING]
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let first = bricks_of(k).start;
                let mut acc = init();
                for i in par::chunk_range(bricks.len(), chunk, c) {
                    if control.is_cancelled() {
                        break 'chunks;
                    }
                    f(&mut acc, bricks[i], &slot.brick_values(i - first)[..n]);
                    progress.brick();
                }
                drop(slot);
                mine.push((c, acc));
                let mut g = lock(&state);
                g.meshed[k] += 1;
                let free = g.meshed[k] == chunks_of(k).len();
                drop(g);
                if free {
                    cv.notify_all();
                }
            }
            mine
        };

        let parts = std::thread::scope(|scope| {
            let device_side = scope.spawn(produce);
            let parts = par::map(par::threads(), consume);
            // Wake the device thread if the meshing threads stopped early.
            cv.notify_all();
            if device_side.join().is_err() {
                lock(&state).error = Some(Error::Internal("hip: device thread panicked".into()));
            }
            parts
        });
        if let Some(e) = lock(&state).error.take() {
            return Err(e);
        }
        control.check()?;
        let mut parts: Vec<(usize, A)> = parts.into_iter().flatten().collect();
        if parts.len() != chunks {
            return Err(Error::Internal(format!(
                "hip: {} of {chunks} chunks meshed",
                parts.len()
            )));
        }
        parts.sort_unstable_by_key(|p| p.0);
        control.report(stage, total, total);
        Ok(parts.into_iter().map(|p| p.1).collect())
    }
}

#[cfg(test)]
mod tests {
    use sesmesh_gpu::hip::Runtime;
    use sesmesh_testkit::clusters::{Cluster, Rng, cage, random_box, random_chain};

    use super::*;
    use crate::field::{ScratchPool, SesField, SimdLevel};
    use crate::geom::Sphere;
    use crate::mesh::extract::{ChunkMesh, assemble};
    use crate::surface::RefineMode;

    /// Device 0, or `None` (skip) without one unless `SESMESH_GPU_REQUIRE=1`.
    fn device() -> Option<Device> {
        match Runtime::load().and_then(|r| r.open(0)) {
            Ok(d) => Some(d),
            // Skipped silently: the library denies printing, also in tests.
            Err(_) if std::env::var("SESMESH_GPU_REQUIRE").as_deref() != Ok("1") => None,
            Err(e) => panic!("{e}"),
        }
    }

    fn spheres(c: &Cluster) -> (Vec<Sphere>, f64) {
        let s = c
            .centers
            .iter()
            .zip(&c.radii)
            .map(|(&p, &r)| Sphere::new(p, r))
            .collect();
        (s, c.probe)
    }

    /// Crambin (`data/1crn.cif`) with the default selection and radii.
    fn crambin() -> (Vec<Sphere>, f64) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/1crn.cif");
        let structure = crate::io::read_path(&path, &crate::io::ReadOptions::default()).unwrap();
        let mut d = crate::Diagnostics::new();
        let atoms = structure
            .select(&crate::Selection::default(), &mut d)
            .unwrap();
        let radii =
            crate::radii::assign(&atoms, &crate::radii::RadiusOptions::default(), &mut d).unwrap();
        (Sphere::from_atoms(&atoms, &radii).unwrap(), 1.4)
    }

    fn engine((spheres, probe): &(Vec<Sphere>, f64), kind: SurfaceKind, h: f64) -> Engine {
        let field = SesField::new(spheres, *probe, kind).unwrap();
        Engine::new(field, h, 10, SimdLevel::Auto).unwrap()
    }

    fn clusters() -> Vec<(Vec<Sphere>, f64)> {
        let mut rng = Rng::new(11);
        vec![
            spheres(&random_chain(&mut rng, 60)),
            spheres(&random_box(&mut rng, 80, 14.0)),
            spheres(&cage(7.0, 1.6, 1.4)),
            crambin(),
        ]
    }

    const BATCHINGS: [Batching; 3] = [
        Batching { chunks: 1 },
        Batching { chunks: 2 },
        Batching { chunks: usize::MAX },
    ];

    /// Per-brick sample values, chunk by chunk.
    fn collect(acc: &mut Vec<Vec<f32>>, _: [i32; 3], v: &[f32]) {
        acc.push(v.to_vec());
    }

    /// Every sample has the CPU's sign, and the CPU's value bit for bit
    /// unless the CPU gave it `±band` (a sub-block only the device samples).
    #[test]
    fn device_values_match_cpu_values() {
        let Some(device) = device() else { return };
        let pool = ScratchPool::default();
        let control = Control::new();
        for c in &clusters() {
            for kind in [SurfaceKind::Ses, SurfaceKind::Sas, SurfaceKind::Vdw] {
                for h in [0.3, 0.7] {
                    let e = engine(c, kind, h);
                    let band = e.band as f32;
                    let (_, bricks) = e.surface_bricks().unwrap();
                    let chunk = par::chunk_len(bricks.len()).max(3);
                    let cpu = e
                        .fold_brick_chunks_with(
                            &bricks,
                            chunk,
                            &control,
                            Stage::Mesh,
                            &pool,
                            Vec::new,
                            collect,
                        )
                        .unwrap();
                    let (mut same, mut uniform) = (0usize, 0usize);
                    for batching in BATCHINGS.into_iter().chain([Batching::default()]) {
                        let gpu = e
                            .fold_brick_chunks_hip(
                                &device,
                                batching,
                                &bricks,
                                chunk,
                                &control,
                                Stage::Mesh,
                                Vec::new,
                                collect,
                            )
                            .unwrap();
                        assert_eq!(gpu.len(), cpu.len());
                        for (g, c) in gpu.iter().flatten().zip(cpu.iter().flatten()) {
                            for (&d, &v) in g.iter().zip(c) {
                                assert_eq!(d < 0.0, v < 0.0, "{kind:?} h {h} {batching:?}");
                                if d.to_bits() == v.to_bits() {
                                    same += 1;
                                } else {
                                    assert_eq!(v.abs(), band, "{kind:?} h {h} {batching:?}");
                                    uniform += 1;
                                }
                            }
                        }
                    }
                    assert!(same > 20 * uniform, "{kind:?} h {h}: {same} {uniform}");
                }
            }
        }
    }

    /// The device path's meshes equal the CPU path's, for any batch size.
    #[test]
    fn device_meshes_equal_cpu_meshes() {
        let Some(device) = device() else { return };
        let pool = ScratchPool::default();
        let control = Control::new();
        for c in &clusters() {
            for kind in [SurfaceKind::Ses, SurfaceKind::Sas, SurfaceKind::Vdw] {
                let e = engine(c, kind, 0.4);
                let (_, bricks) = e.surface_bricks().unwrap();
                let push = |m: &mut ChunkMesh, key, values: &[f32]| {
                    m.push(&e, key, values, RefineMode::Exact);
                };
                let chunk = par::chunk_len(bricks.len());
                let parts = e
                    .fold_brick_chunks_with(
                        &bricks,
                        chunk,
                        &control,
                        Stage::Mesh,
                        &pool,
                        ChunkMesh::default,
                        push,
                    )
                    .unwrap();
                let cpu = assemble(&bricks, chunk, parts).unwrap();
                assert!(!cpu.triangles.is_empty());
                let chunk = device_chunk(bricks.len());
                for batching in BATCHINGS {
                    let parts = e
                        .fold_brick_chunks_hip(
                            &device,
                            batching,
                            &bricks,
                            chunk,
                            &control,
                            Stage::Mesh,
                            ChunkMesh::default,
                            push,
                        )
                        .unwrap();
                    let gpu = assemble(&bricks, chunk, parts).unwrap();
                    let what = format!("{kind:?} {batching:?}");
                    assert_eq!(gpu.positions, cpu.positions, "{what}");
                    assert_eq!(gpu.normals, cpu.normals, "{what}");
                    assert_eq!(gpu.triangles, cpu.triangles, "{what}");
                    assert_eq!(gpu.vertex_atom, cpu.vertex_atom, "{what}");
                    assert_eq!(gpu.vertex_patch, cpu.vertex_patch, "{what}");
                }
            }
        }
    }

    #[test]
    fn cancellation_stops_device_batches() {
        let Some(device) = device() else { return };
        let mut rng = Rng::new(3);
        let e = engine(
            &spheres(&random_box(&mut rng, 60, 12.0)),
            SurfaceKind::Ses,
            0.4,
        );
        let (_, bricks) = e.surface_bricks().unwrap();
        let flag = std::sync::atomic::AtomicBool::new(true);
        let control = Control::new().with_cancel(&flag);
        let r = e.fold_brick_chunks_hip(
            &device,
            Batching::default(),
            &bricks,
            4,
            &control,
            Stage::Mesh,
            Vec::new,
            collect,
        );
        assert!(matches!(r, Err(Error::Cancelled)));
    }
}
