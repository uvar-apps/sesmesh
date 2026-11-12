//! Per-brick candidate gathering and the SIMD field kernel (Design §7.4–7.5).
//!
//! Lanes are samples. Kernels are generic over [`fearless_simd::Simd`] and
//! dispatched once per brick to the best instruction set detected at runtime
//! (AVX-512, AVX2, SSE4.2, NEON, WASM SIMD, or the scalar fallback).
//! Bricks are processed in parallel; results come back in brick order.

use fearless_simd::{Level, Simd, dispatch, f32x8, prelude::*};

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::lattice::{self, Lattice};
use super::point::SesField;
use super::{SimdLevel, SurfaceKind};
use crate::control::{Control, Stage};
use crate::error::{Error, Result};
use crate::geom::{Status, Vec3};
use crate::par;

/// Largest vector width (f32 lanes, AVX-512); sample buffers are padded to a
/// multiple of it so no scalar tail path exists (bitwise consistency, P7).
const MAX_LANES: usize = 16;

/// Tolerance (Å) of the `f32` "strictly inside a blocker" test.
const INSIDE_TOL: f32 = 2e-5;

/// Coordinates beyond this distance from the frame origin (Å) would lose too
/// much `f32` precision; such inputs use the exact `f64` path instead.
const F32_MAX_COORD: f64 = 2000.0;

/// Candidates of one brick in frame-local `f32` structure-of-arrays form.
#[derive(Debug, Default)]
struct Gathered {
    // Spheres that may contain a sample or a candidate point ("near"); the
    // first `n_inside` can contain a sample of the brick.
    n_inside: usize,
    nx: Vec<f32>,
    ny: Vec<f32>,
    nz: Vec<f32>,
    nr: Vec<f32>,
    nr2: Vec<f32>,
    nr2tol: Vec<f32>,
    // Exposed spheres whose surface is within reach (class S).
    sx: Vec<f32>,
    sy: Vec<f32>,
    sz: Vec<f32>,
    sr: Vec<f32>,
    s_off: Vec<u32>,
    s_blk: Vec<u32>,
    // Exposed circles within reach (class C).
    cx: Vec<f32>,
    cy: Vec<f32>,
    cz: Vec<f32>,
    ux: Vec<f32>,
    uy: Vec<f32>,
    uz: Vec<f32>,
    ex: Vec<f32>,
    ey: Vec<f32>,
    ez: Vec<f32>,
    rho: Vec<f32>,
    c_off: Vec<u32>,
    c_blk: Vec<u32>,
    // SAS vertices within reach (class V).
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
}

impl Gathered {
    fn clear(&mut self) {
        for v in [
            &mut self.nx,
            &mut self.ny,
            &mut self.nz,
            &mut self.nr,
            &mut self.nr2,
            &mut self.nr2tol,
            &mut self.sx,
            &mut self.sy,
            &mut self.sz,
            &mut self.sr,
            &mut self.cx,
            &mut self.cy,
            &mut self.cz,
            &mut self.ux,
            &mut self.uy,
            &mut self.uz,
            &mut self.ex,
            &mut self.ey,
            &mut self.ez,
            &mut self.rho,
            &mut self.vx,
            &mut self.vy,
            &mut self.vz,
        ] {
            v.clear();
        }
        self.n_inside = 0;
        self.s_off.clear();
        self.s_off.push(0);
        self.s_blk.clear();
        self.c_off.clear();
        self.c_off.push(0);
        self.c_blk.clear();
    }
}

/// Per-thread buffers reused across bricks.
#[derive(Debug, Default)]
pub(crate) struct BrickScratch {
    g: Gathered,
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    /// Kernel output for the compacted active samples.
    out: Vec<f32>,
    /// Values of all samples of the brick (exact or sign-only, see
    /// [`Engine::evaluate`]).
    values: Vec<f32>,
    /// Brick sample indices of the active (exactly evaluated) samples,
    /// grouped by the sub-block that evaluates them.
    groups: [Vec<u32>; 8],
    /// Global sphere index → local "near" index (`u32::MAX`: not near).
    local_of: Vec<u32>,
    touched: Vec<u32>,
    order: Vec<(f64, u32)>,
}

/// Per-thread brick buffers kept between computations
/// ([`SurfaceEngine`](crate::SurfaceEngine), Spec FR-LIB-3).
#[derive(Debug, Default)]
pub(crate) struct ScratchPool(Mutex<Vec<BrickScratch>>);

impl ScratchPool {
    /// A buffer sized for `engine` (reused if one is pooled).
    fn take<'p>(&'p self, engine: &Engine) -> Pooled<'p> {
        let reused = self.0.lock().ok().and_then(|mut v| v.pop());
        let s = match reused {
            Some(mut s) if s.values.len() == engine.scratch_len() => {
                // `local_of` is all-`u32::MAX` between bricks.
                s.local_of.resize(engine.field.arr.centers.len(), u32::MAX);
                s
            }
            _ => engine.scratch(),
        };
        Pooled {
            scratch: s,
            pool: self,
        }
    }
}

/// A pooled buffer, returned to its pool when dropped.
struct Pooled<'p> {
    scratch: BrickScratch,
    pool: &'p ScratchPool,
}

impl Drop for Pooled<'_> {
    fn drop(&mut self) {
        if let Ok(mut v) = self.pool.0.lock() {
            v.push(std::mem::take(&mut self.scratch));
        }
    }
}

/// Samples the field on the sparse brick lattice.
#[derive(Debug)]
pub(crate) struct Engine {
    pub(crate) field: SesField,
    pub(crate) lattice: Lattice,
    level: Level,
    /// Half-width of the band where sampled values are exact (Design §7.2).
    pub(crate) band: f64,
    f32_ok: bool,
}

impl Engine {
    /// # Errors
    /// [`Error::InvalidInput`] for an invalid spacing or brick size.
    pub(crate) fn new(
        field: SesField,
        spacing: f64,
        brick: usize,
        simd: SimdLevel,
    ) -> Result<Self> {
        if !(spacing.is_finite() && spacing >= 0.01) {
            return Err(Error::InvalidInput(format!(
                "grid spacing must be at least 0.01 Å, got {spacing}"
            )));
        }
        if !(2..=32).contains(&brick) {
            return Err(Error::InvalidInput(format!(
                "brick size must be 2..=32 cells, got {brick}"
            )));
        }
        let a = &field.arr;
        let (center, half) = match a.centers.first() {
            None => (Vec3::default(), 0.0),
            Some(&first) => {
                let (lo, hi) = a
                    .centers
                    .iter()
                    .fold((first, first), |(lo, hi), &c| (lo.min(c), hi.max(c)));
                let d = hi - lo;
                ((lo + hi) * 0.5, 0.5 * d.x.max(d.y).max(d.z))
            }
        };
        let lattice = Lattice::new(spacing, brick, center);
        let reach = half + 2.0 * a.max_expanded + 2.0 * field.probe + 4.0 * brick as f64 * spacing;
        let f32_ok = reach < F32_MAX_COORD && reach / spacing < f64::from(1u32 << 23);
        let level = match simd {
            SimdLevel::Auto => Level::new(),
            SimdLevel::Scalar => Level::fallback(),
        };
        Ok(Self {
            field,
            lattice,
            level,
            band: 3f64.sqrt() * spacing * 1.01,
            f32_ok,
        })
    }

    /// Name of the instruction set used by the kernel.
    pub(crate) fn simd_name(&self) -> &'static str {
        if !self.f32_ok {
            return "scalar-f64";
        }
        level_name(self.level)
    }

    /// Samples per brick: `(b + 1)³`, x varying fastest.
    pub(crate) fn samples_per_brick(&self) -> usize {
        let s = (self.lattice.b + 1) as usize;
        s * s * s
    }

    fn scratch_len(&self) -> usize {
        self.samples_per_brick().div_ceil(MAX_LANES) * MAX_LANES
    }

    pub(crate) fn scratch(&self) -> BrickScratch {
        let padded = self.scratch_len();
        BrickScratch {
            g: Gathered::default(),
            xs: vec![0.0; padded],
            ys: vec![0.0; padded],
            zs: vec![0.0; padded],
            out: vec![0.0; padded],
            values: vec![0.0; padded],
            groups: Default::default(),
            local_of: vec![u32::MAX; self.field.arr.centers.len()],
            touched: Vec::new(),
            order: Vec::new(),
        }
    }

    /// Bricks the surface can pass through, sorted by packed key.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if coordinates exceed the brick key range.
    pub(crate) fn surface_bricks(&self) -> Result<(usize, Vec<[i32; 3]>)> {
        let touched = lattice::touched_bricks(&self.field, &self.lattice)?;
        let surface = lattice::surface_bricks(&self.field, &self.lattice, &touched);
        Ok((touched.len(), surface))
    }

    /// Evaluates `f` on the samples of every brick, in parallel, returning
    /// results in brick order.
    pub(crate) fn map_bricks<T, F>(&self, bricks: &[[i32; 3]], f: F) -> Vec<T>
    where
        T: Send,
        F: Fn([i32; 3], &[f32]) -> T + Sync + Send,
    {
        let n = self.samples_per_brick();
        par::map_init(
            bricks.len(),
            || self.scratch(),
            |scratch, i| {
                self.evaluate(bricks[i], scratch);
                f(bricks[i], &scratch.values[..n])
            },
        )
    }

    /// [`Engine::map_bricks`] with cancellation (checked per brick),
    /// progress reports for `stage`, and per-thread buffers taken from and
    /// returned to `pool`.
    ///
    /// # Errors
    /// [`Error::Cancelled`].
    pub(crate) fn map_bricks_with<T, F>(
        &self,
        bricks: &[[i32; 3]],
        control: &Control<'_>,
        stage: Stage,
        pool: &ScratchPool,
        f: F,
    ) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn([i32; 3], &[f32]) -> T + Sync + Send,
    {
        let n = self.samples_per_brick();
        let total = bricks.len() as u64;
        let step = (total / 100).max(1);
        let done = AtomicU64::new(0);
        control.report(stage, 0, total);
        let out = par::map_init(
            bricks.len(),
            || pool.take(self),
            |scratch, i| {
                if control.is_cancelled() {
                    return None;
                }
                let s = &mut scratch.scratch;
                self.evaluate(bricks[i], s);
                let r = f(bricks[i], &s.values[..n]);
                if control.has_progress() {
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d.is_multiple_of(step) && d < total {
                        control.report(stage, d, total);
                    }
                }
                Some(r)
            },
        );
        control.check()?;
        control.report(stage, total, total);
        Ok(out.into_iter().flatten().collect())
    }

    /// Samples of brick `key` into `scratch.values[..samples_per_brick()]`.
    ///
    /// The brick's 2×2×2 sub-blocks are classified with the same conservative
    /// test as bricks (Design §7.3). Samples lying only in sub-blocks of
    /// provably uniform sign get `±band` (sign exact, value not); all others
    /// get the exact clamped field from the SIMD kernel. Meshing needs exact
    /// values only next to sign changes, which never involve uniform
    /// sub-blocks; signs are always exact, so neighbouring bricks agree on
    /// every inside/outside decision.
    pub(crate) fn evaluate(&self, key: [i32; 3], s: &mut BrickScratch) {
        let b = self.lattice.b;
        let side = b + 1;
        let origin = self.lattice.brick_origin(key);
        let band = self.band as f32;

        // Sub-block classification.
        let sub = if b % 2 == 0 && b >= 4 { b / 2 } else { b };
        let nsub = b / sub;
        let mut state = [None::<bool>; 8];
        if nsub == 2 {
            let base = self.lattice.point(origin);
            let h = self.lattice.h;
            let rho = 0.5 * 3f64.sqrt() * sub as f64 * h;
            for (i, st) in state.iter_mut().enumerate() {
                let half = sub as f64 * 0.5;
                let off = |bit: usize| ((i >> bit) & 1) as f64 * sub as f64 + half;
                let center = base + Vec3::new(off(0), off(1), off(2)) * h;
                *st = lattice::uniform_sign(&self.field, center, rho);
            }
        }
        let range = |v: i64| {
            let hi = (v / sub).min(nsub - 1);
            let lo = if v > 0 && v % sub == 0 {
                (v / sub - 1).min(nsub - 1)
            } else {
                hi
            };
            (lo as usize, hi as usize)
        };

        for group in &mut s.groups {
            group.clear();
        }
        let mut i = 0;
        for z in 0..side {
            let (z0, z1) = range(z);
            for y in 0..side {
                let (y0, y1) = range(y);
                for x in 0..side {
                    let (x0, x1) = range(x);
                    // The first non-uniform sub-block containing the sample
                    // evaluates it; otherwise its sign is known.
                    let mut owner = None;
                    let mut positive = true;
                    'blocks: for sz in z0..=z1 {
                        for sy in y0..=y1 {
                            for sx in x0..=x1 {
                                let sb = sx + 2 * sy + 4 * sz;
                                match state[sb] {
                                    None => {
                                        owner = Some(sb);
                                        break 'blocks;
                                    }
                                    Some(sg) => positive = sg,
                                }
                            }
                        }
                    }
                    match owner {
                        None => s.values[i] = if positive { band } else { -band },
                        Some(sb) if self.f32_ok => s.groups[sb].push(i as u32),
                        Some(_) => {
                            let idx = [origin[0] + x, origin[1] + y, origin[2] + z];
                            let v = self.field.clamped_value(self.lattice.point(idx), self.band);
                            s.values[i] = v as f32;
                        }
                    }
                    i += 1;
                }
            }
        }
        if !self.f32_ok {
            return;
        }
        let probe = self.field.probe as f32;
        let cutoff = (self.field.probe + self.band) as f32;
        let h = self.lattice.h;
        let base = self.lattice.point(origin);
        let (sub_rho, half) = (0.5 * 3f64.sqrt() * sub as f64 * h, sub as f64 * 0.5);
        for sb in 0..(nsub * nsub * nsub) as usize {
            let m = s.groups[sb].len();
            if m == 0 {
                continue;
            }
            // Gather candidates for this sub-block only (tighter reach).
            let (center, rho) = if nsub == 2 {
                let off = |bit: usize| ((sb >> bit) & 1) as f64 * sub as f64 + half;
                (base + Vec3::new(off(0), off(1), off(2)) * h, sub_rho)
            } else {
                (self.lattice.brick_center(key), self.lattice.brick_radius())
            };
            let side_u = side as usize;
            for (j, &i) in s.groups[sb].iter().enumerate() {
                let i = i as usize;
                let (x, y, z) = (i % side_u, (i / side_u) % side_u, i / (side_u * side_u));
                let idx = [
                    origin[0] + x as i64,
                    origin[1] + y as i64,
                    origin[2] + z as i64,
                ];
                let [lx, ly, lz] = self.lattice.local(idx);
                (s.xs[j], s.ys[j], s.zs[j]) = (lx, ly, lz);
            }
            let padded = m.div_ceil(MAX_LANES) * MAX_LANES;
            for k in m..padded {
                (s.xs[k], s.ys[k], s.zs[k]) = (s.xs[m - 1], s.ys[m - 1], s.zs[m - 1]);
            }
            match self.field.kind {
                SurfaceKind::Ses => {
                    self.gather_ses(center, rho, s);
                    let BrickScratch {
                        g, xs, ys, zs, out, ..
                    } = s;
                    let (xs, ys, zs, out) = (
                        &xs[..padded],
                        &ys[..padded],
                        &zs[..padded],
                        &mut out[..padded],
                    );
                    dispatch!(self.level, simd => ses_kernel(simd, g, xs, ys, zs, probe, cutoff * cutoff, out));
                }
                SurfaceKind::Sas | SurfaceKind::Vdw => {
                    self.gather_spheres(center, rho, self.band, s);
                    let BrickScratch {
                        g, xs, ys, zs, out, ..
                    } = s;
                    let (xs, ys, zs, out) = (
                        &xs[..padded],
                        &ys[..padded],
                        &zs[..padded],
                        &mut out[..padded],
                    );
                    dispatch!(self.level, simd => sphere_kernel(simd, g, xs, ys, zs, band, out));
                }
            }
            for (j, &i) in s.groups[sb].iter().enumerate() {
                s.values[i as usize] = s.out[j];
            }
        }
    }

    /// Collects the spheres within `reach` of `b` (beyond their expanded
    /// radius) into the "near" arrays: first those that can contain a sample
    /// of the ball of radius `rho` around `b`, then the rest (possible
    /// blockers).
    fn gather_near(&self, b: Vec3, rho: f64, reach: f64, s: &mut BrickScratch) {
        let a = &self.field.arr;
        let frame = self.lattice.frame_origin();
        s.g.clear();
        s.touched.clear();
        let BrickScratch {
            g,
            local_of,
            touched,
            ..
        } = s;
        for pass in 0..2 {
            a.cells.for_each_near(b, reach + a.max_expanded, |k| {
                let ku = k as usize;
                let r = a.expanded[ku];
                let d = b.dist(a.centers[ku]);
                let contains_sample = d < r + rho;
                if a.status[ku] == Status::Contained
                    || d >= r + reach
                    || contains_sample != (pass == 0)
                {
                    return;
                }
                local_of[ku] = g.nx.len() as u32;
                touched.push(k);
                let c = a.centers[ku] - frame;
                g.nx.push(c.x as f32);
                g.ny.push(c.y as f32);
                g.nz.push(c.z as f32);
                let r32 = r as f32;
                g.nr.push(r32);
                g.nr2.push(r32 * r32);
                let rt = (r32 - INSIDE_TOL).max(0.0);
                g.nr2tol.push(rt * rt);
            });
            if pass == 0 {
                g.n_inside = g.nx.len();
            }
        }
    }

    fn reset_local(&self, s: &mut BrickScratch) {
        for &k in &s.touched {
            s.local_of[k as usize] = u32::MAX;
        }
    }

    fn gather_spheres(&self, b: Vec3, rho: f64, band: f64, s: &mut BrickScratch) {
        self.gather_near(b, rho, band + rho, s);
        self.reset_local(s);
    }

    /// Candidates of the three classes for every sample within `rho` of `b`
    /// (Design §7.4): anything that can be the closest point of `P` within
    /// the cutoff lies within `G = r_p + band + rho` of `b`.
    fn gather_ses(&self, b: Vec3, rho: f64, s: &mut BrickScratch) {
        let a = &self.field.arr;
        let frame = self.lattice.frame_origin();
        let reach = self.field.probe + self.band + rho;
        self.gather_near(b, rho, reach, s);
        let BrickScratch {
            g,
            local_of,
            touched,
            order,
            ..
        } = s;

        // Class S, nearest first (better pruning); ties by index.
        order.clear();
        for &k in touched.iter() {
            let ku = k as usize;
            if a.status[ku] != Status::Exposed {
                continue;
            }
            let lb = (b.dist(a.centers[ku]) - a.expanded[ku]).abs();
            if lb <= reach {
                order.push((lb, k));
            }
        }
        order.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        for &(_, k) in order.iter() {
            let ku = k as usize;
            let local = local_of[ku] as usize;
            g.sx.push(g.nx[local]);
            g.sy.push(g.ny[local]);
            g.sz.push(g.nz[local]);
            g.sr.push(g.nr[local]);
            for &n in a.neighbors.row(ku) {
                let l = local_of[n as usize];
                if l != u32::MAX {
                    g.s_blk.push(l);
                }
            }
            g.s_off.push(g.s_blk.len() as u32);
        }

        // Class C.
        order.clear();
        self.field
            .circle_cells
            .for_each_near(b, reach + self.field.rho_max, |ci| {
                let c = &a.circles[ci as usize];
                let y = b - c.center;
                let along = y.dot(c.axis);
                let wl = (y - c.axis * along).norm();
                let dist = (along * along + (wl - c.radius) * (wl - c.radius)).sqrt();
                if dist <= reach {
                    order.push((dist, ci));
                }
            });
        order.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        for &(_, ci) in order.iter() {
            let c = &a.circles[ci as usize];
            let m = c.center - frame;
            g.cx.push(m.x as f32);
            g.cy.push(m.y as f32);
            g.cz.push(m.z as f32);
            g.ux.push(c.axis.x as f32);
            g.uy.push(c.axis.y as f32);
            g.uz.push(c.axis.z as f32);
            g.ex.push(c.e1.x as f32);
            g.ey.push(c.e1.y as f32);
            g.ez.push(c.e1.z as f32);
            g.rho.push(c.radius as f32);
            for &n in a.circle_blockers.row(ci as usize) {
                let l = local_of[n as usize];
                if l != u32::MAX {
                    g.c_blk.push(l);
                }
            }
            g.c_off.push(g.c_blk.len() as u32);
        }

        // Class V.
        self.field.vertex_cells.for_each_near(b, reach, |vi| {
            let p = a.vertices[vi as usize].position;
            if b.dist(p) <= reach {
                let q = p - frame;
                g.vx.push(q.x as f32);
                g.vy.push(q.y as f32);
                g.vz.push(q.z as f32);
            }
        });
        for &k in touched.iter() {
            local_of[k as usize] = u32::MAX;
        }
    }
}

fn level_name(level: Level) -> &'static str {
    if level.is_fallback() {
        return "scalar";
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if level.as_avx512().is_some() {
            return "avx512";
        }
        if level.as_avx2().is_some() {
            return "avx2";
        }
        if level.as_sse4_2().is_some() {
            return "sse4.2";
        }
        "sse2"
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = level;
        "neon"
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = level;
        "wasm-simd128"
    }
    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        let _ = level;
        "scalar"
    }
}

/// Clamped SES field `r_p − min(d(x, P), r_p + band)` for one brick
/// (Design §7.5). `cutoff2 = (r_p + band)²`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ses_kernel<S: Simd>(
    simd: S,
    g: &Gathered,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    probe: f32,
    cutoff2: f32,
    out: &mut [f32],
) {
    let w = f32x8::<S>::LEN;
    let splat = |v: f32| f32x8::<S>::splat(simd, v);
    for off in (0..xs.len()).step_by(w) {
        let x = f32x8::<S>::from_slice(simd, &xs[off..off + w]);
        let y = f32x8::<S>::from_slice(simd, &ys[off..off + w]);
        let z = f32x8::<S>::from_slice(simd, &zs[off..off + w]);

        // Inside the SAS? Otherwise the sample is in P: F = r_p.
        let zero = splat(0.0);
        let mut inside = zero.simd_lt(zero);
        for k in 0..g.n_inside {
            let dx = x - splat(g.nx[k]);
            let dy = y - splat(g.ny[k]);
            let dz = z - splat(g.nz[k]);
            inside |= (dx * dx + dy * dy + dz * dz).simd_lt(splat(g.nr2[k]));
        }
        if !inside.any_true() {
            splat(probe).store_slice(&mut out[off..off + w]);
            continue;
        }

        let mut best2 = splat(cutoff2);
        // Class V: SAS vertices (always valid).
        for v in 0..g.vx.len() {
            let dx = x - splat(g.vx[v]);
            let dy = y - splat(g.vy[v]);
            let dz = z - splat(g.vz[v]);
            best2 = best2.min(dx * dx + dy * dy + dz * dz);
        }
        // Class S: radial projections onto exposed spheres.
        for si in 0..g.sx.len() {
            let (cx, cy, cz, r) = (
                splat(g.sx[si]),
                splat(g.sy[si]),
                splat(g.sz[si]),
                splat(g.sr[si]),
            );
            let dx = x - cx;
            let dy = y - cy;
            let dz = z - cz;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let lb = (dist - r).abs();
            let lb2 = lb * lb;
            let mut valid = lb2.simd_lt(best2) & splat(1e-6).simd_lt(dist);
            if !valid.any_true() {
                continue;
            }
            let scale = r / dist;
            let qx = cx + dx * scale;
            let qy = cy + dy * scale;
            let qz = cz + dz * scale;
            for &k in &g.s_blk[g.s_off[si] as usize..g.s_off[si + 1] as usize] {
                let k = k as usize;
                let ex = qx - splat(g.nx[k]);
                let ey = qy - splat(g.ny[k]);
                let ez = qz - splat(g.nz[k]);
                valid &= (ex * ex + ey * ey + ez * ez).simd_ge(splat(g.nr2tol[k]));
                if !valid.any_true() {
                    break;
                }
            }
            best2 = valid.select(lb2, best2);
        }
        // Class C: projections onto exposed intersection circles.
        for ci in 0..g.cx.len() {
            let (mx, my, mz) = (splat(g.cx[ci]), splat(g.cy[ci]), splat(g.cz[ci]));
            let (ux, uy, uz) = (splat(g.ux[ci]), splat(g.uy[ci]), splat(g.uz[ci]));
            let rho = splat(g.rho[ci]);
            let yx = x - mx;
            let yy = y - my;
            let yz = z - mz;
            let along = yx * ux + yy * uy + yz * uz;
            let wx = yx - ux * along;
            let wy = yy - uy * along;
            let wz = yz - uz * along;
            let wl = (wx * wx + wy * wy + wz * wz).sqrt();
            let t = wl - rho;
            let d2 = along * along + t * t;
            let mut valid = d2.simd_lt(best2);
            if !valid.any_true() {
                continue;
            }
            // Equidistant lanes (on the axis) use a fixed circle point.
            let degenerate = wl.simd_lt(splat(1e-6));
            let s = rho / wl;
            let qx = degenerate.select(mx + splat(g.ex[ci]) * rho, mx + wx * s);
            let qy = degenerate.select(my + splat(g.ey[ci]) * rho, my + wy * s);
            let qz = degenerate.select(mz + splat(g.ez[ci]) * rho, mz + wz * s);
            for &k in &g.c_blk[g.c_off[ci] as usize..g.c_off[ci + 1] as usize] {
                let k = k as usize;
                let ex = qx - splat(g.nx[k]);
                let ey = qy - splat(g.ny[k]);
                let ez = qz - splat(g.nz[k]);
                valid &= (ex * ex + ey * ey + ez * ez).simd_ge(splat(g.nr2tol[k]));
                if !valid.any_true() {
                    break;
                }
            }
            best2 = valid.select(d2, best2);
        }
        let f = splat(probe) - best2.sqrt();
        inside
            .select(f, splat(probe))
            .store_slice(&mut out[off..off + w]);
    }
}

/// Clamped union-of-spheres field `min(min_k(|x − c_k| − R_k), band)` (SAS
/// and vdW).
#[inline(always)]
fn sphere_kernel<S: Simd>(
    simd: S,
    g: &Gathered,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    band: f32,
    out: &mut [f32],
) {
    let w = f32x8::<S>::LEN;
    let splat = |v: f32| f32x8::<S>::splat(simd, v);
    for off in (0..xs.len()).step_by(w) {
        let x = f32x8::<S>::from_slice(simd, &xs[off..off + w]);
        let y = f32x8::<S>::from_slice(simd, &ys[off..off + w]);
        let z = f32x8::<S>::from_slice(simd, &zs[off..off + w]);
        let mut best = splat(band);
        for k in 0..g.nx.len() {
            let dx = x - splat(g.nx[k]);
            let dy = y - splat(g.ny[k]);
            let dz = z - splat(g.nz[k]);
            best = best.min((dx * dx + dy * dy + dz * dz).sqrt() - splat(g.nr[k]));
        }
        best.store_slice(&mut out[off..off + w]);
    }
}
