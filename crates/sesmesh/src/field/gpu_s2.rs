// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! S2 spike (`docs/gpu-prior-art.md` §5.2): crossing-edge refinement in
//! `f32` on the device (`sesmesh_resident_refine`) against the CPU's exact
//! `f64` refinement, vertex by vertex, and the time of each side.
//!
//! ```text
//! SESMESH_S2_INPUT=tests/data/cache/1aon.cif.gz SESMESH_S2_H=0.5,0.25 \
//!   cargo test --release -p sesmesh --features hip --lib s2 -- --ignored --nocapture
//! ```
//!
//! `SESMESH_S2_KIND` = `ses` (default), `sas` or `vdw`; `SESMESH_S2_QUICK=1`
//! skips the timings.

#![allow(clippy::print_stdout, reason = "the spike's report; test only")]

use std::time::Instant;

use sesmesh_gpu::hip::{Batch, Device, REC_WORDS, RefineRef, Refiner, Runtime, STATE_SAMPLE};

use super::Engine;
use crate::field::{Feature, SesField, SimdLevel, SurfaceKind};
use crate::geom::{Sphere, Vec3};
use crate::mesh::extract::{BrickMesh, Refined, mesh_brick, mesh_brick_refined};
use crate::par;
use crate::surface::RefineMode;

/// Bricks per device batch.
const BATCH: usize = 8192;
/// `vertex()` in `mesh/extract.rs`.
const TOL: f32 = 1e-5;
const MAX_ITER: u32 = 8;
const MAX_BISECT: u32 = 64;

fn load(path: &str) -> Option<Vec<Sphere>> {
    let structure = crate::io::read_path(
        std::path::Path::new(path),
        &crate::io::ReadOptions::default(),
    )
    .ok()?;
    let mut d = crate::Diagnostics::new();
    let atoms = structure
        .select(&crate::Selection::default(), &mut d)
        .ok()?;
    let radii =
        crate::radii::assign(&atoms, &crate::radii::RadiusOptions::default(), &mut d).ok()?;
    Sphere::from_atoms(&atoms, &radii).ok()
}

/// Crossing edges of a brick in `mesh_brick` order: device codes
/// (`x | y << 4 | z << 8 | axis << 12`) and sub-blocks.
fn crossing(b: usize, values: &[f32], codes: &mut Vec<u32>, subs: &mut Vec<u8>) {
    let s = b + 1;
    let sub = b / 2;
    let stride = [1, s, s * s];
    for z in 0..b {
        for y in 0..b {
            for x in 0..b {
                let i = (z * s + y) * s + x;
                let va = values[i];
                for (axis, &st) in stride.iter().enumerate() {
                    if (va < 0.0) != (values[i + st] < 0.0) {
                        codes.push((x | y << 4 | z << 8 | axis << 12) as u32);
                        let sb = [x, y, z].map(|c| (c / sub).min(1));
                        subs.push((sb[0] + 2 * sb[1] + 4 * sb[2]) as u8);
                    }
                }
            }
        }
    }
}

fn feature(f: u32) -> Feature {
    let id = f & ((1 << 29) - 1);
    match f >> 29 {
        0 => Feature::Solvent,
        1 => Feature::Beyond,
        2 => Feature::Sphere(id),
        3 => Feature::Circle(id),
        4 => Feature::Vertex(id),
        k => panic!("feature kind {k}"),
    }
}

/// Device groups of one batch: per brick, its edges grouped by sub-block;
/// `perm[k]` is the device index of the brick's `k`-th edge (CPU order).
#[derive(Default)]
struct Groups {
    groups: Vec<[u32; 4]>,
    edges: Vec<u32>,
    /// Per brick: first edge (device order) and the permutation.
    bricks: Vec<(usize, Vec<u32>)>,
}

#[derive(Debug, Default, Clone, Copy)]
struct Compare {
    vertices: usize,
    tris_bad: usize,
    sum_d: f64,
    max_d: f64,
    /// |Δpos| ≤ 1e-5, 1e-4, 1e-3, 1e-2, more (Å).
    hist: [usize; 5],
    max_f_dev: f64,
    sum_f_dev: f64,
    max_f_cpu: f64,
    sum_f_cpu: f64,
    over_tol_cpu: usize,
    over_tol_dev: usize,
    atom_diff: usize,
    patch_diff: usize,
    max_angle: f64,
    normal_over_1deg: usize,
    unconverged_dev: usize,
    evals_dev: u64,
    evals_cpu: u64,
}

impl Compare {
    fn add(mut self, o: Self) -> Self {
        self.vertices += o.vertices;
        self.tris_bad += o.tris_bad;
        self.sum_d += o.sum_d;
        self.max_d = self.max_d.max(o.max_d);
        for (a, b) in self.hist.iter_mut().zip(o.hist) {
            *a += b;
        }
        self.max_f_dev = self.max_f_dev.max(o.max_f_dev);
        self.sum_f_dev += o.sum_f_dev;
        self.max_f_cpu = self.max_f_cpu.max(o.max_f_cpu);
        self.sum_f_cpu += o.sum_f_cpu;
        self.over_tol_cpu += o.over_tol_cpu;
        self.over_tol_dev += o.over_tol_dev;
        self.atom_diff += o.atom_diff;
        self.patch_diff += o.patch_diff;
        self.max_angle = self.max_angle.max(o.max_angle);
        self.normal_over_1deg += o.normal_over_1deg;
        self.unconverged_dev += o.unconverged_dev;
        self.evals_dev += o.evals_dev;
        self.evals_cpu += o.evals_cpu;
        self
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Times {
    sample: f64,
    host: f64,
    refine: f64,
    cpu_exact: f64,
    cpu_placed: f64,
}

fn residuals(points: impl Iterator<Item = Vec3>, frame: Vec3) -> [Vec<f32>; 3] {
    let mut out: [Vec<f32>; 3] = Default::default();
    for c in points {
        let d = c - frame;
        for (v, x) in out.iter_mut().zip([d.x, d.y, d.z]) {
            v.push((x - f64::from(x as f32)) as f32);
        }
    }
    out
}

#[allow(clippy::too_many_lines, reason = "spike")]
fn run_one(device: &Device, spheres: &[Sphere], kind: SurfaceKind, h: f64) {
    let probe = if kind == SurfaceKind::Vdw { 0.0 } else { 1.4 };
    let field = SesField::new(spheres, probe, kind).unwrap();
    let engine = Engine::new(field, h, 10, SimdLevel::Auto).unwrap();
    if !engine.device_ok() {
        println!("S2 {kind:?} h={h} skipped: f64 path (coordinates beyond the f32 range)");
        return;
    }
    let quick = std::env::var("SESMESH_S2_QUICK").as_deref() == Ok("1");
    let reps = if quick { 1 } else { 3 };
    let (_, bricks) = engine.surface_bricks().unwrap();
    let origins = engine.frame_origins(&bricks);
    let res = engine.resident(device).unwrap();
    let a = &engine.field.arr;
    let frame = engine.lattice.frame_origin();
    let sphere_lo = residuals(a.centers.iter().copied(), frame);
    let circle_lo = residuals(a.circles.iter().map(|c| c.center), frame);
    let vertex_lo = residuals(a.vertices.iter().map(|v| v.position), frame);
    let h_hi = h as f32;
    let rref = RefineRef {
        sphere_lo: std::array::from_fn(|i| sphere_lo[i].as_slice()),
        circle_lo: std::array::from_fn(|i| circle_lo[i].as_slice()),
        vertex_lo: std::array::from_fn(|i| vertex_lo[i].as_slice()),
        h_hi,
        h_lo: (h - f64::from(h_hi)) as f32,
    };
    let mut refiner = Refiner::new(&res, &rref).unwrap();
    let mut batch = Batch::new(device).unwrap();
    let p = engine.batch_params();
    let lat = &engine.lattice;
    let b = lat.b as usize;
    let mut cmp = Compare::default();
    let mut times = Times::default();
    let mut evals_hist = [0usize; 12];
    let mut not_sampled = 0usize;
    let min_of = |reps: usize, f: &mut dyn FnMut()| {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    for start in (0..bricks.len()).step_by(BATCH) {
        let keys = &bricks[start..(start + BATCH).min(bricks.len())];
        let os = &origins[start..start + keys.len()];
        res.run(&p, os, &mut batch).unwrap();
        times.sample += min_of(reps, &mut || {
            res.run(&p, os, &mut batch).unwrap();
        });

        // Host: crossing edges grouped by sub-block (one pass over the
        // samples, as the cell pass).
        let mut g = Groups::default();
        times.host += min_of(reps, &mut || {
            let per = par::map_init(
                keys.len(),
                || (),
                |(), i| {
                    let (mut codes, mut subs) = (Vec::new(), Vec::new());
                    crossing(b, batch.brick_values(i), &mut codes, &mut subs);
                    (codes, subs)
                },
            );
            g = Groups::default();
            for (bi, (codes, subs)) in per.into_iter().enumerate() {
                let first = g.edges.len();
                let mut count = [0u32; 8];
                for &s in &subs {
                    count[s as usize] += 1;
                }
                let mut at = [0u32; 8];
                let mut acc = 0u32;
                for sb in 0..8 {
                    at[sb] = acc;
                    if count[sb] > 0 {
                        g.groups
                            .push([bi as u32, sb as u32, (first as u32) + acc, count[sb]]);
                    }
                    acc += count[sb];
                }
                g.edges.resize(first + codes.len(), 0);
                let mut perm = Vec::with_capacity(codes.len());
                for (&c, &s) in codes.iter().zip(&subs) {
                    let d = at[s as usize];
                    at[s as usize] += 1;
                    g.edges[first + d as usize] = c;
                    perm.push(d);
                }
                g.bricks.push((first, perm));
            }
        });
        let rec = batch.records();
        not_sampled += g
            .groups
            .iter()
            .filter(|gr| rec[REC_WORDS * (8 * gr[0] as usize + gr[1] as usize)] != STATE_SAMPLE)
            .count();

        res.refine(
            &p,
            &mut batch,
            &mut refiner,
            &g.groups,
            &g.edges,
            TOL,
            MAX_ITER,
            MAX_BISECT,
        )
        .unwrap();
        times.refine += min_of(reps, &mut || {
            res.refine(
                &p,
                &mut batch,
                &mut refiner,
                &g.groups,
                &g.edges,
                TOL,
                MAX_ITER,
                MAX_BISECT,
            )
            .unwrap();
        });
        for &e in refiner.evals() {
            evals_hist[((e & 0x7fff_ffff) as usize).min(11)] += 1;
        }
        let (t_dev, f_dev, q_dev, e_dev) = (
            refiner.t(),
            refiner.features(),
            refiner.closest(),
            refiner.evals(),
        );
        let placed = |i: usize, out: &mut Vec<Refined>| {
            let (first, perm) = &g.bricks[i];
            let base = lat.point(lat.brick_origin(keys[i]));
            out.clear();
            out.extend(perm.iter().map(|&d| {
                let k = first + d as usize;
                Refined {
                    t: f64::from(t_dev[k]),
                    feature: feature(f_dev[k]),
                    closest: base
                        + Vec3::new(
                            f64::from(q_dev[3 * k]),
                            f64::from(q_dev[3 * k + 1]),
                            f64::from(q_dev[3 * k + 2]),
                        ),
                }
            }));
        };

        if !quick {
            times.cpu_exact += min_of(reps, &mut || {
                par::map_init(keys.len(), BrickMesh::default, |m, i| {
                    m.clear();
                    mesh_brick(
                        &engine,
                        keys[i],
                        batch.brick_values(i),
                        RefineMode::Exact,
                        m,
                    );
                    m.triangles().len()
                });
            });
            times.cpu_placed += min_of(reps, &mut || {
                par::map_init(
                    keys.len(),
                    || (BrickMesh::default(), Vec::new()),
                    |(m, r), i| {
                        m.clear();
                        placed(i, r);
                        mesh_brick_refined(&engine, keys[i], batch.brick_values(i), r, m);
                        m.triangles().len()
                    },
                );
            });
        }

        let c = par::map_init(
            keys.len(),
            || (BrickMesh::default(), BrickMesh::default(), Vec::new()),
            |(mc, md, r), i| {
                mc.clear();
                md.clear();
                let values = batch.brick_values(i);
                mesh_brick(&engine, keys[i], values, RefineMode::Exact, mc);
                placed(i, r);
                mesh_brick_refined(&engine, keys[i], values, r, md);
                let mut c = Compare {
                    tris_bad: usize::from(mc.triangles() != md.triangles()),
                    evals_cpu: mc.evals,
                    ..Compare::default()
                };
                let (pc, nc, ac, kc) = mc.vertices();
                let (pd, nd, ad, kd) = md.vertices();
                let (first, perm) = &g.bricks[i];
                for k in 0..pc.len() {
                    let v =
                        |p: [f32; 3]| Vec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
                    let (a, b) = (v(pc[k]), v(pd[k]));
                    let d = a.dist(b);
                    c.vertices += 1;
                    c.sum_d += d;
                    c.max_d = c.max_d.max(d);
                    c.hist[[1e-5, 1e-4, 1e-3, 1e-2]
                        .iter()
                        .position(|&t| d <= t)
                        .unwrap_or(4)] += 1;
                    let fd = engine.field.value(b.to_array()).abs();
                    let fc = engine.field.value(a.to_array()).abs();
                    c.max_f_dev = c.max_f_dev.max(fd);
                    c.sum_f_dev += fd;
                    c.max_f_cpu = c.max_f_cpu.max(fc);
                    c.sum_f_cpu += fc;
                    c.over_tol_cpu += usize::from(fc > 1e-4);
                    c.over_tol_dev += usize::from(fd > 1e-4);
                    c.atom_diff += usize::from(ac[k] != ad[k]);
                    c.patch_diff += usize::from(kc[k] != kd[k]);
                    let dot = v(nc[k]).dot(v(nd[k]));
                    let angle = dot.clamp(-1.0, 1.0).acos().to_degrees();
                    c.max_angle = c.max_angle.max(angle);
                    c.normal_over_1deg += usize::from(angle > 1.0);
                    let e = e_dev[first + perm[k] as usize];
                    c.unconverged_dev += usize::from(e >> 31 == 0);
                    c.evals_dev += u64::from(e & 0x7fff_ffff);
                }
                c
            },
        );
        cmp = c.into_iter().fold(cmp, Compare::add);
    }

    let n = cmp.vertices.max(1) as f64;
    println!(
        "S2 {kind:?} h={h} atoms={} bricks={} vertices={} (groups not sampled: {not_sampled})",
        spheres.len(),
        bricks.len(),
        cmp.vertices
    );
    println!(
        "  |Δpos| mean {:.2e} max {:.2e} Å; ≤1e-5 {} ≤1e-4 {} ≤1e-3 {} ≤1e-2 {} more {}",
        cmp.sum_d / n,
        cmp.max_d,
        cmp.hist[0],
        cmp.hist[1],
        cmp.hist[2],
        cmp.hist[3],
        cmp.hist[4]
    );
    println!(
        "  |F| device mean {:.2e} max {:.2e} (> 1e-4: {}); cpu mean {:.2e} max {:.2e} (> 1e-4: {}); \
         atom diff {} patch diff {}; normal max {:.3}° (> 1°: {}); triangles differ in {} bricks",
        cmp.sum_f_dev / n,
        cmp.max_f_dev,
        cmp.over_tol_dev,
        cmp.sum_f_cpu / n,
        cmp.max_f_cpu,
        cmp.over_tol_cpu,
        cmp.atom_diff,
        cmp.patch_diff,
        cmp.max_angle,
        cmp.normal_over_1deg,
        cmp.tris_bad
    );
    println!(
        "  evals per edge: device {:.2} (unconverged {}, {:.3} %), cpu {:.2}; device histogram {:?}",
        cmp.evals_dev as f64 / n,
        cmp.unconverged_dev,
        100.0 * cmp.unconverged_dev as f64 / n,
        cmp.evals_cpu as f64 / n,
        evals_hist
    );
    if !quick {
        println!(
            "  times ({} threads): device sample {:.3} s, refine {:.3} s; host edge groups {:.3} s; \
             cpu mesh exact {:.3} s, placed from device {:.3} s",
            par::threads(),
            times.sample,
            times.refine,
            times.host,
            times.cpu_exact,
            times.cpu_placed
        );
        let now = times.sample.max(times.cpu_exact);
        let fast = (times.sample + times.refine).max(times.host + times.cpu_placed);
        println!(
            "  projected mesh stage (pipelined): T11.3′ {:.3} s, fast mode {:.3} s ({:.2}×)",
            now,
            fast,
            now / fast
        );
    }
    assert_eq!(not_sampled, 0);
    assert_eq!(cmp.tris_bad, 0);
}

#[test]
#[ignore = "S2 spike: needs a device and SESMESH_S2_INPUT"]
fn s2() {
    let Ok(input) = std::env::var("SESMESH_S2_INPUT") else {
        return;
    };
    let device = Runtime::load().and_then(|r| r.open(0)).unwrap();
    let kind = match std::env::var("SESMESH_S2_KIND").as_deref() {
        Ok("sas") => SurfaceKind::Sas,
        Ok("vdw") => SurfaceKind::Vdw,
        _ => SurfaceKind::Ses,
    };
    let hs = std::env::var("SESMESH_S2_H").unwrap_or_else(|_| "0.5".into());
    let Some(spheres) = load(&input) else {
        println!("S2 {input} skipped: does not load");
        return;
    };
    for h in hs.split(',') {
        run_one(&device, &spheres, kind, h.trim().parse().unwrap());
    }
}
