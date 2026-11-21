// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! S1 spike (`docs/gpu-prior-art.md` §5.2): device classification, gathering
//! and sampling from the resident arrangement, compared with the CPU's
//! `Engine::evaluate` brick by brick, and timed against it.
//!
//! ```text
//! SESMESH_S1_INPUT=tests/data/cache/1aon.cif.gz SESMESH_S1_H=0.5,0.25 \
//!   cargo test --release -p sesmesh --features hip --lib s1 -- --ignored --nocapture
//! ```
//!
//! `SESMESH_S1_KIND` = `ses` (default), `sas` or `vdw`; `SESMESH_S1_QUICK=1`
//! skips the timings. It also rebuilds the refinement lists of `extract.rs`
//! from the device's superset lists and times that against `gather_local`.

#![allow(clippy::print_stdout, reason = "the spike's report; test only")]

use std::time::Instant;

use sesmesh_gpu::hip::{
    Batch, Device, REC_WORDS, Resident, Runtime, STATE_IN, STATE_OUT, STATE_SAMPLE,
};

use super::{Engine, SubStates};
use crate::field::LocalCandidates;
use crate::field::point::Superset;
use crate::field::{SesField, SimdLevel, SurfaceKind};
use crate::geom::{Sphere, Vec3};
use crate::par;

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

/// First non-uniform sub-block containing brick sample `(x, y, z)`, and the
/// sign of the last uniform one visited (`Engine::assign`).
fn owner(state: &SubStates, sub: usize, v: [usize; 3]) -> (Option<usize>, bool) {
    let range = |v: usize| {
        let hi = (v / sub).min(1);
        let lo = if v > 0 && v.is_multiple_of(sub) {
            (v / sub - 1).min(1)
        } else {
            hi
        };
        (lo, hi)
    };
    let (x0, x1) = range(v[0]);
    let (y0, y1) = range(v[1]);
    let (z0, z1) = range(v[2]);
    let mut positive = true;
    for sz in z0..=z1 {
        for sy in y0..=y1 {
            for sx in x0..=x1 {
                let sb = sx + 2 * sy + 4 * sz;
                match state[sb] {
                    None => return (Some(sb), positive),
                    Some(s) => positive = s,
                }
            }
        }
    }
    (None, positive)
}

#[derive(Debug, Default, Clone, Copy)]
struct Compare {
    sub_cpu_sampled: usize,
    sub_dev_sampled: usize,
    /// Device uniform where the CPU is not, or with the other sign: must be 0.
    dev_uniform_bad: usize,
    samples_both: usize,
    /// Both evaluated, bits differ, |F| away from the clamp: expected 0.
    diff_core: usize,
    /// Both evaluated, bits differ, at the clamp (±band or probe).
    diff_clamp: usize,
    /// Sign differs anywhere: must be 0.
    sign_bad: usize,
    crossing_edges: usize,
    /// A crossing-edge end differs: must be 0.
    cross_bad: usize,
    max_core_diff: f32,
}

impl Compare {
    fn add(mut self, o: Self) -> Self {
        self.sub_cpu_sampled += o.sub_cpu_sampled;
        self.sub_dev_sampled += o.sub_dev_sampled;
        self.dev_uniform_bad += o.dev_uniform_bad;
        self.samples_both += o.samples_both;
        self.diff_core += o.diff_core;
        self.diff_clamp += o.diff_clamp;
        self.sign_bad += o.sign_bad;
        self.crossing_edges += o.crossing_edges;
        self.cross_bad += o.cross_bad;
        self.max_core_diff = self.max_core_diff.max(o.max_core_diff);
        self
    }
}

fn run_one(device: &Device, spheres: &[Sphere], kind: SurfaceKind, h: f64) {
    let t = Instant::now();
    let probe = if kind == SurfaceKind::Vdw { 0.0 } else { 1.4 };
    let field = SesField::new(spheres, probe, kind).unwrap();
    let t_field = t.elapsed().as_secs_f64();
    let engine = Engine::new(field, h, 10, SimdLevel::Auto).unwrap();
    if !engine.device_ok() {
        println!("S1 {kind:?} h={h} skipped: f64 path (coordinates beyond the f32 range)");
        return;
    }
    let quick = std::env::var("SESMESH_S1_QUICK").as_deref() == Ok("1");
    let t = Instant::now();
    let (_, bricks) = engine.surface_bricks().unwrap();
    let t_bricks = t.elapsed().as_secs_f64();
    let origins = engine.frame_origins(&bricks);
    let t = Instant::now();
    let res = engine.resident(|a| Resident::new(device, a)).unwrap();
    let t_upload = t.elapsed().as_secs_f64();
    let mut batch = Batch::new(device).unwrap();
    let mut p = engine.batch_params();
    p.near = true;

    // Device: one warm-up run, then the best of three.
    res.run(&p, &origins, &mut batch).unwrap();
    let reps = if quick { 0 } else { 3 };
    let mut best = f64::INFINITY;
    let mut dev = Default::default();
    for _ in 0..reps {
        let t = Instant::now();
        let times = res.run(&p, &origins, &mut batch).unwrap();
        let wall = t.elapsed().as_secs_f64();
        if wall < best {
            best = wall;
            dev = times;
        }
    }

    // CPU: the stage the device replaces (classification, gathering,
    // kernel), all threads, best of three.
    let cpu_pass = || {
        par::map_init(
            bricks.len(),
            || engine.scratch(),
            |s, i| {
                engine.evaluate(bricks[i], s);
                s.values[0].to_bits()
            },
        )
        .len()
    };
    if !quick {
        cpu_pass();
    }
    let mut cpu = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        cpu_pass();
        cpu = cpu.min(t.elapsed().as_secs_f64());
    }

    // Both at once: the device batch on its own thread, the CPU pass on
    // the pool (the refining cores of a real run).
    let (dev_busy, cpu_busy) = if quick {
        (0.0, 0.0)
    } else {
        std::thread::scope(|sc| {
            let handle = sc.spawn(|| {
                let t = Instant::now();
                res.run(&p, &origins, &mut batch).unwrap();
                t.elapsed().as_secs_f64()
            });
            let t = Instant::now();
            cpu_pass();
            let c = t.elapsed().as_secs_f64();
            (handle.join().unwrap(), c)
        })
    };

    // Compare with the CPU, brick by brick.
    let values = batch.values();
    let rec = batch.records();
    let side = engine.lattice.b as usize + 1;
    let spb = side * side * side;
    let sub = engine.lattice.b as usize / 2;
    let band = engine.band as f32;
    let probe = engine.field.probe as f32;
    let cmp = par::map_init(
        bricks.len(),
        || engine.scratch(),
        |s, bi| {
            let key = bricks[bi];
            let cpu_state = engine.sub_states(key);
            engine.evaluate(key, s);
            let cv = &s.values[..spb];
            let dv = &values[bi * spb..(bi + 1) * spb];
            let mut dev_state: SubStates = [None; 8];
            let mut c = Compare::default();
            let mut mask = 0u8;
            for sb in 0..8 {
                let st = rec[REC_WORDS * (8 * bi + sb)];
                dev_state[sb] = match st {
                    STATE_SAMPLE => None,
                    STATE_OUT => Some(true),
                    STATE_IN => Some(false),
                    other => panic!("state {other}"),
                };
                c.sub_cpu_sampled += usize::from(cpu_state[sb].is_none());
                c.sub_dev_sampled += usize::from(dev_state[sb].is_none());
                if dev_state[sb].is_some() && dev_state[sb] != cpu_state[sb] {
                    c.dev_uniform_bad += 1;
                }
            }
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        let i = x + side * (y + side * z);
                        let (co, _) = owner(&cpu_state, sub, [x, y, z]);
                        let (dvo, _) = owner(&dev_state, sub, [x, y, z]);
                        if (cv[i] < 0.0) != (dv[i] < 0.0) {
                            c.sign_bad += 1;
                        }
                        if co.is_some() && dvo.is_some() {
                            c.samples_both += 1;
                            if cv[i].to_bits() != dv[i].to_bits() {
                                let clamp = cv[i].abs() >= band - 1e-3 || cv[i] == probe;
                                if clamp {
                                    c.diff_clamp += 1;
                                } else {
                                    c.diff_core += 1;
                                    c.max_core_diff = c.max_core_diff.max((cv[i] - dv[i]).abs());
                                }
                            }
                        }
                        for (dxi, ok) in [
                            (1, x + 1 < side),
                            (side, y + 1 < side),
                            (side * side, z + 1 < side),
                        ] {
                            if ok && (cv[i] < 0.0) != (cv[i + dxi] < 0.0) {
                                c.crossing_edges += 1;
                                // extract.rs: the sub-block of the lower
                                // sample gathers the refinement lists.
                                if x < side - 1 && y < side - 1 && z < side - 1 {
                                    let s = [x, y, z].map(|v| (v / sub).min(1));
                                    mask |= 1 << (s[0] + 2 * s[1] + 4 * s[2]);
                                }
                                if cv[i].to_bits() != dv[i].to_bits()
                                    || cv[i + dxi].to_bits() != dv[i + dxi].to_bits()
                                {
                                    c.cross_bad += 1;
                                }
                            }
                        }
                    }
                }
            }
            (c, mask)
        },
    );
    let masks: Vec<u8> = cmp.iter().map(|c| c.1).collect();
    let cmp = cmp
        .into_iter()
        .map(|c| c.0)
        .fold(Compare::default(), Compare::add);

    // Refinement lists (docs/gpu-prior-art.md §3.3): CPU `gather_local`
    // (cell scans + exact tests + sorts) against the same exact tests and
    // sorts over the device's superset lists.
    let lat = &engine.lattice;
    let h = lat.h;
    let rho_ref = 0.5 * 3f64.sqrt() * sub as f64 * h * (1.0 + 1e-9) + 1e-9;
    let centre = |bi: usize, sb: usize| {
        let base = lat.point(lat.brick_origin(bricks[bi]));
        let o = [sb & 1, (sb >> 1) & 1, sb >> 2].map(|v| (v * sub) as f64 + 0.5 * sub as f64);
        base + Vec3::new(o[0], o[1], o[2]) * h
    };
    let id_lists = batch.id_lists();
    let superset = |bi: usize, sb: usize| {
        let r = &rec[REC_WORDS * (8 * bi + sb)..][..REC_WORDS];
        if r[0] != STATE_SAMPLE {
            return None;
        }
        let mut at = r[1] as usize;
        let mut take = |n: u32| {
            let l = &id_lists[at..at + n as usize];
            at += n as usize;
            l
        };
        Some(Superset {
            inside: take(r[2]),
            spheres: take(r[3]),
            circles: take(r[4]),
            vertices: take(r[5]),
            near: take(r[6]),
        })
    };
    let n_ref: usize = masks.iter().map(|m| m.count_ones() as usize).sum();
    let n_sup: usize = (0..bricks.len())
        .flat_map(|bi| (0..8).map(move |sb| (bi, sb)))
        .filter(|&(bi, sb)| masks[bi] >> sb & 1 == 1)
        .map(|(bi, sb)| {
            let r = &rec[REC_WORDS * (8 * bi + sb)..][..REC_WORDS];
            r[2..7].iter().map(|&n| n as usize).sum::<usize>()
        })
        .sum();
    let ses = kind == SurfaceKind::Ses;
    let (missing, mismatch) = par::map_init(
        bricks.len(),
        || (LocalCandidates::default(), LocalCandidates::default()),
        |(lc, ld), bi| {
            let mut m = (0usize, 0usize);
            for sb in (0..8).filter(|sb| masks[bi] >> sb & 1 == 1) {
                let Some(sup) = superset(bi, sb) else {
                    m.0 += 1;
                    continue;
                };
                engine
                    .field
                    .gather_local(centre(bi, sb), rho_ref, engine.band, lc);
                engine
                    .field
                    .gather_local_from(centre(bi, sb), rho_ref, engine.band, &sup, ld);
                m.1 += usize::from(!lc.same_as(ld, ses));
            }
            m
        },
    )
    .into_iter()
    .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    let gather_cpu = || {
        par::map_init(bricks.len(), LocalCandidates::default, |l, bi| {
            for sb in (0..8).filter(|sb| masks[bi] >> sb & 1 == 1) {
                engine
                    .field
                    .gather_local(centre(bi, sb), rho_ref, engine.band, l);
            }
            l.len_marker()
        })
        .len()
    };
    let gather_dev = || {
        par::map_init(bricks.len(), LocalCandidates::default, |l, bi| {
            for sb in (0..8).filter(|sb| masks[bi] >> sb & 1 == 1) {
                if let Some(sup) = superset(bi, sb) {
                    engine
                        .field
                        .gather_local_from(centre(bi, sb), rho_ref, engine.band, &sup, l);
                }
            }
            l.len_marker()
        })
        .len()
    };
    let (mut t_gc, mut t_gd) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        gather_cpu();
        t_gc = t_gc.min(t.elapsed().as_secs_f64());
        let t = Instant::now();
        gather_dev();
        t_gd = t_gd.min(t.elapsed().as_secs_f64());
    }

    let ids = batch.ids();
    println!(
        "S1 {kind:?} h={h} atoms={} bricks={} sub-blocks sampled cpu={} dev={} (+{:.1} %)",
        spheres.len(),
        bricks.len(),
        cmp.sub_cpu_sampled,
        cmp.sub_dev_sampled,
        100.0 * (cmp.sub_dev_sampled as f64 / cmp.sub_cpu_sampled.max(1) as f64 - 1.0)
    );
    println!(
        "  check: dev_uniform_bad={} sign_bad={} cross_bad={} of {} crossing edges; \
         both-evaluated samples {}: diff at clamp {}, diff elsewhere {} (max {:e})",
        cmp.dev_uniform_bad,
        cmp.sign_bad,
        cmp.cross_bad,
        cmp.crossing_edges,
        cmp.samples_both,
        cmp.diff_clamp,
        cmp.diff_core,
        cmp.max_core_diff
    );
    println!(
        "  device: total {:.3} s (count {:.3}, scan {:.3}, fill {:.3}, sample {:.3}); \
         with CPU busy {:.3} s",
        best, dev.count, dev.scan, dev.fill, dev.sample, dev_busy
    );
    println!(
        "  cpu evaluate ({} threads): {:.3} s; with device busy {:.3} s; device/cpu {:.2}",
        par::threads(),
        cpu,
        cpu_busy,
        best / cpu
    );
    println!(
        "  memory: resident {:.1} MB, ids {:.1} MB ({:.0} per sampled sub-block), values {:.1} MB; \
         field {:.3} s, bricks {:.3} s, upload {:.3} s",
        res.resident_bytes() as f64 / 1e6,
        4.0 * ids as f64 / 1e6,
        ids as f64 / cmp.sub_dev_sampled.max(1) as f64,
        4.0 * values.len() as f64 / 1e6,
        t_field,
        t_bricks,
        t_upload
    );
    println!(
        "  refinement lists: {n_ref} sub-blocks ({:.1} % of sampled), device superset {:.0} ids each; \
         missing {missing}, differ {mismatch}; gather_local {:.3} s, from device lists {:.3} s ({:.2})",
        100.0 * n_ref as f64 / cmp.sub_cpu_sampled.max(1) as f64,
        n_sup as f64 / n_ref.max(1) as f64,
        t_gc,
        t_gd,
        t_gd / t_gc
    );
    assert_eq!(cmp.dev_uniform_bad, 0);
    assert_eq!(cmp.sign_bad, 0);
    assert_eq!(cmp.cross_bad, 0);
    assert_eq!((missing, mismatch), (0, 0));
}

#[test]
#[ignore = "S1 spike: needs a device and SESMESH_S1_INPUT"]
fn s1() {
    let Ok(input) = std::env::var("SESMESH_S1_INPUT") else {
        return;
    };
    let device = Runtime::load().and_then(|r| r.open(0)).unwrap();
    let kind = match std::env::var("SESMESH_S1_KIND").as_deref() {
        Ok("sas") => SurfaceKind::Sas,
        Ok("vdw") => SurfaceKind::Vdw,
        _ => SurfaceKind::Ses,
    };
    let hs = std::env::var("SESMESH_S1_H").unwrap_or_else(|_| "0.5".into());
    let Some(spheres) = load(&input) else {
        println!("S1 {input} skipped: does not load");
        return;
    };
    for h in hs.split(',') {
        run_one(&device, &spheres, kind, h.trim().parse().unwrap());
    }
}
