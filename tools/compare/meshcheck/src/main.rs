// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Helper of the cross-tool comparison (see Cargo.toml for the commands).

mod read;

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::ExitCode;

use rayon::prelude::*;
use sesmesh::{Diagnostics, Selection, SesField, Sphere, SurfaceKind, io, radii};

use read::{RawMesh, Res};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("spheres") => spheres(&args[1..]),
        Some("check") => check(&args[1..]),
        Some("probe") => probe_points(&args[1..]),
        _ => Err(
            "usage: meshcheck spheres <structure> <prefix> [--assembly ID] | \
                  meshcheck check <spheres.xyzr> <mesh> [--format F] [--probe R] [--sample N]"
                .into(),
        ),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("meshcheck: {e}");
            ExitCode::FAILURE
        }
    }
}

fn opt<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].as_str())
}

// ------------------------------------------------------------------ spheres

/// PDB atom name field (columns 13–16): one-letter elements start in 14.
fn pdb_name(name: &str, elem: &str) -> String {
    if name.len() < 4 && elem.len() == 1 {
        format!(" {name:<3}")
    } else {
        format!("{:<4}", &name[..name.len().min(4)])
    }
}

fn spheres(args: &[String]) -> Res<()> {
    let (input, prefix) = match args {
        [a, b, ..] => (a, b),
        _ => return Err("spheres <structure> <prefix> [--assembly ID]".into()),
    };
    let mut ro = io::ReadOptions::default();
    ro.assembly = opt(args, "--assembly").map(str::to_string);
    let s = io::read_path(input, &ro).map_err(|e| e.to_string())?;
    let mut diags = Diagnostics::new();
    let atoms = s
        .select(&Selection::default(), &mut diags)
        .map_err(|e| e.to_string())?;
    let r = radii::assign(&atoms, &radii::RadiusOptions::default(), &mut diags)
        .map_err(|e| e.to_string())?;
    // Every program gets the same spheres. Coordinates keep 3 decimals (as in
    // PDB/mmCIF files) in every file; all atoms are written as ATOM records,
    // since some programs skip HETATM (ligands) or treat them specially.
    let (mut xyzr, mut pqr, mut eses, mut pdb) =
        (String::new(), String::new(), String::new(), String::new());
    for (k, (a, &rad)) in atoms.iter().zip(&r).enumerate() {
        let [x, y, z] = a.position;
        let elem = a.element.map_or("X", |e| e.symbol());
        let name = pdb_name(s.str(a.name), elem);
        let res = s.str(a.res_name);
        let res = &res[..res.len().min(3)];
        let chain = s.str(a.chain).chars().next().unwrap_or('A');
        let serial = (k + 1) % 100_000;
        let seq = a.res_seq.rem_euclid(10_000);
        writeln!(xyzr, "{x:.3} {y:.3} {z:.3} {rad:.4}").unwrap();
        // Whitespace-delimited PQR (the format's definition; sesmesh, ChimeraX).
        writeln!(
            pqr,
            "ATOM {} {} {res} {chain} {} {x:.3} {y:.3} {z:.3} 0.0000 {rad:.4}",
            k + 1,
            s.str(a.name),
            a.res_seq
        )
        .unwrap();
        // ESES reads fixed columns (x 31-38, y 39-46, z 47-54, radius 63-69),
        // only ATOM records, and atom names starting with C, H, O, N, P or S
        // (the type only selects ESES's visualization colours).
        let t = elem
            .chars()
            .next()
            .filter(|c| "CHONPS".contains(*c))
            .unwrap_or('C');
        writeln!(eses, "ATOM  {serial:>5}  {t:<3} {res:>3} {chain}{seq:>4}    {x:>8.3}{y:>8.3}{z:>8.3}{:>8.4}{rad:>7.4}", 0.0).unwrap();
        writeln!(pdb, "ATOM  {serial:>5} {name} {res:>3} {chain}{seq:>4}    {x:>8.3}{y:>8.3}{z:>8.3}  1.00  0.00          {elem:>2}").unwrap();
    }
    pdb.push_str("END\n");
    for (ext, body) in [
        ("xyzr", &xyzr),
        ("pqr", &pqr),
        ("eses.pqr", &eses),
        ("pdb", &pdb),
    ] {
        fs::write(format!("{prefix}.{ext}"), body).map_err(|e| format!("{prefix}.{ext}: {e}"))?;
    }
    println!(
        "{{\"atoms\":{},\"warnings\":{}}}",
        atoms.len(),
        diags.warnings().len()
    );
    Ok(())
}

fn read_xyzr(path: &str) -> Res<Vec<Sphere>> {
    let src = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for l in src.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = l
            .split_whitespace()
            .take(4)
            .map(|t| t.parse().map_err(|_| format!("bad xyzr line {l:?}")))
            .collect::<Res<_>>()?;
        out.push(Sphere::new([v[0], v[1], v[2]], v[3]));
    }
    Ok(out)
}

/// `probe <spheres.xyzr> x,y,z ...`: exact field, closest point, patch.
fn probe_points(args: &[String]) -> Res<()> {
    let spheres = read_xyzr(args.first().ok_or("probe <spheres.xyzr> x,y,z ...")?)?;
    let field = SesField::new(&spheres, 1.4, SurfaceKind::Ses).map_err(|e| e.to_string())?;
    for a in &args[1..] {
        let v: Vec<f64> = a
            .split(',')
            .map(|t| t.parse().map_err(|_| format!("bad point {a}")))
            .collect::<Res<_>>()?;
        let s = field.sample([v[0], v[1], v[2]]);
        println!(
            "{a}: F={:.6e} closest={:?} patch={:?}",
            s.value, s.closest, s.patch
        );
    }
    Ok(())
}

// -------------------------------------------------------------------- check

#[derive(Default)]
struct Topo {
    used_vertices: usize,
    edges: usize,
    boundary: usize,
    nonmanifold: usize,
    misoriented: usize,
    components: usize,
    euler: i64,
}

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

fn topology(nv: usize, tris: &[[u32; 3]]) -> Topo {
    let mut t = Topo::default();
    // Directed edges packed as (min, max, forward).
    let mut keys: Vec<u64> = Vec::with_capacity(tris.len() * 3);
    let mut used = vec![false; nv];
    for tri in tris {
        if tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            used[a as usize] = true;
            let (lo, hi, fwd) = if a < b { (a, b, 1) } else { (b, a, 0) };
            keys.push((u64::from(lo) << 33) | (u64::from(hi) << 1) | fwd);
        }
    }
    keys.par_sort_unstable();
    let mut i = 0;
    while i < keys.len() {
        let e = keys[i] >> 1;
        let mut j = i;
        let mut fwd = 0;
        while j < keys.len() && keys[j] >> 1 == e {
            fwd += keys[j] & 1;
            j += 1;
        }
        let n = j - i;
        t.edges += 1;
        match n {
            1 => t.boundary += 1,
            2 if fwd == 1 => {}
            2 => t.misoriented += 1,
            _ => t.nonmanifold += 1,
        }
        i = j;
    }
    let mut parent: Vec<u32> = (0..nv as u32).collect();
    for tri in tris {
        for k in 1..3 {
            let (a, b) = (find(&mut parent, tri[0]), find(&mut parent, tri[k]));
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }
    }
    t.used_vertices = used.iter().filter(|&&u| u).count();
    t.components = (0..nv as u32)
        .filter(|&v| used[v as usize] && find(&mut parent, v) == v)
        .count();
    let f = tris
        .iter()
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .count();
    t.euler = t.used_vertices as i64 - t.edges as i64 + f as i64;
    t
}

fn topo_json(t: &Topo) -> String {
    format!(
        "\"used_vertices\":{},\"edges\":{},\"boundary_edges\":{},\"nonmanifold_edges\":{},\"misoriented_edges\":{},\"components\":{},\"euler\":{},\"closed\":{},\"oriented\":{}",
        t.used_vertices,
        t.edges,
        t.boundary,
        t.nonmanifold,
        t.misoriented,
        t.components,
        t.euler,
        t.boundary == 0 && t.nonmanifold == 0,
        t.misoriented == 0
    )
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    *v.select_nth_unstable_by(k, f64::total_cmp).1
}

/// Per sampled vertex: field value, (normal angle in degrees, inward), and
/// whether the analytic normal is defined there.
type VertexSample = (f64, Option<(f64, bool)>, bool);

fn check(args: &[String]) -> Res<()> {
    let (xyzr, mesh_path) = match args {
        [a, b, ..] => (a, b),
        _ => {
            return Err("check <spheres.xyzr> <mesh> [--format F] [--probe R] [--sample N]".into());
        }
    };
    let path = Path::new(mesh_path);
    let format = match opt(args, "--format") {
        Some(f) => f,
        None => read::detect(path)?,
    };
    let probe: f64 = opt(args, "--probe")
        .map_or(Ok(1.4), str::parse)
        .map_err(|_| "bad --probe")?;
    let max_sample: usize = opt(args, "--sample")
        .map_or(Ok(2_000_000), str::parse)
        .map_err(|_| "bad --sample")?;
    let t0 = std::time::Instant::now();
    let m: RawMesh = read::read(path, format)?;
    let read_s = t0.elapsed().as_secs_f64();
    let nv = m.positions.len();

    // Degenerate and duplicate triangles, area and signed volume.
    let mut c = [0.0; 3];
    for p in &m.positions {
        for k in 0..3 {
            c[k] += p[k] / nv.max(1) as f64;
        }
    }
    let (mut area, mut vol, mut degen_idx, mut degen_area) = (0.0, 0.0, 0usize, 0usize);
    for t in &m.triangles {
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            degen_idx += 1;
            continue;
        }
        let [a, b, d] = t.map(|i| sub(m.positions[i as usize], c));
        let n = cross(sub(b, a), sub(d, a));
        let a2 = dot(n, n).sqrt() / 2.0;
        if a2 < 1e-12 {
            degen_area += 1;
        }
        area += a2;
        vol += dot(a, cross(b, d)) / 6.0;
    }
    let mut sorted: Vec<[u32; 3]> = m
        .triangles
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort_unstable();
            s
        })
        .collect();
    sorted.par_sort_unstable();
    let dup_tris = sorted.windows(2).filter(|w| w[0] == w[1]).count();
    drop(sorted);

    // Distinct vertices at identical positions (cracks unless welded).
    let mut keyed: Vec<([u64; 3], u32)> = m
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| (p.map(f64::to_bits), i as u32))
        .collect();
    keyed.par_sort_unstable();
    let mut weld: Vec<u32> = (0..nv as u32).collect();
    let mut coincident = 0usize;
    for w in keyed.windows(2) {
        if w[0].0 == w[1].0 {
            coincident += 1;
            weld[w[1].1 as usize] = weld[w[0].1 as usize];
        }
    }
    drop(keyed);

    let topo = topology(nv, &m.triangles);
    let mut out = format!(
        "{{\"format\":\"{format}\",\"vertices\":{nv},\"triangles\":{},\"polygons\":{},\"degenerate_index\":{degen_idx},\"degenerate_area\":{degen_area},\"duplicate_triangles\":{dup_tris},\"coincident_vertices\":{coincident},{},\"area\":{area:.6},\"volume\":{vol:.6},\"read_s\":{read_s:.3}",
        m.triangles.len(),
        m.polygons,
        topo_json(&topo)
    );
    if coincident > 0 {
        let wt: Vec<[u32; 3]> = m
            .triangles
            .iter()
            .map(|t| t.map(|i| weld[i as usize]))
            .collect();
        let tw = topology(nv, &wt);
        write!(out, ",\"welded\":{{{}}}", topo_json(&tw)).unwrap();
    }

    // Distance of the vertices to the exact SES and normal errors.
    let spheres = read_xyzr(xyzr)?;
    let t1 = std::time::Instant::now();
    let field = SesField::new(&spheres, probe, SurfaceKind::Ses).map_err(|e| e.to_string())?;
    let stride = nv.div_ceil(max_sample.max(1)).max(1);
    let idx: Vec<usize> = (0..nv).step_by(stride).collect();
    let res: Vec<VertexSample> = idx
        .par_iter()
        .map(|&i| {
            let s = field.sample(m.positions[i]);
            let nrm = match (&m.normals, s.normal) {
                (Some(ns), Some(a)) => {
                    let b = ns[i];
                    let lb = dot(b, b).sqrt();
                    (lb > 0.0).then(|| {
                        let cosv = (dot(a, b) / lb).clamp(-1.0, 1.0);
                        (cosv.acos().to_degrees(), cosv < 0.0)
                    })
                }
                _ => None,
            };
            (s.value, nrm, s.normal.is_some())
        })
        .collect();
    let mut absf: Vec<f64> = res.iter().map(|r| r.0.abs()).collect();
    let n = absf.len().max(1) as f64;
    let mean = absf.iter().sum::<f64>() / n;
    let rms = (absf.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
    let signed = res.iter().map(|r| r.0).sum::<f64>() / n;
    let gt = |t: f64| absf.iter().filter(|&&x| x > t).count() as f64 / n;
    let (g1, g2) = (gt(0.01), gt(0.1));
    let (imax, max) = absf
        .iter()
        .copied()
        .enumerate()
        .fold((0, 0.0), |acc, (i, x)| if x > acc.1 { (i, x) } else { acc });
    let wp = idx.get(imax).map_or([f64::NAN; 3], |&i| m.positions[i]);
    let (p50, p99) = (pct(&mut absf, 0.5), pct(&mut absf, 0.99));
    write!(
        out,
        ",\"dist\":{{\"sampled\":{},\"stride\":{stride},\"max\":{max:.6e},\"mean\":{mean:.6e},\"rms\":{rms:.6e},\"p50\":{p50:.6e},\"p99\":{p99:.6e},\"gt_0.01\":{g1:.6},\"gt_0.1\":{g2:.6},\"mean_signed\":{signed:.6e},\"worst\":[{:.4},{:.4},{:.4}]}}",
        res.len(),
        wp[0],
        wp[1],
        wp[2]
    )
    .unwrap();
    if m.normals.is_some() {
        let mut ang: Vec<f64> = res.iter().filter_map(|r| r.1.map(|a| a.0)).collect();
        let inward = res.iter().filter(|r| r.1.is_some_and(|a| a.1)).count();
        let cov = ang.len() as f64 / n;
        let na = ang.len().max(1) as f64;
        let am = ang.iter().sum::<f64>() / na;
        let amax = ang.iter().copied().fold(0.0, f64::max);
        let a99 = pct(&mut ang, 0.99);
        write!(
            out,
            ",\"normals\":{{\"compared\":{},\"coverage\":{cov:.6},\"mean_deg\":{am:.6e},\"p99_deg\":{a99:.6e},\"max_deg\":{amax:.6e},\"inward\":{inward}}}",
            ang.len()
        )
        .unwrap();
    }
    if args.iter().any(|a| a == "--components") {
        // Per component (welded): triangles, area, signed volume, centroid,
        // largest |F| over its vertices.
        let wt: Vec<[u32; 3]> = m
            .triangles
            .iter()
            .map(|t| t.map(|i| weld[i as usize]))
            .collect();
        let mut parent: Vec<u32> = (0..nv as u32).collect();
        for t in &wt {
            for k in 1..3 {
                let (a, b) = (find(&mut parent, t[0]), find(&mut parent, t[k]));
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
            }
        }
        let mut comps: std::collections::BTreeMap<u32, (usize, f64, f64, [f64; 3], f64)> =
            Default::default();
        for t in &wt {
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                continue;
            }
            let r = find(&mut parent, t[0]);
            let [a, b, d] = t.map(|i| sub(m.positions[i as usize], c));
            let n = cross(sub(b, a), sub(d, a));
            let e = comps.entry(r).or_insert((0, 0.0, 0.0, [0.0; 3], 0.0));
            e.0 += 1;
            e.1 += dot(n, n).sqrt() / 2.0;
            e.2 += dot(a, cross(b, d)) / 6.0;
            for k in 0..3 {
                e.3[k] += (a[k] + b[k] + d[k]) / 3.0 + c[k];
            }
        }
        for (k, &i) in idx.iter().enumerate() {
            let r = find(&mut parent, weld[i]);
            if let Some(e) = comps.get_mut(&r) {
                e.4 = e.4.max(res[k].0.abs());
            }
        }
        let list: Vec<String> = comps
            .values()
            .map(|e| {
                format!(
                    "[{},{:.4},{:.4},[{:.3},{:.3},{:.3}],{:.3e}]",
                    e.0,
                    e.1,
                    e.2,
                    e.3[0] / e.0 as f64,
                    e.3[1] / e.0 as f64,
                    e.3[2] / e.0 as f64,
                    e.4
                )
            })
            .collect();
        write!(out, ",\"component_list\":[{}]", list.join(",")).unwrap();
    }
    write!(out, ",\"field_s\":{:.3}}}", t1.elapsed().as_secs_f64()).unwrap();
    std::io::stdout()
        .write_all(out.as_bytes())
        .and_then(|()| std::io::stdout().write_all(b"\n"))
        .map_err(|e| e.to_string())
}
