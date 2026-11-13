// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 9 campaigns: sweep sampling, accuracy study, robustness sweep.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use sesmesh::io::{ReadOptions, read_path};
use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection, Sphere, SurfaceParams, compute_surface};

/// SplitMix64: a small, well-mixed seeded generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Samples `count` current PDB entries (RCSB holdings) with `seed` and
/// writes them as `[sets.sweep]` into tests/data/sets.toml.
pub(crate) fn sample_sweep(data_dir: &Path, count: usize, seed: u64) -> Result<()> {
    let tmp = data_dir.join("cache").join("holdings.json.part");
    fs::create_dir_all(data_dir.join("cache"))?;
    crate::download(
        "https://data.rcsb.org/rest/v1/holdings/current/entry_ids",
        &tmp,
    )?;
    let text = fs::read_to_string(&tmp).context("reading the holdings list")?;
    fs::remove_file(&tmp).ok();
    let mut ids: Vec<String> = serde_json::from_str(&text).context("parsing the holdings list")?;
    ids.retain(|id| id.len() == 4 && id.bytes().all(|b| b.is_ascii_alphanumeric()));
    ids.sort_unstable();
    if ids.len() < count {
        bail!("only {} entries available", ids.len());
    }
    // Partial Fisher–Yates with the seeded generator.
    let mut rng = Rng(seed);
    for i in 0..count {
        let j = i + (rng.next() % (ids.len() - i) as u64) as usize;
        ids.swap(i, j);
    }
    let mut chosen: Vec<String> = ids[..count]
        .iter()
        .map(|id| format!("{id}.cif.gz"))
        .collect();
    chosen.sort_unstable();
    let mut section = format!(
        "[sets.sweep]\ndescription = \"Robustness sweep (Plan T9.4): {count} random current PDB entries \
         (seed {seed}, sampled from {} entries), mmCIF\"\nfiles = [\n",
        ids.len()
    );
    for row in chosen.chunks(5) {
        let items: Vec<String> = row.iter().map(|f| format!("\"{f}\"")).collect();
        let _ = writeln!(section, "    {},", items.join(", "));
    }
    section.push_str("]\n");
    let path = data_dir.join("sets.toml");
    let mut toml = fs::read_to_string(&path)?;
    if let Some(start) = toml.find("[sets.sweep]") {
        let end = toml[start + 1..]
            .find("\n[sets.")
            .map_or(toml.len(), |e| start + 1 + e + 1);
        toml.replace_range(start..end, "");
    }
    if !toml.ends_with("\n\n") {
        toml.push('\n');
    }
    toml.push_str(&section);
    fs::write(&path, toml)?;
    println!(
        "wrote [sets.sweep] with {count} entries (seed {seed}); now run `cargo xtask fetch --set sweep`"
    );
    Ok(())
}

/// Spheres of the default selection with Bondi radii.
fn load(path: &Path) -> Result<(Vec<Sphere>, usize)> {
    let structure = read_path(path, &ReadOptions::default())?;
    let mut diags = Diagnostics::new();
    let atoms = structure.select(&Selection::default(), &mut diags)?;
    let radii = assign(&atoms, &RadiusOptions::default(), &mut diags)?;
    let n = atoms.len();
    Ok((Sphere::from_atoms(&atoms, &radii)?, n))
}

/// Least-squares fit of `y = a + b·h²` (returns `a`).
fn extrapolate(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy) = points
        .iter()
        .fold((0.0, 0.0), |(sx, sy), &(h, y)| (sx + h * h, sy + y));
    let (sxx, sxy) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(h, y)| (a + h.powi(4), b + h * h * y));
    let b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    (sy - b * sx) / n
}

/// One surface of the accuracy study.
struct Row {
    h: f64,
    vertices: usize,
    triangles: usize,
    components: usize,
    cavities: usize,
    area: f64,
    volume: f64,
    seconds: f64,
}

/// Relative area and volume errors of one entry at 0.5 and 0.25 Å.
struct EntryErrors {
    coarse: Option<(f64, f64)>,
    fine: Option<(f64, f64)>,
}

fn percent(e: Option<(f64, f64)>) -> String {
    e.map_or_else(
        || "-".to_string(),
        |(a, v)| format!("A {:+.3}% V {:+.3}%", 100.0 * a, 100.0 * v),
    )
}

pub(crate) fn validate(
    root: &Path,
    files: &[String],
    spacings: &[f64],
    max_atoms: usize,
) -> Result<()> {
    let cache = root.join("tests/data/cache");
    let out_dir = root.join("docs/validation");
    fs::create_dir_all(&out_dir)?;
    let mut csv = String::from(
        "entry,atoms,spacing,vertices,triangles,components,cavities,area,volume,area_err,volume_err,vertices_h2_per_area,seconds\n",
    );
    let mut summary: Vec<EntryErrors> = Vec::new();
    for file in files {
        let path = cache.join(file);
        let entry = file.trim_end_matches(".cif.gz");
        let (spheres, atoms) = match load(&path) {
            Ok(v) => v,
            Err(e) => {
                println!("{entry}: skipped ({e})");
                continue;
            }
        };
        let mut rows = Vec::new();
        for &h in spacings {
            if atoms > max_atoms && h < 0.5 {
                continue;
            }
            let t = Instant::now();
            let out = compute_surface(&spheres, &SurfaceParams::ses(1.4).with_spacing(h))?;
            let s = out.stats;
            rows.push(Row {
                h,
                vertices: s.vertices,
                triangles: s.triangles,
                components: s.components,
                cavities: s.cavities,
                area: s.area,
                volume: s.volume,
                seconds: t.elapsed().as_secs_f64(),
            });
        }
        // Extrapolate from the three finest spacings.
        let fine: Vec<&Row> = rows.iter().rev().take(3).collect();
        let a0 = extrapolate(&fine.iter().map(|r| (r.h, r.area)).collect::<Vec<_>>());
        let v0 = extrapolate(&fine.iter().map(|r| (r.h, r.volume)).collect::<Vec<_>>());
        for r in &rows {
            let _ = writeln!(
                csv,
                "{entry},{atoms},{},{},{},{},{},{:.3},{:.3},{:.5},{:.5},{:.4},{:.3}",
                r.h,
                r.vertices,
                r.triangles,
                r.components,
                r.cavities,
                r.area,
                r.volume,
                r.area / a0 - 1.0,
                r.volume / v0 - 1.0,
                r.vertices as f64 * r.h * r.h / r.area,
                r.seconds
            );
        }
        let err = |hh: f64| {
            rows.iter()
                .find(|r| (r.h - hh).abs() < 1e-9)
                .map(|r| (r.area / a0 - 1.0, r.volume / v0 - 1.0))
        };
        let e = EntryErrors {
            coarse: err(0.5),
            fine: err(0.25),
        };
        println!(
            "{entry:6} {atoms:7} atoms  A0 {a0:10.1}  V0 {v0:11.1}  err@0.5 {}  err@0.25 {}",
            percent(e.coarse),
            percent(e.fine)
        );
        summary.push(e);
    }
    fs::write(out_dir.join("accuracy.csv"), csv)?;
    let report = |label: &str, values: Vec<f64>| {
        let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
        let worst = values
            .iter()
            .fold(0.0f64, |m, x| if x.abs() > m.abs() { *x } else { m });
        println!(
            "{label:12} n={:3}  mean {:+.3}%  worst {:+.3}%",
            values.len(),
            100.0 * mean,
            100.0 * worst
        );
    };
    report(
        "area@0.5",
        summary
            .iter()
            .filter_map(|e| e.coarse.map(|v| v.0))
            .collect(),
    );
    report(
        "volume@0.5",
        summary
            .iter()
            .filter_map(|e| e.coarse.map(|v| v.1))
            .collect(),
    );
    report(
        "area@0.25",
        summary.iter().filter_map(|e| e.fine.map(|v| v.0)).collect(),
    );
    report(
        "volume@0.25",
        summary.iter().filter_map(|e| e.fine.map(|v| v.1)).collect(),
    );
    println!("wrote docs/validation/accuracy.csv");
    Ok(())
}

pub(crate) fn sweep(
    root: &Path,
    files: &[String],
    spacing: f64,
    coarse_above: usize,
) -> Result<()> {
    let cache = root.join("tests/data/cache");
    let out_dir = root.join("docs/validation");
    fs::create_dir_all(&out_dir)?;
    let mut csv = String::from(
        "entry,status,atoms,spacing,vertices,triangles,components,cavities,euler,seconds,detail\n",
    );
    let (mut ok, mut input_errors, mut failures) = (0, 0, 0);
    for file in files {
        let entry = file.trim_end_matches(".cif.gz");
        let path = cache.join(file);
        if !path.exists() {
            let _ = writeln!(csv, "{entry},missing,,,,,,,,,not downloaded");
            continue;
        }
        let t = Instant::now();
        let loaded = load(&path);
        let (spheres, atoms) = match loaded {
            Ok(v) => v,
            Err(e) => {
                input_errors += 1;
                let _ = writeln!(
                    csv,
                    "{entry},input-error,,,,,,,,,\"{}\"",
                    e.to_string().replace('"', "'")
                );
                println!("{entry}: input error: {e}");
                continue;
            }
        };
        let h = if atoms > coarse_above { 1.0 } else { spacing };
        match compute_surface(&spheres, &SurfaceParams::ses(1.4).with_spacing(h)) {
            Ok(out) => {
                ok += 1;
                let s = out.stats;
                let _ = writeln!(
                    csv,
                    "{entry},ok,{atoms},{h},{},{},{},{},{},{:.3},",
                    s.vertices,
                    s.triangles,
                    s.components,
                    s.cavities,
                    s.euler,
                    t.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                failures += 1;
                let _ = writeln!(
                    csv,
                    "{entry},failure,{atoms},{h},,,,,,,\"{}\"",
                    e.to_string().replace('"', "'")
                );
                println!("{entry}: FAILURE: {e}");
            }
        }
    }
    fs::write(out_dir.join("sweep.csv"), csv)?;
    println!(
        "sweep: {ok} ok, {input_errors} input errors, {failures} failures; wrote docs/validation/sweep.csv"
    );
    if failures > 0 {
        bail!("{failures} surface failures");
    }
    Ok(())
}
