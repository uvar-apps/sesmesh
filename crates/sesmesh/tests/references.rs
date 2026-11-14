// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-tool references (GitHub issues #5, #6; Design §15 "Cross-tool").
//!
//! `tests/data/reference/*.toml` hold numbers measured offline with other
//! programs on the same spheres (tools/compare): area, volume and cavity
//! count of their meshes for the validation corpus at h = 0.5 and 0.25 Å.
//! NanoShaper (ray casting of the analytic patches) and ESES (exact
//! intersections on a Cartesian grid) also place their vertices on the
//! exact SES, so at the same grid spacing their meshes must agree with
//! sesmesh's to the discretization noise: measured over the 53-entry corpus,
//! at most 0.026 % (area) and 0.006 % (volume) at 0.25 Å, 0.10 % and 0.035 %
//! at 0.5 Å. The tolerances below leave a factor of about two.
//!
//! The vendored fixtures run in CI; the rest of the corpus runs with
//! `--include-ignored` when `tests/data/cache` holds it (`cargo xtask fetch
//! --set validation`).
#![cfg(feature = "gzip")]

use std::path::{Path, PathBuf};

use sesmesh::radii::{RadiusOptions, assign};
use sesmesh::{Diagnostics, Selection, Sphere, SurfaceParams, compute_surface, io};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// One reference entry: (area, volume, cavities) for (id, spacing).
struct Ref {
    id: String,
    spacing: f64,
    area: f64,
    volume: f64,
    cavities: usize,
}

fn references(tool: &str) -> Vec<Ref> {
    let path = repo()
        .join("tests/data/reference")
        .join(format!("{tool}.toml"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let table: toml::Table = text
        .parse()
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    table["entry"]
        .as_array()
        .expect("[[entry]] array")
        .iter()
        .filter(|e| e["status"].as_str() == Some("ok"))
        .map(|e| Ref {
            id: e["id"].as_str().unwrap().to_string(),
            spacing: e["spacing"].as_float().unwrap(),
            area: e["area"].as_float().unwrap(),
            volume: e["volume"].as_float().unwrap(),
            cavities: usize::try_from(e["cavities"].as_integer().unwrap()).unwrap(),
        })
        .collect()
}

fn spheres(path: &Path) -> Vec<Sphere> {
    let s = io::read_path(path, &io::ReadOptions::default()).unwrap();
    let mut diags = Diagnostics::new();
    let atoms = s.select(&Selection::default(), &mut diags).unwrap();
    let radii = assign(&atoms, &RadiusOptions::default(), &mut diags).unwrap();
    Sphere::from_atoms(&atoms, &radii).unwrap()
}

/// Compares sesmesh with the references of `tools` for the entries whose
/// structure file `locate` finds (each surface computed once, compared with
/// every tool); returns how many (tool, entry, spacing) triples were
/// compared. Entries with more than `max_atoms_fine` atoms are skipped at
/// h < 0.5 Å (CI time in debug builds). Cavity counts are compared only when
/// `cavities` is set: on large structures, grids of different origin can
/// split or join cavities at necks thinner than h, so counts differ by a few
/// while areas and volumes still agree.
fn compare(
    tools: &[&str],
    cavities: bool,
    max_atoms_fine: usize,
    locate: impl Fn(&str) -> Option<PathBuf>,
) -> usize {
    let mut refs: Vec<(&str, Ref)> = tools
        .iter()
        .flat_map(|t| references(t).into_iter().map(move |r| (*t, r)))
        .collect();
    refs.sort_by(|a, b| (&a.1.id, a.1.spacing.to_bits()).cmp(&(&b.1.id, b.1.spacing.to_bits())));
    let mut n = 0;
    let mut current: Option<(String, Vec<Sphere>)> = None;
    let mut surface: Option<(String, u64, sesmesh::SurfaceStats)> = None;
    for (tool, r) in &refs {
        let Some(path) = locate(&r.id) else { continue };
        if current.as_ref().is_none_or(|(id, _)| *id != r.id) {
            current = Some((r.id.clone(), spheres(&path)));
        }
        let s = &current.as_ref().unwrap().1;
        if r.spacing < 0.5 && s.len() > max_atoms_fine {
            continue;
        }
        if surface
            .as_ref()
            .is_none_or(|(id, h, _)| *id != r.id || *h != r.spacing.to_bits())
        {
            let out = compute_surface(s, &SurfaceParams::ses(1.4).with_spacing(r.spacing)).unwrap();
            surface = Some((r.id.clone(), r.spacing.to_bits(), out.stats));
        }
        let st = &surface.as_ref().unwrap().2;
        let (tol_a, tol_v) = if r.spacing <= 0.25 {
            (5e-4, 1e-4)
        } else {
            (2e-3, 1e-3)
        };
        let (ea, ev) = (st.area / r.area - 1.0, st.volume / r.volume - 1.0);
        assert!(
            ea.abs() <= tol_a && ev.abs() <= tol_v,
            "{tool} {} h={}: area {:.3} vs {:.3} ({:+.4} %), volume {:.3} vs {:.3} ({:+.4} %)",
            r.id,
            r.spacing,
            st.area,
            r.area,
            100.0 * ea,
            st.volume,
            r.volume,
            100.0 * ev
        );
        if cavities {
            assert_eq!(
                st.cavities, r.cavities,
                "{tool} {} h={}: cavities",
                r.id, r.spacing
            );
        }
        n += 1;
    }
    n
}

fn fixture(id: &str) -> Option<PathBuf> {
    let p = repo()
        .join("tests/data/fixtures")
        .join(format!("{id}.cif.gz"));
    p.exists().then_some(p)
}

fn cached(id: &str) -> Option<PathBuf> {
    let p = repo().join("tests/data/cache").join(format!("{id}.cif.gz"));
    p.exists().then_some(p)
}

#[test]
fn fixtures_match_nanoshaper_and_eses() {
    // 1CRN, 1EJG, 1L2Y, 1UBQ, 2PTN, 3FX2 at 0.5 Å, the four smallest also
    // at 0.25 Å (2PTN and 3FX2 at 0.25 Å run in the ignored corpus test).
    assert_eq!(compare(&["nanoshaper", "eses"], true, 1000, fixture), 20);
}

/// Issue #6: 2PTN has 14 cavities in sesmesh, NanoShaper and ESES at both
/// spacings (MSMS reports 16, see docs/comparison.md); sesmesh's count at
/// 0.5 Å is checked by `fixtures_match_nanoshaper_and_eses`.
#[test]
fn trypsin_cavities_match() {
    for tool in ["nanoshaper", "eses"] {
        let r: Vec<_> = references(tool)
            .into_iter()
            .filter(|r| r.id == "2PTN")
            .collect();
        assert_eq!(r.len(), 2, "{tool}");
        assert!(r.iter().all(|r| r.cavities == 14), "{tool}");
    }
}

/// The whole validation corpus (nightly; needs `tests/data/cache`).
#[test]
#[ignore = "needs the validation set in tests/data/cache; minutes in release"]
fn corpus_matches_nanoshaper_and_eses() {
    if compare(&["nanoshaper", "eses"], false, usize::MAX, cached) == 0 {
        eprintln!("skipping: validation set not in tests/data/cache");
    }
}
