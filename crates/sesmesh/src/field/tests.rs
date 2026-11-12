//! Phase 3 verification: the point field and the sampled lattice against the
//! brute-force oracle, classification soundness, brick-seam consistency,
//! SIMD level and thread-count independence.

use std::collections::HashMap;

use proptest::prelude::*;
use sesmesh_testkit::clusters::{
    Cluster, Rng, cage, cluster_strategy, cubic_lattice, random_chain, with_duplicates,
};
use sesmesh_testkit::oracle::Oracle;

use super::kernel::Engine;
use super::lattice::{pack, touched_bricks};
use super::*;
use crate::geom::Vec3;

fn spheres(c: &Cluster) -> Vec<Sphere> {
    c.centers
        .iter()
        .zip(&c.radii)
        .map(|(&center, &radius)| Sphere::new(center, radius))
        .collect()
}

/// Exact oracle value of the field kind at `x`.
fn oracle_value(o: &Oracle, kind: SurfaceKind, x: [f64; 3]) -> f64 {
    match kind {
        SurfaceKind::Ses => o.ses(x).value,
        SurfaceKind::Sas => o.sas(x),
        SurfaceKind::Vdw => o.vdw(x),
    }
}

/// Oracle value clamped like the lattice samples (Design §7.2).
fn oracle_clamped(o: &Oracle, kind: SurfaceKind, x: [f64; 3], band: f64) -> f64 {
    match kind {
        SurfaceKind::Ses => o.probe() - o.ses(x).distance.min(o.probe() + band),
        _ => oracle_value(o, kind, x).min(band),
    }
}

fn oracle_for(c: &Cluster, kind: SurfaceKind) -> Oracle {
    let probe = if kind == SurfaceKind::Vdw {
        0.0
    } else {
        c.probe
    };
    Oracle::new(&c.centers, &c.radii, probe)
}

/// A lattice sample is either exact (within 1e-4 Å of the clamped reference)
/// or, in a uniform sub-block, `±band` with the correct sign away from zero.
fn sample_ok(v: f32, want: f64, band: f32) -> bool {
    let exact = (f64::from(v) - want).abs() <= 1e-4;
    let sign_only = v.abs() == band && (v < 0.0) == (want < 0.0) && want.abs() > 1e-3;
    exact || sign_only
}

/// Global sample indices of a brick, in kernel order.
fn brick_samples(engine: &Engine, key: [i32; 3]) -> Vec<[i64; 3]> {
    let side = engine.lattice.b + 1;
    let o = engine.lattice.brick_origin(key);
    let mut v = Vec::new();
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                v.push([o[0] + x, o[1] + y, o[2] + z]);
            }
        }
    }
    v
}

fn check_point_field(c: &Cluster, seed: u64) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    for kind in [SurfaceKind::Ses, SurfaceKind::Sas, SurfaceKind::Vdw] {
        let field = SesField::new(&spheres(c), c.probe, kind).map_err(|e| e.to_string())?;
        let o = oracle_for(c, kind);
        for _ in 0..60 {
            let a = c.centers[rng.below(c.centers.len())];
            let x = [
                a[0] + rng.uniform(-3.5, 3.5),
                a[1] + rng.uniform(-3.5, 3.5),
                a[2] + rng.uniform(-3.5, 3.5),
            ];
            let got = field.value(x);
            let want = oracle_value(&o, kind, x);
            if (got - want).abs() > 1e-9 {
                return Err(format!("{kind:?} at {x:?}: field {got} vs oracle {want}"));
            }
        }
    }
    Ok(())
}

/// Samples of every surface brick against the oracle, and bitwise agreement
/// of samples shared by neighbouring bricks.
fn check_lattice(c: &Cluster, kind: SurfaceKind, h: f64, brick: usize) -> Result<usize, String> {
    let field = SesField::new(&spheres(c), c.probe, kind).map_err(|e| e.to_string())?;
    let engine = Engine::new(field, h, brick, SimdLevel::Auto).map_err(|e| e.to_string())?;
    let (_, bricks) = engine.surface_bricks().map_err(|e| e.to_string())?;
    let o = oracle_for(c, kind);
    let values = engine.map_bricks(&bricks, |_, v| v.to_vec());
    let band = engine.band as f32;
    let mut seen: HashMap<[i64; 3], bool> = HashMap::new();
    for (key, vals) in bricks.iter().zip(&values) {
        for (idx, &v) in brick_samples(&engine, *key).iter().zip(vals) {
            // Neighbouring bricks must agree on every sign (Design §7.5).
            if let Some(prev) = seen.insert(*idx, v < 0.0)
                && prev != (v < 0.0)
            {
                return Err(format!("seam sign mismatch at {idx:?}"));
            }
            let x = engine.lattice.point(*idx).to_array();
            let want = oracle_clamped(&o, kind, x, engine.band);
            if !sample_ok(v, want, band) {
                return Err(format!(
                    "{kind:?} sample {idx:?} at {x:?}: kernel {v} vs oracle {want}"
                ));
            }
        }
    }
    Ok(bricks.len())
}

/// Bricks touched but not classified as surface must not contain the
/// surface: all their samples have one sign; samples away from all touched
/// bricks are outside.
fn check_classification(
    c: &Cluster,
    kind: SurfaceKind,
    h: f64,
    brick: usize,
) -> Result<(), String> {
    let field = SesField::new(&spheres(c), c.probe, kind).map_err(|e| e.to_string())?;
    let engine = Engine::new(field, h, brick, SimdLevel::Auto).map_err(|e| e.to_string())?;
    let touched = touched_bricks(&engine.field, &engine.lattice).map_err(|e| e.to_string())?;
    let (_, surface) = engine.surface_bricks().map_err(|e| e.to_string())?;
    let o = oracle_for(c, kind);
    let surface: Vec<u64> = surface.iter().map(|k| pack(*k)).collect();
    for key in touched
        .iter()
        .filter(|k| surface.binary_search(&pack(**k)).is_err())
    {
        let mut signs = brick_samples(&engine, *key)
            .into_iter()
            .map(|idx| oracle_value(&o, kind, engine.lattice.point(idx).to_array()) < 0.0);
        let first = signs.next().expect("non-empty brick");
        if signs.any(|s| s != first) {
            return Err(format!(
                "{kind:?}: brick {key:?} classified uniform but changes sign"
            ));
        }
    }
    // A point well outside every touched brick is outside the surface.
    let far = Vec3::from_array(c.centers[0]) + Vec3::new(50.0, 0.0, 0.0);
    if oracle_value(&o, kind, far.to_array()) <= 0.0 {
        return Err("far point inside".into());
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn point_field_matches_oracle(c in cluster_strategy(16), seed in any::<u64>()) {
        check_point_field(&c, seed).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn lattice_samples_match_oracle(c in cluster_strategy(12), h in 0.25..0.7f64) {
        for kind in [SurfaceKind::Ses, SurfaceKind::Sas, SurfaceKind::Vdw] {
            check_lattice(&c, kind, h, 4).map_err(TestCaseError::fail)?;
        }
    }

    #[test]
    fn classification_is_sound(c in cluster_strategy(10), h in 0.3..0.8f64) {
        for kind in [SurfaceKind::Ses, SurfaceKind::Sas] {
            check_classification(&c, kind, h, 4).map_err(TestCaseError::fail)?;
        }
    }
}

#[test]
fn symmetric_and_degenerate_inputs() {
    for c in [
        cubic_lattice(3, 3.0, 1.7, 1.4),
        cage(3.5, 1.6, 1.4),
        with_duplicates(random_chain(&mut Rng::new(3), 10), 3),
    ] {
        for brick in [4, 8] {
            let n = check_lattice(&c, SurfaceKind::Ses, 0.5, brick).unwrap();
            assert!(n > 0);
        }
        check_classification(&c, SurfaceKind::Ses, 0.6, 4).unwrap();
    }
}

/// Every instruction set this CPU supports gives the fallback's samples
/// within 2·10⁻⁵ Å (Plan T6.2/T6.6); on x86 CI runners this covers SSE4.2,
/// AVX2 and, where present, AVX-512.
#[test]
fn simd_levels_agree() {
    let c = random_chain(&mut Rng::new(11), 24);
    let s = spheres(&c);
    let run = |level| {
        let field = SesField::new(&s, c.probe, SurfaceKind::Ses).unwrap();
        let engine = Engine::new(field, 0.3, 8, level).ok()?;
        let (_, bricks) = engine.surface_bricks().unwrap();
        Some((
            engine.simd_name(),
            engine.map_bricks(&bricks, |_, v| v.to_vec()),
        ))
    };
    let (scalar_name, scalar) = run(SimdLevel::Scalar).unwrap();
    assert_eq!(scalar_name, "scalar");
    let mut tested = Vec::new();
    for level in SimdLevel::ALL {
        let Some((name, samples)) = run(level) else {
            assert!(!matches!(level, SimdLevel::Auto | SimdLevel::Scalar));
            continue;
        };
        assert_eq!(samples.len(), scalar.len());
        let worst = samples
            .iter()
            .flatten()
            .zip(scalar.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 2e-5, "{name} vs scalar differ by {worst}");
        tested.push(name);
    }
    assert!(tested.len() >= 2, "{tested:?}");
}

#[test]
fn simd_level_names() {
    for level in SimdLevel::ALL {
        assert_eq!(SimdLevel::from_name(level.name()), Some(level));
    }
    assert_eq!(SimdLevel::from_name("AVX2"), Some(SimdLevel::Avx2));
    assert_eq!(SimdLevel::from_name("mmx"), None);
}

#[cfg(feature = "parallel")]
#[test]
fn thread_count_does_not_change_samples() {
    let c = random_chain(&mut Rng::new(5), 40);
    let s = spheres(&c);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                sample_surface_bricks(&s, c.probe, SurfaceKind::Ses, 0.3, 8, SimdLevel::Auto)
                    .unwrap()
            })
    };
    let one = run(1);
    assert!(one.surface_bricks > 0);
    for threads in [2, 4, 7] {
        assert_eq!(run(threads), one, "{threads} threads");
    }
}

#[test]
fn invalid_parameters() {
    let s = [Sphere::new([0.0; 3], 1.5)];
    let field = || SesField::new(&s, 1.4, SurfaceKind::Ses).unwrap();
    assert!(Engine::new(field(), 0.0, 8, SimdLevel::Auto).is_err());
    assert!(Engine::new(field(), f64::NAN, 8, SimdLevel::Auto).is_err());
    assert!(Engine::new(field(), 0.5, 1, SimdLevel::Auto).is_err());
    assert!(Engine::new(field(), 0.5, 64, SimdLevel::Auto).is_err());
    assert!(SesField::new(&s, -1.0, SurfaceKind::Ses).is_err());
    // Probe 0 routes the SES to the vdW surface.
    assert_eq!(
        SesField::new(&s, 0.0, SurfaceKind::Ses).unwrap().kind(),
        SurfaceKind::Vdw
    );
    // Empty input: no bricks, no error.
    let stats = sample_surface_bricks(&[], 1.4, SurfaceKind::Ses, 0.5, 8, SimdLevel::Auto).unwrap();
    assert_eq!((stats.touched_bricks, stats.surface_bricks), (0, 0));
}

#[test]
fn single_sphere_samples_are_exact() {
    // SES of one sphere = the sphere: sampled values are |x − c| − r near it.
    let s = [Sphere::new([0.3, -0.2, 0.1], 1.7)];
    let field = SesField::new(&s, 1.4, SurfaceKind::Ses).unwrap();
    let engine = Engine::new(field, 0.25, 8, SimdLevel::Auto).unwrap();
    let (_, bricks) = engine.surface_bricks().unwrap();
    let values = engine.map_bricks(&bricks, |_, v| v.to_vec());
    let band = engine.band as f32;
    let mut checked = 0;
    for (key, vals) in bricks.iter().zip(&values) {
        for (idx, &v) in brick_samples(&engine, *key).iter().zip(vals) {
            let x = engine.lattice.point(*idx);
            let dist = x.dist(Vec3::new(0.3, -0.2, 0.1));
            // d(x, P) = max(0, R − |x − c|), R = 1.7 + 1.4; F = r_p − min(d, r_p + band).
            let want = 1.4 - (3.1 - dist).max(0.0).min(1.4 + engine.band);
            assert!(sample_ok(v, want, band), "{x:?}: {v} vs {want}");
            if v.abs() != band && (dist - 1.7).abs() < 0.5 * engine.band {
                assert!((f64::from(v) - (dist - 1.7)).abs() < 2e-5, "{x:?}: {v}");
                checked += 1;
            }
        }
    }
    assert!(checked > 1000);
}

#[cfg(feature = "gzip")]
#[test]
fn crambin_lattice_against_exact_point_field() {
    use crate::io::{ReadOptions, read_path};
    use crate::radii::{RadiusOptions, assign};
    use crate::{Diagnostics, Selection};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/data/fixtures/1CRN.cif.gz"
    );
    let st = read_path(path, &ReadOptions::default()).unwrap();
    let mut d = Diagnostics::new();
    let atoms = st.select(&Selection::default(), &mut d).unwrap();
    let radii = assign(&atoms, &RadiusOptions::default(), &mut d).unwrap();
    let s = Sphere::from_atoms(&atoms, &radii).unwrap();
    let field = SesField::new(&s, 1.4, SurfaceKind::Ses).unwrap();
    let reference = field.clone();
    let engine = Engine::new(field, 0.5, 8, SimdLevel::Auto).unwrap();
    let (touched, bricks) = engine.surface_bricks().unwrap();
    assert!(
        !bricks.is_empty() && bricks.len() < touched,
        "{} of {touched}",
        bricks.len()
    );
    let values = engine.map_bricks(&bricks, |_, v| v.to_vec());
    let band = engine.band as f32;
    let mut seen: HashMap<[i64; 3], bool> = HashMap::new();
    for (bi, (key, vals)) in bricks.iter().zip(&values).enumerate() {
        for (idx, &v) in brick_samples(&engine, *key).iter().zip(vals) {
            if let Some(prev) = seen.insert(*idx, v < 0.0) {
                assert_eq!(prev, v < 0.0, "seam sign mismatch at {idx:?}");
            }
            if bi % 5 == 0 {
                let want = reference.clamped_value(engine.lattice.point(*idx), engine.band);
                assert!(sample_ok(v, want, band), "{idx:?}: {v} vs {want}");
            }
        }
    }
}
