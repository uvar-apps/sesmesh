// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Meshes the analytic references used by `crates/sesmesh/tests/surface.rs`
//! (one sphere; four two-sphere configurations covering overlapping,
//! probe-bridged and self-intersecting-torus regimes) at several grid
//! spacings and prints one CSV row per (case, spacing): mesh and reference
//! area/volume, relative errors, the largest |F| at a vertex and the largest
//! angle between mesh and analytic normals.

use sesmesh::{SesField, Sphere, SurfaceParams, compute_surface, mesh};
use sesmesh_testkit::analytic;

fn main() {
    let rp = 1.4;
    // (label, spheres, reference) — same configurations as the tests.
    let mut cases: Vec<(String, Vec<Sphere>, analytic::AreaVolume)> = vec![(
        "single r=1.8".to_string(),
        vec![Sphere::new([0.1, 0.2, 0.3], 1.8)],
        analytic::single_sphere(1.8),
    )];
    for (r1, r2, d) in [
        (1.7, 1.5, 2.5),
        (1.6, 1.6, 4.0),
        (1.8, 1.2, 5.0),
        (1.5, 1.5, 5.6),
    ] {
        cases.push((
            format!("pair r={r1}/{r2} d={d}"),
            vec![
                Sphere::new([0.0, 0.0, 0.0], r1),
                Sphere::new([d, 0.0, 0.0], r2),
            ],
            analytic::two_spheres(r1, r2, d, rp),
        ));
    }
    println!(
        "case,spacing,vertices,triangles,components,ref_components,area,ref_area,volume,ref_volume,area_err,volume_err,max_abs_f,max_normal_deg"
    );
    for (label, spheres, want) in &cases {
        let field = SesField::new(spheres, rp, sesmesh::SurfaceKind::Ses).expect("field");
        for h in [1.0, 0.5, 0.25, 0.125] {
            let params = SurfaceParams::ses(rp).with_spacing(h);
            let out = compute_surface(spheres, &params).expect("surface");
            assert!(mesh::validate(&out.mesh).is_ok(), "{label} {h}: invalid mesh");
            let (mut max_f, mut max_angle) = (0.0f64, 0.0f64);
            for (p, n) in out.mesh.positions.iter().zip(&out.mesh.normals) {
                let s = field.sample(p.map(f64::from));
                max_f = max_f.max(s.value.abs());
                if let Some(a) = s.normal {
                    let b = n.map(f64::from);
                    let dot: f64 = (0..3).map(|k| a[k] * b[k]).sum();
                    let c = [
                        a[1] * b[2] - a[2] * b[1],
                        a[2] * b[0] - a[0] * b[2],
                        a[0] * b[1] - a[1] * b[0],
                    ];
                    let sin = c.iter().map(|x| x * x).sum::<f64>().sqrt();
                    max_angle = max_angle.max(sin.atan2(dot).to_degrees());
                }
            }
            let st = &out.stats;
            println!(
                "\"{label}\",{h},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6e},{:.6e},{:.3e},{:.3e}",
                st.vertices,
                st.triangles,
                st.components,
                want.components,
                st.area,
                want.area,
                st.volume,
                want.volume,
                (st.area - want.area) / want.area,
                (st.volume - want.volume) / want.volume,
                max_f,
                max_angle,
            );
        }
    }
}
