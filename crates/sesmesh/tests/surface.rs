//! End-to-end surface tests against analytic references (Plan T4.2–T4.5,
//! T4.9): mesh invariants, vertex and normal accuracy, convergence.

use sesmesh::export::obj::{ObjOptions, write_obj};
use sesmesh::{
    RefineMode, SesField, Sphere, SurfaceKind, SurfaceOutput, SurfaceParams, compute_surface, mesh,
};
use sesmesh_testkit::{analytic, obj};

fn run(spheres: &[Sphere], params: &SurfaceParams) -> SurfaceOutput {
    let out = compute_surface(spheres, params).expect("surface");
    let v = mesh::validate(&out.mesh);
    assert!(v.is_ok(), "{v:?}");
    out
}

/// Largest |F(v)| over the vertices and the largest angle (degrees) between
/// mesh and analytic normals.
fn accuracy(spheres: &[Sphere], params: &SurfaceParams, out: &SurfaceOutput) -> (f64, f64) {
    let field = SesField::new(spheres, params.effective_probe(), params.kind).unwrap();
    let mut max_f = 0.0f64;
    let mut max_angle = 0.0f64;
    for (p, n) in out.mesh.positions.iter().zip(&out.mesh.normals) {
        let s = field.sample(p.map(f64::from));
        max_f = max_f.max(s.value.abs());
        if let Some(a) = s.normal {
            let b = n.map(f64::from);
            let dot: f64 = (0..3).map(|k| a[k] * b[k]).sum();
            let cross = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
            max_angle = max_angle.max(sin.atan2(dot).to_degrees());
        }
    }
    (max_f, max_angle)
}

/// Bound on |F(v)|: the edge-parameter clamp (10⁻⁴·h), refinement
/// tolerance and `f32` storage.
fn max_f(h: f64) -> f64 {
    1e-4 * h + 2e-6
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn single_sphere_all_kinds() {
    let s = [Sphere::new([0.3, -0.2, 0.1], 1.8)];
    for (params, r) in [
        (SurfaceParams::ses(1.4).with_spacing(0.25), 1.8),
        (SurfaceParams::sas(1.4).with_spacing(0.25), 3.2),
        (SurfaceParams::vdw().with_spacing(0.25), 1.8),
    ] {
        let out = run(&s, &params);
        let want = analytic::single_sphere(r);
        assert_eq!(out.stats.components, 1);
        assert_eq!(out.stats.euler, 2);
        // An inscribed mesh underestimates both (volume about twice as much
        // as area); see `convergence_is_quadratic`.
        assert!(
            rel(out.stats.area, want.area) < 5e-3,
            "{params:?} {}",
            out.stats.area
        );
        assert!(
            rel(out.stats.volume, want.volume) < 1.2e-2,
            "{}",
            out.stats.volume
        );
        let (f, angle) = accuracy(&s, &params, &out);
        assert!(f < max_f(0.25), "{params:?}: |F| {f}");
        assert!(angle < 0.01, "{params:?}: normal {angle}°");
    }
}

#[test]
fn two_spheres_match_reference() {
    // Overlapping, touching-by-probe and self-intersecting torus regimes.
    for (r1, r2, d) in [
        (1.7, 1.5, 2.5),
        (1.6, 1.6, 4.0),
        (1.8, 1.2, 5.0),
        (1.5, 1.5, 5.6),
    ] {
        let s = [
            Sphere::new([0.0, 0.0, 0.0], r1),
            Sphere::new([d, 0.0, 0.0], r2),
        ];
        let params = SurfaceParams::ses(1.4).with_spacing(0.25);
        let out = run(&s, &params);
        let want = analytic::two_spheres(r1, r2, d, 1.4);
        assert_eq!(out.stats.components, want.components, "{r1} {r2} {d}");
        // Measured at 0.25 Å: area −0.4…−0.65 %, volume −0.8…−1.35 % (small
        // radii are the worst case for an inscribed mesh; docs/02 §15).
        let (ea, ev) = (
            rel(out.stats.area, want.area),
            rel(out.stats.volume, want.volume),
        );
        assert!(
            ea < 7e-3,
            "{r1} {r2} {d}: area {} vs {}",
            out.stats.area,
            want.area
        );
        assert!(
            ev < 1.5e-2,
            "{r1} {r2} {d}: volume {} vs {}",
            out.stats.volume,
            want.volume
        );
        let (f, _) = accuracy(&s, &params, &out);
        assert!(f < max_f(0.25), "|F| {f}");
    }
}

/// Area and volume errors shrink as O(h²) (NFR-ACC).
#[test]
fn convergence_is_quadratic() {
    let cases: [(&[Sphere], analytic::AreaVolume); 2] = [
        (
            &[Sphere::new([0.1, 0.2, 0.3], 1.8)],
            analytic::single_sphere(1.8),
        ),
        (
            &[
                Sphere::new([0.0, 0.0, 0.0], 1.7),
                Sphere::new([3.1, 0.2, 0.0], 1.5),
            ],
            analytic::two_spheres(1.7, 1.5, 3.1f64.hypot(0.2), 1.4),
        ),
    ];
    for (spheres, want) in cases {
        let mut errors = Vec::new();
        for h in [1.0, 0.5, 0.25, 0.125] {
            let out = run(spheres, &SurfaceParams::ses(1.4).with_spacing(h));
            errors.push((
                h,
                rel(out.stats.area, want.area),
                rel(out.stats.volume, want.volume),
            ));
        }
        let order = |e0: f64, e1: f64| (e0 / e1).log2();
        let n = errors.len();
        let (_, a0, v0) = errors[n - 3];
        let (_, a1, v1) = errors[n - 1];
        // Average order over the last two halvings.
        let (oa, ov) = (order(a0, a1) / 2.0, order(v0, v1) / 2.0);
        assert!(
            (1.5..2.8).contains(&oa) && (1.5..2.8).contains(&ov),
            "{errors:?}: orders {oa} {ov}"
        );
        assert!(
            errors[n - 1].1 < 2e-3 && errors[n - 1].2 < 3.5e-3,
            "{errors:?}"
        );
    }
}

#[test]
fn brick_sizes_agree() {
    let s = [
        Sphere::new([0.0, 0.0, 0.0], 1.7),
        Sphere::new([2.6, 0.3, 0.0], 1.5),
        Sphere::new([1.2, 2.4, 0.5], 1.6),
        Sphere::new([1.0, 1.0, 2.3], 1.8),
    ];
    let base = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.3));
    for b in [2, 3, 5, 8, 16] {
        let out = run(
            &s,
            &SurfaceParams::ses(1.4).with_spacing(0.3).with_brick_size(b),
        );
        assert!(rel(out.stats.area, base.stats.area) < 1e-6, "brick {b}");
        assert!(
            out.stats.vertices.abs_diff(base.stats.vertices) <= 2,
            "brick {b}"
        );
    }
}

#[test]
fn linear_refinement_is_close() {
    let s = [
        Sphere::new([0.0, 0.0, 0.0], 1.7),
        Sphere::new([2.5, 0.0, 0.0], 1.5),
    ];
    let exact = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.25));
    let params = SurfaceParams::ses(1.4)
        .with_spacing(0.25)
        .with_refine(RefineMode::Linear);
    let linear = run(&s, &params);
    assert_eq!(exact.stats.vertices, linear.stats.vertices);
    assert!(rel(linear.stats.area, exact.stats.area) < 2e-3);
    assert!(linear.stats.refine_evaluations < exact.stats.refine_evaluations);
    let (f, _) = accuracy(&s, &params, &linear);
    assert!(f < 0.01, "{f}");
}

#[test]
fn empty_input_gives_empty_mesh() {
    let out = run(&[], &SurfaceParams::default());
    assert!(out.mesh.is_empty());
    assert_eq!(out.stats.components, 0);
}

#[test]
fn invalid_parameters_are_rejected() {
    let s = [Sphere::new([0.0; 3], 1.0)];
    for p in [
        SurfaceParams::ses(1.4).with_spacing(0.0),
        SurfaceParams::ses(1.4).with_spacing(f64::NAN),
        SurfaceParams::ses(-1.0),
        SurfaceParams::ses(1.4).with_brick_size(1),
    ] {
        assert!(compute_surface(&s, &p).is_err(), "{p:?}");
    }
    let _ = SurfaceKind::Ses;
}

#[test]
fn obj_round_trip() {
    let s = [
        Sphere::new([10.0, -20.0, 5.0], 1.7),
        Sphere::new([12.5, -20.0, 5.0], 1.5),
    ];
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.5));
    let m = &out.mesh;
    for (precision, normals) in [(4, true), (6, false), (0, true)] {
        let mut options = ObjOptions::default();
        options.precision = precision;
        options.normals = normals;
        options.comments = vec!["test header".into(), "two\nlines".into()];
        let mut buf = Vec::new();
        write_obj(&mut buf, m, &options).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let back = obj::parse(&text).unwrap();
        assert_eq!(back.comments[0], format!("sesmesh {}", sesmesh::VERSION));
        assert_eq!(back.comments[1..], ["test header", "two lines"]);
        assert_eq!(back.positions.len(), m.vertex_count());
        assert_eq!(back.triangles, m.triangles);
        let tol = 0.5 * 10f64.powi(-(precision as i32)) + 1e-6;
        for (a, b) in back.positions.iter().zip(&m.positions) {
            assert!((0..3).all(|k| (a[k] - f64::from(b[k])).abs() <= tol));
        }
        if normals {
            assert_eq!(back.triangle_normals, m.triangles);
            assert_eq!(back.normals.len(), m.vertex_count());
        } else {
            assert!(back.normals.is_empty() && back.triangle_normals.is_empty());
        }
    }
    let mut bad = ObjOptions::default();
    bad.precision = 10;
    assert!(write_obj(Vec::new(), m, &bad).is_err());
}

#[test]
fn obj_snapshot() {
    let mut m = sesmesh::Mesh::default();
    m.positions = vec![[0.0, 0.0, 0.0], [1.0, -0.5, 0.0], [0.0, 1.25, -2.718_49]];
    m.normals = vec![[0.0, 0.0, 1.0]; 3];
    m.triangles = vec![[0, 1, 2]];
    let mut buf = Vec::new();
    write_obj(&mut buf, &m, &ObjOptions::default()).unwrap();
    let want = format!(
        "# sesmesh {}\no ses\nv 0.0000 0.0000 0.0000\nv 1.0000 -0.5000 0.0000\n\
         v 0.0000 1.2500 -2.7185\nvn 0.0000 0.0000 1.0000\nvn 0.0000 0.0000 1.0000\n\
         vn 0.0000 0.0000 1.0000\nf 1//1 2//2 3//3\n",
        sesmesh::VERSION
    );
    assert_eq!(String::from_utf8(buf).unwrap(), want);
}
