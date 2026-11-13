//! Phase 7 mesh features (Plan T7.1–T7.5): components, cavity modes and
//! filters, per-atom areas, normal modes, OBJ groups and colours.

use sesmesh::export::obj::{ObjOptions, write_obj};
use sesmesh::{
    CavityMode, NormalMode, Sphere, SurfaceOutput, SurfaceParams, color, compute_surface, mesh,
};
use sesmesh_testkit::{obj, oracle::fibonacci_sphere};

fn run(spheres: &[Sphere], params: &SurfaceParams) -> SurfaceOutput {
    let out = compute_surface(spheres, params).expect("surface");
    assert!(
        mesh::validate(&out.mesh).is_ok(),
        "{:?}",
        mesh::validate(&out.mesh)
    );
    out
}

/// Triangles are grouped by component, consistent with `vertex_component`.
fn assert_grouped(out: &SurfaceOutput) {
    let m = &out.mesh;
    assert_eq!(m.vertex_component.len(), m.vertex_count());
    let mut next = 0;
    for (c, comp) in out.components.iter().enumerate() {
        assert_eq!(comp.first_triangle, next);
        for t in &m.triangles[comp.first_triangle..comp.first_triangle + comp.triangles] {
            for &v in t {
                assert_eq!(m.vertex_component[v as usize] as usize, c);
            }
        }
        next += comp.triangles;
    }
    assert_eq!(next, m.triangle_count());
}

/// A closed shell of atoms (radius 9 Å) with one atom at its centre: outer
/// surface (+), the shell's inner wall (cavity, −) and the central atom (+)
/// inside the cavity.
fn shell_with_core() -> Vec<Sphere> {
    let mut s: Vec<Sphere> = fibonacci_sphere([1.0, 2.0, 3.0], 9.0, 260)
        .into_iter()
        .map(|c| Sphere::new(c, 1.8))
        .collect();
    s.push(Sphere::new([1.0, 2.0, 3.0], 2.0));
    s
}

#[test]
fn separate_molecules_are_separate_components() {
    let mut s = vec![
        Sphere::new([0.0; 3], 1.7),
        Sphere::new([2.5, 0.0, 0.0], 1.6),
    ];
    s.extend([
        Sphere::new([20.0, 0.0, 0.0], 1.8),
        Sphere::new([0.0, 15.0, 0.0], 1.5),
    ]);
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.4));
    assert_eq!(out.stats.components, 3);
    assert_eq!(out.stats.cavities, 0);
    assert!(
        out.components
            .iter()
            .all(|c| c.euler == 2 && c.genus() == 0)
    );
    assert_grouped(&out);
}

#[test]
fn ring_of_atoms_has_genus_one() {
    let n = 20;
    let s: Vec<Sphere> = (0..n)
        .map(|k| {
            let a = std::f64::consts::TAU * f64::from(k) / f64::from(n);
            Sphere::new([7.0 * a.cos(), 7.0 * a.sin(), 0.0], 1.7)
        })
        .collect();
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.4));
    assert_eq!(out.stats.components, 1);
    assert_eq!(out.components[0].euler, 0);
    assert_eq!(out.components[0].genus(), 1);
}

#[test]
fn cavity_modes_and_filters() {
    let s = shell_with_core();
    let base = SurfaceParams::ses(1.4).with_spacing(0.5);
    let keep = run(&s, &base);
    assert_eq!(keep.stats.components, 3, "{:?}", keep.components);
    assert_eq!(keep.stats.cavities, 1);
    assert_grouped(&keep);
    let cavity = *keep.components.iter().find(|c| c.is_cavity()).unwrap();

    let remove = run(&s, &base.clone().with_cavities(CavityMode::Remove));
    assert_eq!(remove.stats.components, 1);
    assert_eq!(
        remove.stats.removed_components, 2,
        "cavity and the core inside it"
    );
    assert!(!remove.components[0].is_cavity());
    assert_grouped(&remove);
    // Only vertices of the kept component remain.
    assert_eq!(remove.mesh.vertex_count(), remove.components[0].vertices);

    let only = run(&s, &base.clone().with_cavities(CavityMode::Only));
    assert_eq!(only.stats.components, 1);
    assert!(only.components[0].is_cavity());
    assert!((only.stats.volume - cavity.volume).abs() < 1e-6 * cavity.volume.abs());

    // A cavity smaller than the filter is dropped together with its content.
    let small = run(
        &s,
        &base.clone().with_min_cavity_volume(-cavity.volume + 1.0),
    );
    assert_eq!(small.stats.components, 1);
    let tiny = run(
        &s,
        &base.clone().with_min_cavity_volume(-cavity.volume - 1.0),
    );
    assert_eq!(tiny.stats.components, 3);

    // Area filter: drop the central atom's surface (the smallest component).
    let core_area = keep
        .components
        .iter()
        .map(|c| c.area)
        .fold(f64::INFINITY, f64::min);
    let filtered = run(&s, &base.clone().with_min_component_area(core_area + 1.0));
    assert_eq!(filtered.stats.components, 2);
    assert!(compute_surface(&s, &base.with_min_component_area(-1.0)).is_err());
}

#[test]
fn atom_areas_sum_to_total() {
    let s = shell_with_core();
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.5));
    let areas = mesh::atom_areas(&out.mesh, s.len());
    let total: f64 = areas.iter().sum();
    let mesh_area: f64 = out.components.iter().map(|c| c.area).sum();
    assert!(
        (total - mesh_area).abs() < 1e-9 * mesh_area,
        "{total} vs {mesh_area}"
    );
    // The central atom carries the whole core component.
    let core = *areas.last().unwrap();
    assert!(
        (core - 4.0 * std::f64::consts::PI * 4.0).abs() < 0.02 * core,
        "{core}"
    );
}

#[test]
fn normal_modes() {
    let s = [
        Sphere::new([0.0; 3], 1.7),
        Sphere::new([2.6, 0.0, 0.0], 1.6),
    ];
    let p = SurfaceParams::ses(1.4).with_spacing(0.3);
    let analytic = run(&s, &p);
    let face = run(&s, &p.clone().with_normals(NormalMode::FaceAverage));
    let none = run(&s, &p.with_normals(NormalMode::None));
    assert!(none.mesh.normals.is_empty());
    assert_eq!(face.mesh.positions, analytic.mesh.positions);
    // Face-averaged normals stay close to the analytic ones.
    let mean_dot: f64 = face
        .mesh
        .normals
        .iter()
        .zip(&analytic.mesh.normals)
        .map(|(a, b)| {
            (0..3)
                .map(|k| f64::from(a[k]) * f64::from(b[k]))
                .sum::<f64>()
        })
        .sum::<f64>()
        / face.mesh.vertex_count() as f64;
    assert!(mean_dot > 0.99, "{mean_dot}");
    // A mesh without normals is written without `vn`.
    let mut buf = Vec::new();
    write_obj(&mut buf, &none.mesh, &ObjOptions::default()).unwrap();
    let back = obj::parse(&String::from_utf8(buf).unwrap()).unwrap();
    assert!(back.normals.is_empty() && back.triangle_normals.is_empty());
}

#[test]
fn obj_groups_and_colors() {
    let s = shell_with_core();
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.7));
    let mut o = ObjOptions::default();
    o.groups = out
        .components
        .iter()
        .enumerate()
        .map(|(c, comp)| (comp.first_triangle, format!("component_{c}")))
        .collect();
    o.vertex_colors = out
        .mesh
        .vertex_component
        .iter()
        .map(|&c| color::categorical(c as usize))
        .collect();
    let mut buf = Vec::new();
    write_obj(&mut buf, &out.mesh, &o).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let groups: Vec<&str> = text.lines().filter(|l| l.starts_with("g ")).collect();
    assert_eq!(groups, ["g component_0", "g component_1", "g component_2"]);
    let v = text.lines().find(|l| l.starts_with("v ")).unwrap();
    assert_eq!(v.split_whitespace().count(), 7, "{v}");
    let back = obj::parse(&text).unwrap();
    assert_eq!(back.triangles, out.mesh.triangles);
    // Faces follow their group headers in component order.
    let mut faces_before_group = Vec::new();
    let mut faces = 0;
    for line in text.lines() {
        if line.starts_with("g ") {
            faces_before_group.push(faces);
        } else if line.starts_with("f ") {
            faces += 1;
        }
    }
    let firsts: Vec<usize> = out.components.iter().map(|c| c.first_triangle).collect();
    assert_eq!(faces_before_group, firsts);
    // Wrong colour count is rejected.
    o.vertex_colors.pop();
    assert!(write_obj(Vec::new(), &out.mesh, &o).is_err());
}
