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

#[test]
fn density_mode_hits_its_target() {
    let s = shell_with_core();
    for density in [2.0, 6.0] {
        let p = SurfaceParams::ses(1.4).with_density(density);
        assert!((p.spacing() - (1.5f64 / density).sqrt()).abs() < 1e-12);
        let out = run(&s, &p);
        // Calibrated on the validation set: vertices·h²/area = 1.4999 ± 0.0025
        // (53 entries, h ≤ 0.5 Å; docs/validation-report.md).
        let achieved = out.stats.vertices as f64 / out.stats.area;
        assert!(
            (achieved / density - 1.0).abs() < 0.05,
            "{density}: {achieved}"
        );
    }
    assert!(compute_surface(&s, &SurfaceParams::ses(1.4).with_density(0.0)).is_err());
}

#[test]
fn memory_guard_refuses_large_meshes() {
    let s = shell_with_core();
    let p = SurfaceParams::ses(1.4).with_spacing(0.3);
    let out = run(&s, &p.clone().with_max_memory(u64::MAX));
    // The guard's triangle count (MC cases of a brick sample) is close to
    // the mesh and errs high.
    let tris = out.stats.estimated_triangles.unwrap();
    let real = out.stats.triangles as f64;
    assert!(
        tris as f64 >= real && (tris as f64) < 1.3 * real,
        "{tris} vs {real}"
    );
    let bytes = out.stats.estimated_memory.unwrap();
    let err = compute_surface(&s, &p.clone().with_max_memory(bytes - 1)).unwrap_err();
    assert!(matches!(err, sesmesh::Error::ResourceLimit { .. }), "{err}");
    let fits = run(&s, &p.clone().with_max_memory(bytes));
    assert_eq!(fits.mesh, out.mesh);
    assert_eq!(run(&s, &p).stats.estimated_memory, None);
    // The closed-form estimate is an upper envelope of the triangle count.
    let (closed, _) =
        sesmesh::estimate_memory(out.stats.surface_bricks, p.brick_size, 0.3, s.len());
    assert!(closed as f64 >= real, "{closed} vs {real}");
}

#[test]
fn smoothing_keeps_vertices_on_the_surface() {
    let s = shell_with_core();
    let p = SurfaceParams::ses(1.4).with_spacing(0.5);
    let raw = run(&s, &p);
    let smooth = run(&s, &p.clone().with_smoothing(5));
    assert_eq!(smooth.mesh.triangles, raw.mesh.triangles);
    let field = sesmesh::SesField::new(&s, 1.4, sesmesh::SurfaceKind::Ses).unwrap();
    let worst = smooth
        .mesh
        .positions
        .iter()
        .map(|q| field.value(q.map(f64::from)).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-4, "{worst}");
    // Fewer slivers.
    let slivers = |m: &sesmesh::Mesh| {
        m.triangles
            .iter()
            .filter(|t| {
                let p = t.map(|i| m.positions[i as usize].map(f64::from));
                let ang = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| {
                    let u = [0, 1, 2].map(|k| b[k] - a[k]);
                    let v = [0, 1, 2].map(|k| c[k] - a[k]);
                    let d: f64 = (0..3).map(|k| u[k] * v[k]).sum();
                    let nu = (0..3).map(|k| u[k] * u[k]).sum::<f64>().sqrt();
                    let nv = (0..3).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
                    (d / (nu * nv)).clamp(-1.0, 1.0).acos().to_degrees()
                };
                ang(p[0], p[1], p[2])
                    .min(ang(p[1], p[2], p[0]))
                    .min(ang(p[2], p[0], p[1]))
                    < 10.0
            })
            .count()
    };
    let (before, after) = (slivers(&raw.mesh), slivers(&smooth.mesh));
    assert!(after * 2 < before, "{before} -> {after}");
    assert!((smooth.stats.area / raw.stats.area - 1.0).abs() < 0.01);
}

#[test]
fn ply_round_trip() {
    use sesmesh::export::ply::{PlyOptions, write_ply};
    let s = shell_with_core();
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.7));
    let mut o = PlyOptions::new();
    o.comments = vec!["test".into()];
    o.vertex_colors = out
        .mesh
        .vertex_patch
        .iter()
        .map(|&p| color::patch_color(p))
        .collect();
    let mut buf = Vec::new();
    write_ply(&mut buf, &out.mesh, &o).unwrap();
    let back = sesmesh_testkit::ply::parse(&buf).unwrap();
    assert_eq!(back.comments[1], "test");
    assert_eq!(back.vertices.len(), out.mesh.vertex_count());
    let names: Vec<&str> = back.vertex_properties.iter().map(String::as_str).collect();
    assert_eq!(
        names,
        [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "atom",
            "patch",
            "component"
        ]
    );
    let x = back.column("x").unwrap();
    assert!(
        x.iter()
            .zip(&out.mesh.positions)
            .all(|(a, p)| *a == f64::from(p[0]))
    );
    let atom = back.column("atom").unwrap();
    assert!(
        atom.iter()
            .zip(&out.mesh.vertex_atom)
            .all(|(a, &b)| *a == f64::from(b))
    );
    let faces: Vec<[u32; 3]> = back.faces.iter().map(|f| [f[0], f[1], f[2]]).collect();
    assert_eq!(faces, out.mesh.triangles);
}

#[test]
fn msms_files() {
    use sesmesh::export::msms::{MsmsHeader, write_msms};
    let s = [
        Sphere::new([0.0; 3], 1.7),
        Sphere::new([2.6, 0.0, 0.0], 1.6),
    ];
    let out = run(&s, &SurfaceParams::ses(1.4).with_spacing(0.5));
    let (mut v, mut f) = (Vec::new(), Vec::new());
    let header = MsmsHeader::new("two atoms", 2, 3.0, 1.4);
    write_msms(&mut v, &mut f, &out.mesh, &header).unwrap();
    let (v, f) = (String::from_utf8(v).unwrap(), String::from_utf8(f).unwrap());
    let vl: Vec<&str> = v.lines().collect();
    assert_eq!(
        vl[0],
        "# MSMS solvent excluded surface vertices for two atoms"
    );
    assert_eq!(vl[1], "#vertices #sphere density probe_r");
    assert_eq!(
        vl[2],
        format!(
            "{:7} {:7} {:5.2} {:5.2}",
            out.mesh.vertex_count(),
            2,
            3.0,
            1.4
        )
    );
    assert_eq!(vl.len(), 3 + out.mesh.vertex_count());
    // Fixed-width fields: 6 × 9 + 8 + 8 + 3 characters.
    assert!(
        vl[3..].iter().all(|l| l.len() == 6 * 9 + 5 + 2 * 8 + 3),
        "{}",
        vl[3]
    );
    let fl: Vec<&str> = f.lines().collect();
    assert_eq!(fl.len(), 3 + out.mesh.triangle_count());
    let first: Vec<u32> = fl[3]
        .split_whitespace()
        .take(3)
        .map(|t| t.parse().unwrap())
        .collect();
    assert_eq!(first, out.mesh.triangles[0].map(|i| i + 1));
    // Every vertex has a valid 1-based sphere and an MSMS type.
    for l in &vl[3..] {
        let t: Vec<&str> = l.split_whitespace().collect();
        let sphere: usize = t[7].parse().unwrap();
        assert!((1..=2).contains(&sphere));
        assert!(["1", "2", "3"].contains(&t[8]));
    }
}

#[cfg(feature = "gzip")]
fn fixture_path(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data")
        .join(rel)
}

#[test]
#[cfg(feature = "gzip")] // reads cached .cif.gz files
fn identity_assembly_keeps_the_deposited_atoms() {
    use sesmesh::io::{ReadOptions, read_assemblies, read_path};
    for rel in ["fixtures/1CRN.pdb.gz", "fixtures/1CRN.cif.gz"] {
        let path = fixture_path(rel);
        let list = read_assemblies(&path, &ReadOptions::default()).unwrap();
        assert_eq!(list.len(), 1, "{rel}");
        assert_eq!((list[0].id.as_str(), list[0].copies), ("1", 1));
        let plain = read_path(&path, &ReadOptions::default()).unwrap();
        let mut opts = ReadOptions::default();
        opts.assembly = Some("1".into());
        let asm = read_path(&path, &opts).unwrap();
        assert_eq!(asm.atom_count(), plain.atom_count(), "{rel}");
        opts.assembly = Some("7".into());
        assert!(read_path(&path, &opts).is_err());
    }
}

/// 1K4R assembly 1 is the 60-copy capsid (≈ 545 k atoms, Spec R15).
#[test]
#[cfg(feature = "gzip")] // reads cached .cif.gz files
fn capsid_assembly() {
    use sesmesh::io::{ReadOptions, read_path};
    let path = fixture_path("cache/1K4R.cif.gz");
    if !path.exists() {
        eprintln!(
            "skipping: {} not found (cargo xtask fetch --set bench)",
            path.display()
        );
        return;
    }
    let mut opts = ReadOptions::default();
    opts.assembly = Some("1".into());
    let s = read_path(&path, &opts).unwrap();
    assert_eq!(s.atom_count(), 9_084 * 60);
}

/// Decimation through `compute_surface` (FR-LOD-4): target reached, mesh
/// valid, components kept, retained vertices still on the exact SES.
#[test]
fn decimation_keeps_vertices_on_the_surface() {
    use sesmesh::mesh::Decimation;
    let s = shell_with_core();
    let p = SurfaceParams::ses(1.4).with_spacing(0.4);
    let full = run(&s, &p);
    let half = run(&s, &p.clone().with_decimation(Decimation::ratio(0.3)));
    let d = half.stats.decimation.expect("decimation stats");
    assert_eq!(d.triangles_before, full.stats.triangles);
    assert!(half.stats.triangles <= (full.stats.triangles as f64 * 0.3).ceil() as usize + 2);
    assert_eq!(half.stats.components, full.stats.components);
    assert_eq!(half.stats.cavities, full.stats.cavities);
    assert_grouped(&half);
    assert!((half.stats.area / full.stats.area - 1.0).abs() < 0.03);
    let field = sesmesh::SesField::new(&s, 1.4, sesmesh::SurfaceKind::Ses).unwrap();
    let worst = half
        .mesh
        .positions
        .iter()
        .map(|q| field.value(q.map(f64::from)).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-4, "{worst}");
    // An error bound wins over the target.
    let bounded = run(
        &s,
        &p.clone()
            .with_decimation(Decimation::ratio(0.05).with_max_error(0.01)),
    );
    assert!(bounded.stats.triangles > half.stats.triangles);
    // Deterministic.
    let again = run(&s, &p.clone().with_decimation(Decimation::ratio(0.3)));
    assert_eq!(again.mesh, half.mesh);
    assert!(compute_surface(&s, &p.with_decimation(Decimation::ratio(1.5))).is_err());
}

/// FR-LOD-6: with auto-coarsening the memory limit is met by a coarser grid
/// instead of an error.
#[test]
fn memory_guard_can_coarsen() {
    let s = shell_with_core();
    let p = SurfaceParams::ses(1.4).with_spacing(0.25);
    let full = run(&s, &p.clone().with_max_memory(u64::MAX));
    let limit = full.stats.estimated_memory.unwrap() * 3 / 5;
    assert!(compute_surface(&s, &p.clone().with_max_memory(limit)).is_err());
    let coarse = run(&s, &p.with_max_memory(limit).with_auto_coarsen(true));
    assert!(coarse.stats.spacing > 0.25, "{}", coarse.stats.spacing);
    let used = coarse.stats.estimated_memory.unwrap();
    assert!(used <= limit, "{used} > {limit}");
    assert!(coarse.stats.estimated_triangles.unwrap() >= coarse.stats.triangles as u64);
    assert_eq!(coarse.stats.components, full.stats.components);
}
