use super::*;

/// Octahedron with outward (counter-clockwise) faces.
fn octahedron() -> Mesh {
    let positions = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let triangles = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    let n = positions.len();
    Mesh {
        normals: positions.clone(),
        positions,
        triangles,
        vertex_atom: vec![0; n],
        vertex_patch: vec![PatchKind::Convex; n],
        vertex_component: Vec::new(),
    }
}

#[test]
fn octahedron_measures() {
    let m = octahedron();
    assert!(validate(&m).is_ok(), "{:?}", validate(&m));
    let (area, volume) = area_volume(&m);
    assert!((area - 4.0 * 3f64.sqrt()).abs() < 1e-12, "{area}");
    assert!((volume - 4.0 / 3.0).abs() < 1e-12, "{volume}");
    let (ids, comps) = components(&m);
    assert_eq!(ids, vec![0; 6]);
    assert_eq!(comps.len(), 1);
    assert_eq!(comps[0].euler, 2);
    assert_eq!(comps[0].genus(), 0);
    assert!(!comps[0].is_cavity());
}

#[test]
fn inverted_octahedron_is_a_cavity() {
    let mut m = octahedron();
    for t in &mut m.triangles {
        t.swap(1, 2);
    }
    let (_, comps) = components(&m);
    assert!(comps[0].is_cavity());
    assert!((comps[0].volume + 4.0 / 3.0).abs() < 1e-12);
}

#[test]
fn two_components_in_vertex_order() {
    let a = octahedron();
    let mut m = a.clone();
    let off = a.positions.len() as u32;
    m.positions
        .extend(a.positions.iter().map(|p| [p[0] + 5.0, p[1], p[2]]));
    m.normals.extend_from_slice(&a.normals);
    m.vertex_atom.extend_from_slice(&a.vertex_atom);
    m.vertex_patch.extend_from_slice(&a.vertex_patch);
    m.triangles
        .extend(a.triangles.iter().map(|t| t.map(|i| i + off)));
    let (ids, comps) = components(&m);
    assert_eq!(comps.len(), 2);
    assert_eq!(ids[0], 0);
    assert_eq!(ids[6], 1);
    assert!(validate(&m).is_ok());
}

#[test]
fn validation_finds_defects() {
    let mut m = octahedron();
    m.triangles.pop();
    let v = validate(&m);
    assert_eq!(v.boundary_edges, 3);
    let mut m = octahedron();
    m.triangles[0].swap(1, 2);
    assert_eq!(validate(&m).misoriented_edges, 3);
    let mut m = octahedron();
    let t = m.triangles[0];
    m.triangles.push(t);
    let v = validate(&m);
    assert_eq!(v.duplicate_triangles, 1);
    assert_eq!(v.non_manifold_edges, 3);
    let mut m = octahedron();
    m.triangles.push([0, 0, 1]);
    assert!(validate(&m).degenerate_triangles >= 1);
    let mut m = octahedron();
    m.triangles.push([0, 1, 99]);
    assert_eq!(validate(&m).invalid_indices, 1);
    let mut m = octahedron();
    m.normals[0] = [0.0, 0.0, 0.0];
    m.positions[1][0] = f32::NAN;
    assert_eq!(validate(&m).bad_vertices, 2);
}
