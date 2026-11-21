// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// The edge uses of the 1.3 validation: every directed edge as a key
/// `(min, max, forward)`, sorted globally, grouped into runs.
fn reference_runs(m: &Mesh) -> Vec<EdgeUses> {
    let mut keys: Vec<(u32, u32, bool)> = m
        .triangles
        .iter()
        .flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3])))
        .map(|(a, b)| (a.min(b), a.max(b), a < b))
        .collect();
    keys.sort_unstable();
    let mut runs: Vec<EdgeUses> = Vec::new();
    for (a, b, fwd) in keys {
        match runs.last_mut() {
            Some(r) if (r.0, r.1) == (a, b) => {
                if fwd {
                    r.2 += 1;
                } else {
                    r.3 += 1;
                }
            }
            _ => runs.push((a, b, usize::from(fwd), usize::from(!fwd))),
        }
    }
    runs
}

/// Random meshes with every kind of edge defect: repeated vertices, the
/// same triangle in several corner orders and orientations, edges shared
/// by many triangles, unused vertices, and an octahedron as a valid part.
fn defect_meshes() -> Vec<Mesh> {
    let mut rng = sesmesh_testkit::clusters::Rng::new(0x5e5);
    let mut out = vec![Mesh::default(), octahedron()];
    for case in 0..300 {
        let mut m = octahedron();
        let n = 6 + rng.below(1 + case % 40);
        while m.positions.len() < n {
            let p = rng.unit().map(|c| c as f32);
            m.positions.push(p);
            m.normals.push(p);
        }
        let random = rng.below(1 + case);
        for _ in 0..random {
            let t = [0, 1, 2].map(|_| rng.below(n) as u32);
            m.triangles.push(t);
        }
        // Copies of existing triangles, rotated or reversed.
        for _ in 0..rng.below(1 + case % 12) {
            let t = m.triangles[rng.below(m.triangles.len())];
            let r = rng.below(3);
            let mut c = [t[r], t[(r + 1) % 3], t[(r + 2) % 3]];
            if rng.below(2) == 1 {
                c.swap(1, 2);
            }
            m.triangles.push(c);
        }
        let n = m.positions.len();
        m.vertex_atom = vec![0; n];
        m.vertex_patch = vec![PatchKind::Convex; n];
        out.push(m);
    }
    out
}

#[test]
fn bucketed_edges_match_sorted_keys() {
    for m in defect_meshes() {
        let edges = Edges::new(&m);
        assert!(matches!(edges, Edges::Buckets { .. }));
        let runs: Vec<EdgeUses> = edges
            .par_runs(Vec::new, Vec::push)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(runs, reference_runs(&m));
    }
}

#[test]
fn validation_matches_the_sorting_algorithm() {
    let mut defects = 0;
    for m in defect_meshes() {
        let v = validate(&m);
        let (mut boundary, mut non_manifold, mut misoriented) = (0, 0, 0);
        for (a, b, fwd, bwd) in reference_runs(&m) {
            if a == b || (fwd, bwd) == (1, 1) {
                continue;
            }
            match fwd + bwd {
                1 => boundary += 1,
                2 => misoriented += 1,
                _ => non_manifold += 1,
            }
        }
        assert_eq!(v.boundary_edges, boundary);
        assert_eq!(v.non_manifold_edges, non_manifold);
        assert_eq!(v.misoriented_edges, misoriented);
        assert_eq!(v.duplicate_triangles, duplicates_by_sort(&m.triangles));
        defects += usize::from(v.duplicate_triangles > 0);
    }
    assert!(defects > 100, "{defects} meshes with duplicates");
}

#[test]
fn duplicates_with_repeated_vertices() {
    let mut m = octahedron();
    // Same vertex sets in every corner order, with repeated vertices.
    m.triangles.extend([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [2, 2, 2],
        [2, 2, 2],
        [3, 4, 3],
        [3, 3, 4],
        [4, 3, 4],
    ]);
    assert_eq!(validate(&m).duplicate_triangles, 4);
    assert_eq!(duplicates_by_sort(&m.triangles), 4);
}

#[test]
fn component_edges_match_sorted_keys() {
    for m in defect_meshes() {
        let (id, comps) = components(&m);
        let mut edges = vec![0i64; comps.len()];
        for (a, ..) in reference_runs(&m) {
            edges[id[a as usize] as usize] += 1;
        }
        for (c, e) in comps.iter().zip(edges) {
            assert_eq!(c.euler, c.vertices as i64 - e + c.triangles as i64);
        }
    }
}

#[test]
fn bucketed_edges_span_several_chunks() {
    // More vertices and triangles than `SUM_CHUNK`, so the scan, the bucket
    // sort and the runs cross chunk boundaries.
    let mut rng = sesmesh_testkit::clusters::Rng::new(7);
    let n = 3 * SUM_CHUNK + 123;
    let positions: Vec<[f32; 3]> = (0..n).map(|_| rng.unit().map(|c| c as f32)).collect();
    let mut triangles: Vec<[u32; 3]> = (0..4 * SUM_CHUNK)
        .map(|i| {
            // Mostly local triangles, as meshes from bricks are, some long.
            let a = rng.below(n);
            let near = |rng: &mut sesmesh_testkit::clusters::Rng| {
                if i % 7 == 0 {
                    rng.below(n)
                } else {
                    (a + rng.below(16)) % n
                }
            };
            [a, near(&mut rng), near(&mut rng)].map(|v| v as u32)
        })
        .collect();
    for i in 0..1000 {
        let t = triangles[i * 37];
        triangles.push([t[1], t[2], t[0]]);
    }
    let m = Mesh {
        normals: positions.clone(),
        vertex_atom: vec![0; n],
        vertex_patch: vec![PatchKind::Convex; n],
        positions,
        triangles,
        vertex_component: Vec::new(),
    };
    let runs: Vec<EdgeUses> = Edges::new(&m)
        .par_runs(Vec::new, Vec::push)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(runs, reference_runs(&m));
    let dups = duplicates_by_sort(&m.triangles);
    assert!(dups >= 1000);
    assert_eq!(validate(&m).duplicate_triangles, dups);
}
