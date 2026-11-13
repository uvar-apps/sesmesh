//! Mesh post-processing (Design §9): grouping triangles by component,
//! sub-grid fragments (§9.1), cavity handling and component filters (§9.2),
//! face-averaged normals and per-atom areas (§9.6).

use super::{Component, Mesh, PatchKind, triangle_area_volume};

/// Reorders triangles so each component's triangles are contiguous, keeping
/// their relative order (deterministic), and sets
/// [`Component::first_triangle`].
pub(crate) fn group_by_component(
    mesh: &mut Mesh,
    vertex_component: &[u32],
    comps: &mut [Component],
) {
    let mut start = 0;
    for c in comps.iter_mut() {
        c.first_triangle = start;
        start += c.triangles;
    }
    let mut next: Vec<usize> = comps.iter().map(|c| c.first_triangle).collect();
    let mut out = vec![[0u32; 3]; mesh.triangles.len()];
    for t in &mesh.triangles {
        let c = vertex_component[t[0] as usize] as usize;
        out[next[c]] = *t;
        next[c] += 1;
    }
    mesh.triangles = out;
}

/// Axis-aligned bounding box of each component.
fn component_boxes(mesh: &Mesh, vertex_component: &[u32], n: usize) -> Vec<([f64; 3], [f64; 3])> {
    let mut boxes = vec![([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]); n];
    for (p, &c) in mesh.positions.iter().zip(vertex_component) {
        let b = &mut boxes[c as usize];
        for (k, &v) in p.iter().enumerate() {
            b.0[k] = b.0[k].min(f64::from(v));
            b.1[k] = b.1[k].max(f64::from(v));
        }
    }
    boxes
}

/// `true` if `point` is inside the closed component `c` (ray parity; the
/// ray direction is fixed and generic, so hits on edges are vanishingly
/// rare).
pub(crate) fn inside_component(mesh: &Mesh, c: &Component, point: [f64; 3]) -> bool {
    const DIR: [f64; 3] = [
        0.801_783_725_737_273_2,
        0.534_522_483_824_848_8,
        0.267_261_241_912_424_4,
    ];
    let tris = &mesh.triangles[c.first_triangle..c.first_triangle + c.triangles];
    let mut crossings = 0usize;
    for t in tris {
        let [a, b, d] = t.map(|i| mesh.positions[i as usize].map(f64::from));
        // Möller–Trumbore.
        let e1 = sub(b, a);
        let e2 = sub(d, a);
        let p = cross(DIR, e2);
        let det = dot(e1, p);
        if det.abs() < 1e-18 {
            continue;
        }
        let inv = 1.0 / det;
        let s = sub(point, a);
        let u = dot(s, p) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = cross(s, e1);
        let v = dot(DIR, q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if dot(e2, q) * inv > 0.0 {
            crossings += 1;
        }
    }
    crossings % 2 == 1
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Marks every component lying inside one of the `removed` cavities (one of
/// its vertices is inside the cavity's closed surface), so material inside a
/// removed cavity goes with it (Design §9.2). Requires grouped triangles.
pub(crate) fn enclosed_by(
    mesh: &Mesh,
    vertex_component: &[u32],
    comps: &[Component],
    removed: &[bool],
) -> Vec<bool> {
    let n = comps.len();
    let boxes = component_boxes(mesh, vertex_component, n);
    // A vertex of each component.
    let mut probe = vec![None::<[f64; 3]>; n];
    for (p, &c) in mesh.positions.iter().zip(vertex_component) {
        probe[c as usize].get_or_insert(p.map(f64::from));
    }
    let cavities: Vec<usize> = (0..n)
        .filter(|&c| removed[c] && comps[c].is_cavity())
        .collect();
    (0..n)
        .map(|c| {
            if removed[c] || cavities.is_empty() {
                return false;
            }
            let Some(point) = probe[c] else {
                return false;
            };
            cavities.iter().any(|&k| {
                k != c
                    && (0..3)
                        .all(|a| boxes[k].0[a] <= boxes[c].0[a] && boxes[c].1[a] <= boxes[k].1[a])
                    && inside_component(mesh, &comps[k], point)
            })
        })
        .collect()
}

/// Marks the sub-grid fragments of an SES mesh: solid (non-cavity)
/// components none of whose vertices lies on a convex patch (Design §9.1).
///
/// The solid bounded by an SES component lies outside the probe balls on
/// concave patches and is saddle-shaped on toroidal ones, so its extreme
/// points (in every direction) lie on convex atom patches: a true solid
/// component contains an atom and has convex area of the order of
/// `4π·R²`. A solid component without a convex vertex is a piece of a
/// reentrant sheet thinner than the grid spacing (two probe balls from
/// opposite sides of an atom ring almost meeting) that Marching Cubes cut
/// off from the body it belongs to. Cavities hold a whole probe ball and are
/// never fragments.
pub(crate) fn fragments(mesh: &Mesh, vertex_component: &[u32], comps: &[Component]) -> Vec<bool> {
    let mut convex = vec![false; comps.len()];
    for (&patch, &c) in mesh.vertex_patch.iter().zip(vertex_component) {
        if patch == PatchKind::Convex {
            convex[c as usize] = true;
        }
    }
    comps
        .iter()
        .zip(convex)
        .map(|(c, convex)| !convex && !c.is_cavity() && c.triangles > 0)
        .collect()
}

/// Keeps the components with `keep[c]`, dropping their unused vertices and
/// renumbering components in order. Requires grouped triangles; the result
/// is grouped too.
pub(crate) fn retain_components(
    mesh: &mut Mesh,
    vertex_component: &mut Vec<u32>,
    comps: &mut Vec<Component>,
    keep: &[bool],
) {
    if keep.iter().all(|&k| k) {
        return;
    }
    let mut new_id = vec![u32::MAX; comps.len()];
    let mut kept = Vec::new();
    for (c, comp) in comps.iter().enumerate() {
        if keep[c] {
            new_id[c] = kept.len() as u32;
            kept.push(*comp);
        }
    }
    let mut remap = vec![u32::MAX; mesh.positions.len()];
    let mut out = Mesh::default();
    let mut out_component = Vec::new();
    let has_normals = !mesh.normals.is_empty();
    for (v, &c) in vertex_component.iter().enumerate() {
        if new_id[c as usize] == u32::MAX {
            continue;
        }
        remap[v] = out.positions.len() as u32;
        out.positions.push(mesh.positions[v]);
        if has_normals {
            out.normals.push(mesh.normals[v]);
        }
        out.vertex_atom.push(mesh.vertex_atom[v]);
        out.vertex_patch.push(mesh.vertex_patch[v]);
        out_component.push(new_id[c as usize]);
    }
    let mut start = 0;
    for c in &mut kept {
        let range = c.first_triangle..c.first_triangle + c.triangles;
        out.triangles.extend(
            mesh.triangles[range]
                .iter()
                .map(|t| t.map(|i| remap[i as usize])),
        );
        c.first_triangle = start;
        start += c.triangles;
    }
    *mesh = out;
    *vertex_component = out_component;
    *comps = kept;
}

/// Replaces the normals by area-weighted averages of the incident face
/// normals (Spec FR-MESH-3 option).
pub(crate) fn face_average_normals(mesh: &mut Mesh) {
    let mut acc = vec![[0.0f64; 3]; mesh.positions.len()];
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|i| mesh.positions[i as usize].map(f64::from));
        let n = cross(sub(b, a), sub(c, a));
        for &i in t {
            for k in 0..3 {
                acc[i as usize][k] += n[k];
            }
        }
    }
    let fallback = mesh.normals.clone();
    mesh.normals = acc
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let len = dot(*n, *n).sqrt();
            if len > 0.0 {
                n.map(|c| (c / len) as f32)
            } else {
                fallback.get(i).copied().unwrap_or([0.0, 0.0, 1.0])
            }
        })
        .collect();
}

/// SES area attributed to each of `atoms` input spheres (Spec FR-OUT-7,
/// Design §9.6): each triangle's area is split equally among its three
/// vertices' nearest atoms. Sums run in triangle order (deterministic); the
/// total equals the mesh area up to rounding.
///
/// # Panics
/// If a vertex's atom index is not below `atoms`, or a triangle references a
/// missing vertex.
pub fn atom_areas(mesh: &Mesh, atoms: usize) -> Vec<f64> {
    let mut out = vec![0.0; atoms];
    let origin = mesh
        .positions
        .first()
        .map_or([0.0; 3], |p| p.map(f64::from));
    for t in &mesh.triangles {
        let (area, _) = triangle_area_volume(mesh, *t, origin);
        for &v in t {
            out[mesh.vertex_atom[v as usize] as usize] += area / 3.0;
        }
    }
    out
}

/// Taubin λ|μ smoothing (λ = 0.5, μ = −0.53, uniform umbrella Laplacian)
/// for `iterations` rounds, then exact re-projection of every vertex onto
/// the surface (`x ← x − F(x)·∇F(x)`, up to 4 steps) with analytic normals,
/// patch kinds and nearest atoms recomputed (Design §9.4, Spec FR-LOD-5).
/// Improves triangle shape while keeping vertices on the surface.
pub(crate) fn smooth_and_project(
    mesh: &mut Mesh,
    field: &crate::field::SesField,
    spheres: &[crate::geom::Sphere],
    iterations: u32,
) {
    if iterations == 0 || mesh.positions.is_empty() {
        return;
    }
    // Vertex adjacency (CSR, sorted, deduplicated).
    let n = mesh.positions.len();
    let mut pairs: Vec<u64> = Vec::with_capacity(6 * mesh.triangles.len());
    for t in &mesh.triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            pairs.push((u64::from(a) << 32) | u64::from(b));
            pairs.push((u64::from(b) << 32) | u64::from(a));
        }
    }
    crate::par::sort_unstable(&mut pairs);
    pairs.dedup();
    let mut offsets = vec![0u32; n + 1];
    for p in &pairs {
        offsets[(p >> 32) as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let neighbours: Vec<u32> = pairs.iter().map(|&p| p as u32).collect();
    drop(pairs);
    let mut x: Vec<[f64; 3]> = mesh.positions.iter().map(|p| p.map(f64::from)).collect();
    let step = |x: &[[f64; 3]], f: f64| -> Vec<[f64; 3]> {
        crate::par::map(n, |i| {
            let nb = &neighbours[offsets[i] as usize..offsets[i + 1] as usize];
            if nb.is_empty() {
                return x[i];
            }
            let mut m = [0.0; 3];
            for &j in nb {
                for k in 0..3 {
                    m[k] += x[j as usize][k];
                }
            }
            let inv = 1.0 / nb.len() as f64;
            [0, 1, 2].map(|k| x[i][k] + f * (m[k] * inv - x[i][k]))
        })
    };
    for _ in 0..iterations {
        x = step(&x, 0.5);
        x = step(&x, -0.53);
    }
    let projected = crate::par::map(n, |i| {
        let mut p = x[i];
        let mut s = field.sample(p);
        for _ in 0..4 {
            let Some(g) = s.normal else { break };
            if s.value.abs() < 1e-9 {
                break;
            }
            p = [0, 1, 2].map(|k| p[k] - s.value * g[k]);
            s = field.sample(p);
        }
        let atom = nearest_atom(&s.patch, p, spheres).unwrap_or(mesh.vertex_atom[i]);
        let patch = match s.patch {
            crate::field::Patch::Saddle { .. } => PatchKind::Saddle,
            crate::field::Patch::Concave { .. } => PatchKind::Concave,
            _ => PatchKind::Convex,
        };
        let normal = s.normal.map_or(
            mesh.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
            |g| g.map(|c| c as f32),
        );
        (p.map(|c| c as f32), normal, atom, patch)
    });
    let has_normals = !mesh.normals.is_empty();
    for (i, (p, q, atom, patch)) in projected.into_iter().enumerate() {
        mesh.positions[i] = p;
        if has_normals {
            mesh.normals[i] = q;
        }
        mesh.vertex_atom[i] = atom;
        mesh.vertex_patch[i] = patch;
    }
}

/// Nearest generating atom of a patch (`|v − c| − r`, MSMS semantics).
fn nearest_atom(
    patch: &crate::field::Patch,
    v: [f64; 3],
    spheres: &[crate::geom::Sphere],
) -> Option<u32> {
    use crate::field::Patch;
    let atoms: &[usize] = match patch {
        Patch::Convex { atom } => std::slice::from_ref(atom),
        Patch::Saddle { atoms } => atoms,
        Patch::Concave { atoms } => atoms,
        _ => return None,
    };
    atoms
        .iter()
        .map(|&a| {
            let s = &spheres[a];
            let d = (0..3)
                .map(|k| (v[k] - s.center[k]).powi(2))
                .sum::<f64>()
                .sqrt()
                - s.radius;
            (d, a)
        })
        .min_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)))
        .map(|(_, a)| a as u32)
}
