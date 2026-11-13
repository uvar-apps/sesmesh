//! Mesh decimation (Spec FR-LOD-4, Design §9.5): quadric-error-metric
//! *half-edge* collapses (Garland–Heckbert error, collapse of one endpoint
//! onto the other). No new positions are created, so every retained vertex
//! keeps its exact position on the SES, its analytic normal and its
//! attributes.
//!
//! A collapse `u → v` is applied only if it keeps the mesh a closed
//! edge-manifold (link condition), flips no triangle, creates no sliver
//! below a minimum quality and keeps every component at least a
//! tetrahedron. Collapses run in order of increasing error (ties by vertex
//! index), so the result is deterministic.

use std::cmp::Ordering;

use super::Mesh;

/// Decimation targets (Spec FR-LOD-4).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Decimation {
    /// Keep at most this fraction of the triangles (0 < ratio ≤ 1).
    pub ratio: f64,
    /// Keep at most this many triangles (combined with `ratio`: the smaller
    /// target wins).
    pub max_triangles: Option<usize>,
    /// Never collapse when the quadric error (≈ distance in Å of the kept
    /// vertex from the planes of the removed triangles) would exceed this;
    /// infinite by default (the target decides). The bound wins over the
    /// target when both are set.
    pub max_error: f64,
}

impl Decimation {
    /// Keep about `ratio` of the triangles (no error bound).
    pub fn ratio(ratio: f64) -> Self {
        Self {
            ratio,
            max_triangles: None,
            max_error: f64::INFINITY,
        }
    }

    /// Sets the triangle limit.
    #[must_use]
    pub fn with_max_triangles(mut self, triangles: usize) -> Self {
        self.max_triangles = Some(triangles);
        self
    }

    /// Sets the error bound (Å).
    #[must_use]
    pub fn with_max_error(mut self, error: f64) -> Self {
        self.max_error = error;
        self
    }

    /// Target triangle count for a mesh of `triangles`.
    pub fn target(&self, triangles: usize) -> usize {
        let by_ratio = (triangles as f64 * self.ratio).ceil() as usize;
        self.max_triangles.map_or(by_ratio, |m| m.min(by_ratio))
    }
}

/// Symmetric 4×4 quadric, upper triangle `[a², ab, ac, ad, b², bc, bd, c², cd, d²]`.
#[derive(Debug, Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: [f64; 3], d: f64) -> Self {
        let [a, b, c] = n;
        Self([
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ])
    }

    fn add(&mut self, o: &Quadric) {
        for (x, y) in self.0.iter_mut().zip(o.0) {
            *x += y;
        }
    }

    fn eval(&self, p: [f64; 3]) -> f64 {
        let q = &self.0;
        let [x, y, z] = p;
        q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9]
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Ratio of the area to the squared longest edge, normalised to 1 for an
/// equilateral triangle (0 for a degenerate one).
fn quality(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let n = cross(sub(b, a), sub(c, a));
    let area2 = dot(n, n).sqrt();
    let l = [
        dot(sub(b, a), sub(b, a)),
        dot(sub(c, b), sub(c, b)),
        dot(sub(a, c), sub(a, c)),
    ];
    let longest = l.iter().fold(0.0f64, |m, &x| m.max(x));
    if longest == 0.0 {
        0.0
    } else {
        area2 / longest * (2.0 / 3f64.sqrt())
    }
}

/// A candidate collapse `from → to` (orders as a min-heap on the error).
#[derive(Debug, Clone, Copy)]
struct Candidate {
    error: f64,
    from: u32,
    to: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Candidate {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reversed for a min-heap; ties by vertex indices (deterministic).
        o.error
            .total_cmp(&self.error)
            .then(o.from.cmp(&self.from))
            .then(o.to.cmp(&self.to))
    }
}

/// Result of [`decimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct DecimationStats {
    /// Triangles before.
    pub triangles_before: usize,
    /// Triangles after.
    pub triangles_after: usize,
    /// Collapses applied.
    pub collapses: usize,
}

/// No corner (end of a corner list).
const NONE: u32 = u32::MAX;

/// Vertex → triangle incidence as singly linked lists of triangle corners
/// (corner `c` is slot `c % 3` of triangle `c / 3`): two flat arrays, 4 bytes
/// per corner and per vertex, instead of one small vector per vertex (whose
/// allocations dominated the decimation's peak memory, issue #21 follow-up).
/// Lists may hold corners of dead triangles; readers filter on `alive_tri`.
struct Corners {
    head: Vec<u32>,
    next: Vec<u32>,
}

impl Corners {
    fn new(vertices: usize, tris: &[[u32; 3]]) -> Self {
        let mut head = vec![NONE; vertices];
        let mut next = vec![NONE; 3 * tris.len()];
        for (t, tri) in tris.iter().enumerate() {
            for (k, &v) in tri.iter().enumerate() {
                let c = 3 * t + k;
                next[c] = head[v as usize];
                head[v as usize] = c as u32;
            }
        }
        Self { head, next }
    }

    /// Corners of vertex `v` (any order).
    fn of(&self, v: u32) -> impl Iterator<Item = u32> + '_ {
        let mut c = self.head[v as usize];
        std::iter::from_fn(move || {
            (c != NONE).then(|| {
                let out = c;
                c = self.next[c as usize];
                out
            })
        })
    }

    /// Triangles of vertex `v` that are alive.
    fn alive_tris<'a>(&'a self, v: u32, alive_tri: &'a [bool]) -> impl Iterator<Item = u32> + 'a {
        self.of(v)
            .map(|c| c / 3)
            .filter(move |&t| alive_tri[t as usize])
    }

    /// Moves the corners of `u` whose triangle is alive to `v`, drops the
    /// dead corners of `v`, and empties `u`.
    fn merge(&mut self, u: u32, v: u32, alive_tri: &[bool]) {
        let mut c = self.head[u as usize];
        while c != NONE {
            let following = self.next[c as usize];
            if alive_tri[(c / 3) as usize] {
                self.next[c as usize] = self.head[v as usize];
                self.head[v as usize] = c;
            }
            c = following;
        }
        self.head[u as usize] = NONE;
        // Unlink the dead corners of `v`.
        let mut prev = NONE;
        let mut c = self.head[v as usize];
        while c != NONE {
            let following = self.next[c as usize];
            if alive_tri[(c / 3) as usize] {
                prev = c;
            } else if prev == NONE {
                self.head[v as usize] = following;
            } else {
                self.next[prev as usize] = following;
            }
            c = following;
        }
    }
}

/// Sorted neighbours of vertex `x` over the live triangles, into `out`.
fn neighbours_into(
    x: u32,
    corners: &Corners,
    tris: &[[u32; 3]],
    alive_tri: &[bool],
    out: &mut Vec<u32>,
) {
    out.clear();
    out.extend(
        corners
            .alive_tris(x, alive_tri)
            .flat_map(|t| tris[t as usize])
            .filter(|&w| w != x),
    );
    out.sort_unstable();
    out.dedup();
}

/// Minimum triangle quality a collapse may create.
const MIN_QUALITY: f64 = 0.05;
/// Minimum cosine between a triangle's normal before and after a collapse.
const MIN_NORMAL_COS: f64 = 0.2;

/// Decimates a closed, edge-manifold mesh in place (see the module docs).
/// Vertex attributes (normals, nearest atom, patch, component) follow the
/// retained vertices; unused vertices are removed.
///
/// Working memory: ≈ 90 bytes per input triangle on top of the mesh (corner
/// lists, quadrics, the candidate list of a pass), Design §11.3.
pub fn decimate(mesh: &mut Mesh, options: &Decimation) -> DecimationStats {
    let nv = mesh.positions.len();
    let before = mesh.triangles.len();
    let target = options.target(before);
    let mut stats = DecimationStats {
        triangles_before: before,
        triangles_after: before,
        collapses: 0,
    };
    if target >= before || nv == 0 {
        return stats;
    }
    // Positions in f64 (exact conversion), read in place rather than copied.
    let positions = &mesh.positions;
    let pos = |i: u32| positions[i as usize].map(f64::from);
    let mut tris: Vec<[u32; 3]> = std::mem::take(&mut mesh.triangles);
    let mut alive_tri = vec![true; tris.len()];
    let mut corners = Corners::new(nv, &tris);
    // Triangles per component (never collapse below a tetrahedron).
    let vertex_component = &mesh.vertex_component;
    let comp_of = |v: u32| vertex_component.get(v as usize).copied().unwrap_or(0);
    let ncomp = vertex_component
        .iter()
        .copied()
        .max()
        .map_or(1, |m| m as usize + 1);
    let mut comp_tris = vec![0usize; ncomp];
    for tri in &tris {
        comp_tris[comp_of(tri[0]) as usize] += 1;
    }
    // Plane quadrics (unweighted: the error approximates a distance).
    let mut quadric = vec![Quadric::default(); nv];
    for tri in &tris {
        let [a, b, c] = tri.map(pos);
        let n = cross(sub(b, a), sub(c, a));
        let len = dot(n, n).sqrt();
        if len == 0.0 {
            continue;
        }
        let n = n.map(|x| x / len);
        let q = Quadric::plane(n, -dot(n, a));
        for &v in tri {
            quadric[v as usize].add(&q);
        }
    }
    let mut alive_v = vec![true; nv];
    let limit = if options.max_error.is_finite() {
        options.max_error * options.max_error
    } else {
        f64::INFINITY
    };
    // Reused buffers (no allocation per collapse). A closed mesh has
    // 3/2 edges per triangle, so the first pass's candidate list never
    // grows beyond its initial capacity.
    let (mut nbrs, mut ut, mut shared, mut nu, mut nv_) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut candidates: Vec<Candidate> = Vec::with_capacity(before / 2 * 3 + 3);
    let mut touched = vec![false; nv];
    let mut remaining = before;
    // Passes (as in meshoptimizer): rank every live edge by the error of
    // its cheaper direction, then collapse in that order, skipping edges
    // whose neighbourhood already changed in this pass; repeat until the
    // target is met or nothing collapses.
    'passes: while remaining > target {
        candidates.clear();
        for v in 0..nv as u32 {
            if !alive_v[v as usize] {
                continue;
            }
            neighbours_into(v, &corners, &tris, &alive_tri, &mut nbrs);
            for &w in &nbrs {
                if w < v {
                    continue;
                }
                let mut q = quadric[v as usize];
                q.add(&quadric[w as usize]);
                let a = q.eval(pos(w)).max(0.0);
                let b = q.eval(pos(v)).max(0.0);
                let c = if a <= b {
                    Candidate {
                        error: a,
                        from: v,
                        to: w,
                    }
                } else {
                    Candidate {
                        error: b,
                        from: w,
                        to: v,
                    }
                };
                if c.error <= limit {
                    candidates.push(c);
                }
            }
        }
        // Ascending error (Candidate orders as a min-heap: reverse).
        crate::par::sort_unstable(&mut candidates);
        candidates.reverse();
        touched.fill(false);
        let mut collapsed = 0usize;
        for c in &candidates {
            if remaining <= target {
                break 'passes;
            }
            let (u, v) = (c.from, c.to);
            if touched[u as usize] || touched[v as usize] {
                continue;
            }
            ut.clear();
            ut.extend(corners.alive_tris(u, &alive_tri));
            // Triangles on edge (u, v): exactly two in a closed manifold.
            shared.clear();
            shared.extend(
                ut.iter()
                    .copied()
                    .filter(|&t| tris[t as usize].contains(&v)),
            );
            if shared.len() != 2 || comp_tris[comp_of(u) as usize] <= 4 {
                continue;
            }
            // Link condition: common neighbours are exactly the two
            // opposite vertices of the shared triangles.
            neighbours_into(u, &corners, &tris, &alive_tri, &mut nu);
            neighbours_into(v, &corners, &tris, &alive_tri, &mut nv_);
            let common = nu.iter().filter(|w| nv_.binary_search(w).is_ok()).count();
            if common != 2 {
                continue;
            }
            // No flip, no sliver.
            let ok = ut.iter().filter(|t| !shared.contains(t)).all(|&t| {
                let tri = tris[t as usize];
                let old = tri.map(pos);
                let new = tri.map(|i| pos(if i == u { v } else { i }));
                let n0 = cross(sub(old[1], old[0]), sub(old[2], old[0]));
                let n1 = cross(sub(new[1], new[0]), sub(new[2], new[0]));
                let (l0, l1) = (dot(n0, n0).sqrt(), dot(n1, n1).sqrt());
                l0 > 0.0
                    && l1 > 0.0
                    && dot(n0, n1) / (l0 * l1) >= MIN_NORMAL_COS
                    && quality(new[0], new[1], new[2]) >= MIN_QUALITY
            });
            if !ok {
                continue;
            }
            // Apply u → v.
            for &t in &shared {
                alive_tri[t as usize] = false;
            }
            for &t in &ut {
                if alive_tri[t as usize] {
                    for x in &mut tris[t as usize] {
                        if *x == u {
                            *x = v;
                        }
                    }
                }
            }
            corners.merge(u, v, &alive_tri);
            remaining -= 2;
            comp_tris[comp_of(u) as usize] -= 2;
            alive_v[u as usize] = false;
            let qu = quadric[u as usize];
            quadric[v as usize].add(&qu);
            stats.collapses += 1;
            collapsed += 1;
            // Lock the changed neighbourhood until the next pass.
            touched[u as usize] = true;
            touched[v as usize] = true;
            for &w in &nu {
                touched[w as usize] = true;
            }
            for &w in &nv_ {
                touched[w as usize] = true;
            }
        }
        if collapsed == 0 {
            break;
        }
    }
    // Release the working set before building the compacted mesh.
    drop((candidates, corners, quadric, alive_v, touched));
    // Compact.
    let mut remap = vec![u32::MAX; nv];
    let mut out = Mesh::default();
    let used: Vec<bool> = {
        let mut u = vec![false; nv];
        for (t, tri) in tris.iter().enumerate() {
            if alive_tri[t] {
                for &v in tri {
                    u[v as usize] = true;
                }
            }
        }
        u
    };
    let has_normals = !mesh.normals.is_empty();
    for v in 0..nv {
        if !used[v] {
            continue;
        }
        remap[v] = out.positions.len() as u32;
        out.positions.push(mesh.positions[v]);
        if has_normals {
            out.normals.push(mesh.normals[v]);
        }
        out.vertex_atom.push(mesh.vertex_atom[v]);
        out.vertex_patch.push(mesh.vertex_patch[v]);
        if let Some(&c) = mesh.vertex_component.get(v) {
            out.vertex_component.push(c);
        }
    }
    out.triangles = tris
        .iter()
        .zip(&alive_tri)
        .filter(|(_, a)| **a)
        .map(|(t, _)| t.map(|v| remap[v as usize]))
        .collect();
    stats.triangles_after = out.triangles.len();
    *mesh = out;
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Sphere, SurfaceParams, compute_surface, mesh};

    #[test]
    fn halves_a_sphere_mesh_and_keeps_it_valid() {
        let out = compute_surface(
            &[Sphere::new([0.0; 3], 3.0)],
            &SurfaceParams::vdw().with_spacing(0.2),
        )
        .unwrap();
        let mut m = out.mesh.clone();
        let stats = decimate(&mut m, &Decimation::ratio(0.5).with_max_error(0.5));
        assert!(mesh::validate(&m).is_ok(), "{:?}", mesh::validate(&m));
        assert!(
            stats.triangles_after <= stats.triangles_before / 2 + 2,
            "{stats:?}"
        );
        // Retained vertices are original ones (still on the sphere).
        for p in &m.positions {
            let r = p.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>().sqrt();
            assert!((r - 3.0).abs() < 1e-4, "{r}");
        }
        let (_, comps) = mesh::components(&m);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].euler, 2);
    }

    #[test]
    fn error_bound_stops_collapses() {
        let out = compute_surface(
            &[Sphere::new([0.0; 3], 3.0)],
            &SurfaceParams::vdw().with_spacing(0.3),
        )
        .unwrap();
        let mut m = out.mesh.clone();
        let stats = decimate(&mut m, &Decimation::ratio(0.01).with_max_error(1e-6));
        assert!(stats.collapses < stats.triangles_before / 20, "{stats:?}");
        assert!(mesh::validate(&m).is_ok());
    }

    #[test]
    fn targets() {
        assert_eq!(Decimation::ratio(0.5).target(100), 50);
        assert_eq!(
            Decimation::ratio(0.5).with_max_triangles(10).target(100),
            10
        );
        let mut m = Mesh::default();
        assert_eq!(decimate(&mut m, &Decimation::ratio(0.5)).collapses, 0);
    }
}
