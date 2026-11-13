// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Triangle meshes: the [`Mesh`] type, isosurface extraction (Design §8),
//! measurements, connected components and validation (Design §9.1, §9.3).

mod decimate;
pub(crate) mod extract;
pub(crate) mod post;
mod tables;

pub use decimate::{Decimation, DecimationStats, decimate};
pub use post::atom_areas;

use std::sync::atomic::{AtomicU32, Ordering};

use crate::par;

/// Which SES patch carries a vertex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PatchKind {
    /// On an atom sphere (also every SAS and vdW vertex).
    #[default]
    Convex = 0,
    /// On a toroidal patch between two atoms.
    Saddle = 1,
    /// On a probe sphere touching three atoms.
    Concave = 2,
}

impl PatchKind {
    /// Lower-case name (`convex`, `saddle`, `concave`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Convex => "convex",
            Self::Saddle => "saddle",
            Self::Concave => "concave",
        }
    }
}

/// An indexed triangle mesh with per-vertex normals and attributes.
///
/// Coordinates are in the input frame (Å). Triangles are counter-clockwise
/// seen from outside (the solvent side); normals are unit vectors pointing to
/// the solvent.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Mesh {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Unit vertex normals (analytic, Design §8.2).
    pub normals: Vec<[f32; 3]>,
    /// Vertex indices of each triangle.
    pub triangles: Vec<[u32; 3]>,
    /// Input index of the sphere (atom) nearest to each vertex.
    pub vertex_atom: Vec<u32>,
    /// Patch carrying each vertex.
    pub vertex_patch: Vec<PatchKind>,
    /// Component of each vertex (index into
    /// [`SurfaceOutput::components`](crate::SurfaceOutput::components);
    /// empty for meshes not made by `compute_surface`).
    pub vertex_component: Vec<u32>,
}

impl Mesh {
    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// `true` if the mesh has no triangles.
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

/// Size of the fixed chunks used for deterministic parallel sums.
const SUM_CHUNK: usize = 1 << 14;

/// Total area (Å²) and signed volume (Å³) of a mesh (positive for outward
/// orientation). Sums run in a fixed order, so results do not depend on the
/// thread count.
pub fn area_volume(mesh: &Mesh) -> (f64, f64) {
    let origin = mesh
        .positions
        .first()
        .map_or([0.0; 3], |p| p.map(f64::from));
    let tris = &mesh.triangles;
    let parts = par::map(tris.len().div_ceil(SUM_CHUNK), |c| {
        let end = ((c + 1) * SUM_CHUNK).min(tris.len());
        tris[c * SUM_CHUNK..end]
            .iter()
            .fold((0.0, 0.0), |(a, v), t| {
                let (ta, tv) = triangle_area_volume(mesh, *t, origin);
                (a + ta, v + tv)
            })
    });
    parts
        .iter()
        .fold((0.0, 0.0), |(a, v), &(pa, pv)| (a + pa, v + pv))
}

pub(crate) fn triangle_area_volume(mesh: &Mesh, t: [u32; 3], origin: [f64; 3]) -> (f64, f64) {
    let p = t.map(|i| {
        let q = mesh.positions[i as usize];
        [0, 1, 2].map(|k| f64::from(q[k]) - origin[k])
    });
    let e1 = sub(p[1], p[0]);
    let e2 = sub(p[2], p[0]);
    let area = 0.5 * norm(cross(e1, e2));
    let volume = dot(p[0], cross(p[1], p[2])) / 6.0;
    (area, volume)
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
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// One connected component of a mesh (Design §9.1).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Component {
    /// Number of vertices.
    pub vertices: usize,
    /// Number of triangles.
    pub triangles: usize,
    /// Area, Å².
    pub area: f64,
    /// Signed volume, Å³: positive for an outer surface, negative for an
    /// internal cavity (its triangles face into the cavity, Design §2.4).
    pub volume: f64,
    /// Euler characteristic `V − E + F` (2 for a sphere-like component).
    pub euler: i64,
    /// First triangle of the component when triangles are grouped by
    /// component (as in [`compute_surface`](crate::compute_surface)
    /// output: component `c` owns triangles
    /// `first_triangle..first_triangle + triangles`); 0 otherwise.
    pub first_triangle: usize,
}

impl Component {
    /// `true` for an internal cavity (negative signed volume).
    pub fn is_cavity(&self) -> bool {
        self.volume < 0.0
    }

    /// Genus `(2 − χ) / 2` of a closed component.
    pub fn genus(&self) -> i64 {
        (2 - self.euler) / 2
    }
}

/// Per-vertex component id (`0..n`, ordered by smallest vertex index) and the
/// components' properties. Vertices used by no triangle get their own
/// component. Parallel and deterministic.
///
/// # Panics
/// If a triangle references a vertex that does not exist (see
/// [`validate`]).
pub fn components(mesh: &Mesh) -> (Vec<u32>, Vec<Component>) {
    components_with(mesh, &Edges::new(mesh))
}

pub(crate) fn components_with(mesh: &Mesh, edges: &Edges) -> (Vec<u32>, Vec<Component>) {
    let n = mesh.positions.len();
    let tris = &mesh.triangles;
    // Lock-free union–find: links always go from the larger root to the
    // smaller, so parents only decrease (no cycles, stale reads are still
    // ancestors) and every root is its component's smallest vertex; the
    // partition, hence the result, does not depend on scheduling.
    let parent: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();
    let find = |mut x: u32| loop {
        let p = parent[x as usize].load(Ordering::Relaxed);
        if p == x {
            return x;
        }
        let gp = parent[p as usize].load(Ordering::Relaxed);
        if gp != p {
            let _ =
                parent[x as usize].compare_exchange(p, gp, Ordering::Relaxed, Ordering::Relaxed);
        }
        x = gp;
    };
    let union = |a: u32, b: u32| loop {
        let (ra, rb) = (find(a), find(b));
        if ra == rb {
            return;
        }
        let (lo, hi) = (ra.min(rb), ra.max(rb));
        if parent[hi as usize]
            .compare_exchange(hi, lo, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
    };
    par::map(tris.len().div_ceil(SUM_CHUNK), |c| {
        let end = ((c + 1) * SUM_CHUNK).min(tris.len());
        for t in &tris[c * SUM_CHUNK..end] {
            union(t[0], t[1]);
            union(t[0], t[2]);
        }
    });
    let roots = par::map(n, |v| find(v as u32));
    let mut id = vec![0u32; n];
    let mut comps: Vec<Component> = Vec::new();
    for v in 0..n {
        let r = roots[v] as usize;
        if r == v {
            id[v] = comps.len() as u32;
            comps.push(Component {
                vertices: 0,
                triangles: 0,
                area: 0.0,
                volume: 0.0,
                euler: 0,
                first_triangle: 0,
            });
        } else {
            id[v] = id[r];
        }
        comps[id[v] as usize].vertices += 1;
    }
    drop(roots);
    // Per-component sums: per fixed chunk, merged in chunk order.
    let origin = mesh
        .positions
        .first()
        .map_or([0.0; 3], |p| p.map(f64::from));
    let parts = par::map(tris.len().div_ceil(SUM_CHUNK), |c| {
        let end = ((c + 1) * SUM_CHUNK).min(tris.len());
        let mut acc: Vec<(u32, usize, f64, f64)> = Vec::new();
        for t in &tris[c * SUM_CHUNK..end] {
            let comp = id[t[0] as usize];
            let (a, v) = triangle_area_volume(mesh, *t, origin);
            match acc.iter_mut().find(|e| e.0 == comp) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += a;
                    e.3 += v;
                }
                None => acc.push((comp, 1, a, v)),
            }
        }
        acc
    });
    for part in parts {
        for (comp, t, a, v) in part {
            let c = &mut comps[comp as usize];
            c.triangles += t;
            c.area += a;
            c.volume += v;
        }
    }
    // Undirected edges per component, each counted once.
    let edge_parts = edges.par_runs(Vec::<(u32, i64)>::new, |acc, (a, _, _, _)| {
        let comp = id[a as usize];
        match acc.iter_mut().find(|e| e.0 == comp) {
            Some(e) => e.1 += 1,
            None => acc.push((comp, 1)),
        }
    });
    let mut edge_count = vec![0i64; comps.len()];
    for part in edge_parts {
        for (comp, e) in part {
            edge_count[comp as usize] += e;
        }
    }
    for (c, e) in comps.iter_mut().zip(edge_count) {
        c.euler = c.vertices as i64 - e + c.triangles as i64;
    }
    (id, comps)
}

/// The edges of all triangles as sorted keys `(min, max)` with the
/// traversal direction, so the uses of each undirected edge are adjacent.
pub(crate) enum Edges {
    /// `min << 33 | max << 1 | forward` (fewer than 2³¹ vertices).
    Packed(Vec<u64>),
    /// `(min << 32 | max, forward)`.
    Wide(Vec<(u64, bool)>),
}

/// Uses of one undirected edge: `(min, max, forward uses, backward uses)`.
type EdgeUses = (u32, u32, usize, usize);

impl Edges {
    pub(crate) fn new(mesh: &Mesh) -> Self {
        let tris = &mesh.triangles;
        let each = |f: &mut dyn FnMut(u32, u32)| {
            for t in tris {
                for k in 0..3 {
                    f(t[k], t[(k + 1) % 3]);
                }
            }
        };
        if mesh.positions.len() < (1 << 31) {
            let mut keys = Vec::with_capacity(3 * tris.len());
            each(&mut |a, b| {
                keys.push(
                    (u64::from(a.min(b)) << 33) | (u64::from(a.max(b)) << 1) | u64::from(a < b),
                );
            });
            par::sort_unstable(&mut keys);
            Self::Packed(keys)
        } else {
            let mut keys = Vec::with_capacity(3 * tris.len());
            each(&mut |a, b| {
                keys.push(((u64::from(a.min(b)) << 32) | u64::from(a.max(b)), a < b));
            });
            par::sort_unstable(&mut keys);
            Self::Wide(keys)
        }
    }

    /// Calls `f(acc, uses)` for every undirected edge, in parallel fixed
    /// chunks; returns the per-chunk accumulators in order.
    fn par_runs<A, I, F>(&self, init: I, f: F) -> Vec<A>
    where
        A: Send,
        I: Fn() -> A + Sync + Send,
        F: Fn(&mut A, EdgeUses) + Sync + Send,
    {
        match self {
            Self::Packed(k) => runs(
                k,
                |x| x >> 1,
                |x| x & 1 == 1,
                |g| ((g >> 32) as u32, g as u32),
                init,
                f,
            ),
            Self::Wide(k) => runs(
                k,
                |x| x.0,
                |x| x.1,
                |g| ((g >> 32) as u32, g as u32),
                init,
                f,
            ),
        }
    }
}

/// Groups equal `group(x)` runs of the sorted `keys` in parallel chunks
/// (a run belongs to the chunk where it starts).
fn runs<T, A, I, F>(
    keys: &[T],
    group: impl Fn(&T) -> u64 + Sync + Send,
    forward: impl Fn(&T) -> bool + Sync + Send,
    split: impl Fn(u64) -> (u32, u32) + Sync + Send,
    init: I,
    f: F,
) -> Vec<A>
where
    T: Sync,
    A: Send,
    I: Fn() -> A + Sync + Send,
    F: Fn(&mut A, EdgeUses) + Sync + Send,
{
    par::map(keys.len().div_ceil(SUM_CHUNK), |c| {
        let mut acc = init();
        let mut i = c * SUM_CHUNK;
        let end = ((c + 1) * SUM_CHUNK).min(keys.len());
        while i > 0 && i < end && group(&keys[i]) == group(&keys[i - 1]) {
            i += 1;
        }
        while i < end {
            let g = group(&keys[i]);
            let (mut fwd, mut bwd) = (0, 0);
            while i < keys.len() && group(&keys[i]) == g {
                if forward(&keys[i]) {
                    fwd += 1;
                } else {
                    bwd += 1;
                }
                i += 1;
            }
            let (a, b) = split(g);
            f(&mut acc, (a, b, fwd, bwd));
        }
        acc
    })
}

/// Result of [`validate`]: counts of each defect (all zero for a closed,
/// consistently oriented, edge-manifold mesh).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Validation {
    /// Edges used by a single triangle.
    pub boundary_edges: usize,
    /// Edges used by more than two triangles.
    pub non_manifold_edges: usize,
    /// Edges whose two triangles traverse them in the same direction.
    pub misoriented_edges: usize,
    /// Triangles with a repeated vertex or zero area.
    pub degenerate_triangles: usize,
    /// Triangles with the same vertex set as an earlier one.
    pub duplicate_triangles: usize,
    /// Triangles referencing a vertex that does not exist.
    pub invalid_indices: usize,
    /// Vertices with a non-finite coordinate, or (when the mesh has normals)
    /// a non-finite or non-unit normal or a missing one.
    pub bad_vertices: usize,
}

impl Validation {
    /// `true` if no defect was found.
    pub fn is_ok(&self) -> bool {
        *self == Self::default()
    }
}

/// Checks a mesh for the invariants of Spec FR-MESH-2 and FR-MESH-5
/// (Design §9.3).
pub fn validate(mesh: &Mesh) -> Validation {
    validate_with(mesh, None).0
}

/// [`validate`], returning the sorted directed edges for reuse (`None` when
/// indices are invalid).
pub(crate) fn validate_with(mesh: &Mesh, edges: Option<Edges>) -> (Validation, Option<Edges>) {
    let n = mesh.positions.len();
    // Normals are optional (empty); when present there is one unit normal
    // per vertex.
    let bad_positions = mesh
        .positions
        .iter()
        .filter(|p| !p.iter().all(|c| c.is_finite()))
        .count();
    let bad_normals = if mesh.normals.is_empty() {
        0
    } else {
        mesh.normals
            .iter()
            .filter(|q| {
                let len2: f32 = q.iter().map(|c| c * c).sum();
                !(q.iter().all(|c| c.is_finite()) && (len2 - 1.0).abs() < 1e-3)
            })
            .count()
            + n.abs_diff(mesh.normals.len())
    };
    let bad_vertices = bad_positions + bad_normals;
    let invalid_indices = mesh
        .triangles
        .iter()
        .filter(|t| t.iter().any(|&i| i as usize >= n))
        .count();
    let mut v = Validation {
        bad_vertices,
        invalid_indices,
        ..Validation::default()
    };
    if v.invalid_indices > 0 {
        return (v, None);
    }
    let origin = mesh
        .positions
        .first()
        .map_or([0.0; 3], |p| p.map(f64::from));
    let tris = &mesh.triangles;
    v.degenerate_triangles = par::map(tris.len().div_ceil(SUM_CHUNK), |c| {
        let end = ((c + 1) * SUM_CHUNK).min(tris.len());
        tris[c * SUM_CHUNK..end]
            .iter()
            .filter(|t| {
                t[0] == t[1]
                    || t[1] == t[2]
                    || t[0] == t[2]
                    || triangle_area_volume(mesh, **t, origin).0 == 0.0
            })
            .count()
    })
    .iter()
    .sum();
    let mut sorted: Vec<[u32; 3]> = mesh
        .triangles
        .iter()
        .map(|t| {
            let mut s = *t;
            s.sort_unstable();
            s
        })
        .collect();
    par::sort_unstable(&mut sorted);
    v.duplicate_triangles = sorted.windows(2).filter(|w| w[0] == w[1]).count();
    let edges = edges.unwrap_or_else(|| Edges::new(mesh));
    for [boundary, non_manifold, misoriented] in edges.par_runs(
        || [0usize; 3],
        |acc, (a, b, fwd, bwd)| {
            if a == b || (fwd, bwd) == (1, 1) {
                return;
            }
            match fwd + bwd {
                1 => acc[0] += 1,
                2 => acc[2] += 1,
                _ => acc[1] += 1,
            }
        },
    ) {
        v.boundary_edges += boundary;
        v.non_manifold_edges += non_manifold;
        v.misoriented_edges += misoriented;
    }
    (v, Some(edges))
}

#[cfg(test)]
mod tests;
