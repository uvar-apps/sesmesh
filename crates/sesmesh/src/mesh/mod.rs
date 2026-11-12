//! Triangle meshes: the [`Mesh`] type, isosurface extraction (Design §8),
//! measurements, connected components and validation (Design §9.1, §9.3).

pub(crate) mod extract;
mod tables;

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

fn triangle_area_volume(mesh: &Mesh, t: [u32; 3], origin: [f64; 3]) -> (f64, f64) {
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
/// component.
pub fn components(mesh: &Mesh) -> (Vec<u32>, Vec<Component>) {
    let n = mesh.positions.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let up = parent[parent[x as usize] as usize];
            parent[x as usize] = up;
            x = up;
        }
        x
    }
    for t in &mesh.triangles {
        for k in 1..3 {
            let (a, b) = (find(&mut parent, t[0]), find(&mut parent, t[k]));
            if a != b {
                // Smaller index becomes the root: deterministic.
                let (lo, hi) = (a.min(b), a.max(b));
                parent[hi as usize] = lo;
            }
        }
    }
    let mut id = vec![u32::MAX; n];
    let mut comps: Vec<Component> = Vec::new();
    for v in 0..n {
        let root = find(&mut parent, v as u32) as usize;
        if id[root] == u32::MAX {
            id[root] = comps.len() as u32;
            comps.push(Component {
                vertices: 0,
                triangles: 0,
                area: 0.0,
                volume: 0.0,
                euler: 0,
            });
        }
        id[v] = id[root];
        comps[id[v] as usize].vertices += 1;
    }
    let origin = mesh
        .positions
        .first()
        .map_or([0.0; 3], |p| p.map(f64::from));
    for t in &mesh.triangles {
        let c = &mut comps[id[t[0] as usize] as usize];
        let (a, v) = triangle_area_volume(mesh, *t, origin);
        c.triangles += 1;
        c.area += a;
        c.volume += v;
    }
    // Edges per component (each undirected edge counted once).
    let mut edges = undirected_edges(mesh);
    edges.dedup_by_key(|e| e.0);
    let mut edge_count = vec![0i64; comps.len()];
    for (key, _) in &edges {
        edge_count[id[(key >> 32) as usize] as usize] += 1;
    }
    for (c, e) in comps.iter_mut().zip(edge_count) {
        c.euler = c.vertices as i64 - e + c.triangles as i64;
    }
    (id, comps)
}

/// Sorted `(min << 32 | max, forward)` keys of all triangle edges.
fn undirected_edges(mesh: &Mesh) -> Vec<(u64, bool)> {
    let mut edges: Vec<(u64, bool)> = Vec::with_capacity(3 * mesh.triangles.len());
    for t in &mesh.triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let key = (u64::from(a.min(b)) << 32) | u64::from(a.max(b));
            edges.push((key, a < b));
        }
    }
    par::sort_unstable(&mut edges);
    edges
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
    /// Vertices with a non-finite coordinate or normal, or a normal that is
    /// not of unit length.
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
    let n = mesh.positions.len();
    let bad_vertices = mesh
        .positions
        .iter()
        .zip(&mesh.normals)
        .filter(|(p, q)| {
            let len2: f32 = q.iter().map(|c| c * c).sum();
            !(p.iter().chain(q.iter()).all(|c| c.is_finite()) && (len2 - 1.0).abs() < 1e-3)
        })
        .count()
        + n.abs_diff(mesh.normals.len());
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
        return v;
    }
    let origin = mesh
        .positions
        .first()
        .map_or([0.0; 3], |p| p.map(f64::from));
    v.degenerate_triangles = mesh
        .triangles
        .iter()
        .filter(|t| {
            t[0] == t[1]
                || t[1] == t[2]
                || t[0] == t[2]
                || triangle_area_volume(mesh, **t, origin).0 == 0.0
        })
        .count();
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
    let edges = undirected_edges(mesh);
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 {
            j += 1;
        }
        match j - i {
            1 => v.boundary_edges += 1,
            2 if edges[i].1 == edges[i + 1].1 => v.misoriented_edges += 1,
            2 => {}
            _ => v.non_manifold_edges += 1,
        }
        i = j;
    }
    v
}

#[cfg(test)]
mod tests;
