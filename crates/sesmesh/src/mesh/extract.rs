// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-brick Marching Cubes with exact edge refinement (Design §8.2) and the
//! global assembly of brick meshes (Design §8.3).
//!
//! Edges are keyed by their lower sample and axis. A brick owns the edges
//! whose lower sample lies in its half-open cell range `[0, b)³`, creates one
//! vertex per owned crossing edge (in ascending local edge id: deterministic)
//! and emits triangles as references `(neighbour code, local edge id)`; the
//! code's bits select the +x/+y/+z neighbour owning the edge. A neighbour
//! owning a crossing edge is always a surface brick, because sample signs are
//! exact and uniform bricks have no crossing edge, so references resolve.

use rustc_hash::FxHashMap;

use super::tables::{self, corner_offset, edge_axis_lower};
use super::{Mesh, PatchKind};
use crate::error::{Error, Result};
use crate::field::{Engine, Feature, LocalCandidates, NearEval, SurfaceKind, pack};
use crate::geom::Vec3;
use crate::par;
use crate::surface::RefineMode;

/// Field evaluations per edge (at most).
const MAX_ITER: usize = 8;
/// Stop refining when `|F| ≤ TOL` (Å).
const TOL: f64 = 1e-5;
/// Edge parameter clamp: vertices of edges meeting at a sample never
/// coincide, so no triangle has zero area (Spec FR-MESH-5). Raised where
/// needed so the separation also survives `f32` storage (see
/// [`edge_clamp`]).
const T_MIN: f64 = 1e-4;

/// Edge parameter clamp for this engine: at least [`T_MIN`], and at least
/// four `f32` ulps of the largest coordinate (in Å), so vertices that edges
/// meeting at a sample place next to it stay distinct after rounding to the
/// stored `f32` positions (input frame). The vertex then moves by at most
/// that distance: 2.4·10⁻⁴ Å for coordinates up to 1 000 Å.
pub(crate) fn edge_clamp(engine: &Engine) -> f64 {
    let m = engine.coord_max.clamp(1.0, f64::from(f32::MAX) / 4.0) as f32;
    let ulp = f64::from(f32::from_bits(m.to_bits() + 1) - m);
    (4.0 * ulp / engine.lattice.h).clamp(T_MIN, 0.25)
}
/// Bit position of the neighbour code in a triangle reference.
const CODE_SHIFT: u32 = 28;
const ID_MASK: u32 = (1 << CODE_SHIFT) - 1;

/// Brick meshes before assembly (one brick, or consecutive bricks
/// concatenated, see [`ChunkMesh`]).
#[derive(Debug, Default)]
pub(crate) struct BrickMesh {
    /// Owned crossing edges (local ids), ascending; one vertex each.
    edges: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    atoms: Vec<u32>,
    patches: Vec<PatchKind>,
    /// Triangles as packed edge references.
    tris: Vec<[u32; 3]>,
    /// Exact field evaluations spent on vertices.
    pub(crate) evals: u64,
}

/// The meshes of consecutive bricks in flat arrays (one allocation per
/// array and chunk instead of per brick: per-brick vectors kept the
/// assembly's peak memory high, issue #21).
#[derive(Debug, Default)]
pub(crate) struct ChunkMesh {
    mesh: BrickMesh,
    /// Per brick: end of its vertices (and owned edges) in `mesh`.
    vertex_end: Vec<u32>,
    /// Per brick: end of its triangles in `mesh`.
    tri_end: Vec<u32>,
}

impl ChunkMesh {
    /// Meshes brick `key` (see [`mesh_brick`]) and appends it.
    pub(crate) fn push(
        &mut self,
        engine: &Engine,
        key: [i32; 3],
        values: &[f32],
        refine: RefineMode,
    ) {
        mesh_brick(engine, key, values, refine, &mut self.mesh);
        let end = |n: usize| u32::try_from(n).expect("fewer than 2^32 vertices per chunk");
        self.vertex_end.push(end(self.mesh.edges.len()));
        self.tri_end.push(end(self.mesh.tris.len()));
    }

    /// Exact field evaluations spent on vertices.
    pub(crate) fn evals(&self) -> u64 {
        self.mesh.evals
    }

    /// Range of brick `l` (index in the chunk) in `ends`.
    fn range(ends: &[u32], l: usize) -> std::ops::Range<usize> {
        let start = if l == 0 { 0 } else { ends[l - 1] as usize };
        start..ends[l] as usize
    }

    /// Owned edges of brick `l`, ascending.
    fn edges(&self, l: usize) -> &[u32] {
        &self.mesh.edges[Self::range(&self.vertex_end, l)]
    }
}

/// Meshes the cells of brick `key` from its samples (Engine layout: `(b+1)³`,
/// x fastest) and appends the result to `out` (edge ids and triangle
/// references are local to the brick). Only signs are used for case
/// selection; crossing edges have exact sample values at both ends (Engine
/// docs), refined further with the exact `f64` field.
fn mesh_brick(
    engine: &Engine,
    key: [i32; 3],
    values: &[f32],
    refine: RefineMode,
    out: &mut BrickMesh,
) {
    let lat = &engine.lattice;
    let b = lat.b as usize;
    let s = b + 1;
    let stride = [1, s, s * s];
    let origin = lat.brick_origin(key);

    // Vertex refinement evaluates the field against candidate lists
    // gathered once per sub-block (the 2×2×2 split of the sampling engine),
    // instead of a global cell-grid search per evaluation. A crossing edge
    // lies in the closed sub-block of its lower sample, hence in that
    // sub-block's circumscribed ball.
    let sub = if b.is_multiple_of(2) && b >= 4 {
        b / 2
    } else {
        b
    };
    let nsub = b / sub;
    let h = lat.h;
    let base = lat.point(origin);
    let rho = 0.5 * 3f64.sqrt() * sub as f64 * h * (1.0 + 1e-9) + 1e-9;
    let mut locals: [Option<LocalCandidates>; 8] = Default::default();
    let t_min = edge_clamp(engine);

    for z in 0..b {
        for y in 0..b {
            for x in 0..b {
                let i = (z * s + y) * s + x;
                let va = values[i];
                for (axis, &st) in stride.iter().enumerate() {
                    let vb = values[i + st];
                    if (va < 0.0) == (vb < 0.0) {
                        continue;
                    }
                    let ia = [
                        origin[0] + x as i64,
                        origin[1] + y as i64,
                        origin[2] + z as i64,
                    ];
                    let mut ib = ia;
                    ib[axis] += 1;
                    let (pa, pb) = (lat.point(ia), lat.point(ib));
                    let sb = [x, y, z].map(|c| (c / sub).min(nsub - 1));
                    let local = locals[sb[0] + 2 * sb[1] + 4 * sb[2]].get_or_insert_with(|| {
                        let center = base
                            + Vec3::new(
                                (sb[0] * sub) as f64 + 0.5 * sub as f64,
                                (sb[1] * sub) as f64 + 0.5 * sub as f64,
                                (sb[2] * sub) as f64 + 0.5 * sub as f64,
                            ) * h;
                        let mut l = LocalCandidates::default();
                        engine.field.gather_local(center, rho, engine.band, &mut l);
                        l
                    });
                    let (fa, fb) = (f64::from(va), f64::from(vb));
                    let edge = Edge {
                        pa,
                        pb,
                        fa,
                        fb,
                        t_min,
                    };
                    vertex(engine, &edge, refine, local, out);
                    out.edges.push((((z * b + y) * b + x) * 3 + axis) as u32);
                }
            }
        }
    }
    // Cells can cross only on edges owned by "+" neighbours, so the cell
    // pass runs even when this brick owns no vertex.
    let corner_step: [usize; 8] = std::array::from_fn(|c| {
        let o = corner_offset(c);
        o[0] + o[1] * s + o[2] * s * s
    });
    for z in 0..b {
        for y in 0..b {
            for x in 0..b {
                let i0 = (z * s + y) * s + x;
                let mut case = 0;
                for (c, &st) in corner_step.iter().enumerate() {
                    if values[i0 + st] < 0.0 {
                        case |= 1 << c;
                    }
                }
                if case == 0 || case == 255 {
                    continue;
                }
                for t in tables::triangles(case) {
                    out.tris.push(t.map(|e| {
                        let (axis, lo) = edge_axis_lower(e as usize);
                        let o = corner_offset(lo);
                        let l = [x + o[0], y + o[1], z + o[2]];
                        let mut code = 0u32;
                        let mut lc = [0usize; 3];
                        for a in 0..3 {
                            if l[a] == b {
                                code |= 1 << a;
                            } else {
                                lc[a] = l[a];
                            }
                        }
                        (code << CODE_SHIFT) | (((lc[2] * b + lc[1]) * b + lc[0]) * 3 + axis) as u32
                    }));
                }
            }
        }
    }
}

/// Number of triangles [`mesh_brick`] emits for a brick of `b`³ cells with
/// samples `values` (same layout): the cell pass without vertices. Used to
/// estimate the mesh size before meshing (memory guard, Design §8.6).
pub(crate) fn brick_triangles(b: usize, values: &[f32]) -> usize {
    let s = b + 1;
    let corner_step: [usize; 8] = std::array::from_fn(|c| {
        let o = corner_offset(c);
        o[0] + o[1] * s + o[2] * s * s
    });
    let mut n = 0;
    for z in 0..b {
        for y in 0..b {
            for x in 0..b {
                let i0 = (z * s + y) * s + x;
                let mut case = 0;
                for (c, &st) in corner_step.iter().enumerate() {
                    if values[i0 + st] < 0.0 {
                        case |= 1 << c;
                    }
                }
                n += tables::triangles(case).len();
            }
        }
    }
    n
}

/// A crossing edge `pa → pb` with sampled values of opposite inside/outside
/// state and the parameter clamp.
struct Edge {
    pa: Vec3,
    pb: Vec3,
    fa: f64,
    fb: f64,
    t_min: f64,
}

/// Places the vertex of a crossing edge and computes its attributes.
fn vertex(
    engine: &Engine,
    edge: &Edge,
    refine: RefineMode,
    local: &LocalCandidates,
    out: &mut BrickMesh,
) {
    let &Edge {
        pa,
        pb,
        fa,
        fb,
        t_min,
    } = edge;
    let field = &engine.field;
    let band = engine.band;
    let d = pb - pa;
    let mut t = (fa / (fa - fb)).clamp(0.0, 1.0);
    let mut last: Option<NearEval> = None;
    if refine == RefineMode::Exact {
        // Safeguarded Newton: the closest point gives the exact gradient
        // (|∇F| = 1), so one step from the linear guess usually lands within
        // TOL; steps leaving the bracket fall back to Illinois (modified
        // regula falsi). The bracket [t0, t1] always holds the sign change.
        let (mut t0, mut g0, mut t1, mut g1) = (0.0, fa, 1.0, fb);
        let mut side = 0i8;
        for _ in 0..MAX_ITER {
            let e = field.eval_local(pa + d * t, band, local);
            out.evals += 1;
            last = Some(e);
            let g = e.value;
            if g.abs() <= TOL {
                break;
            }
            if (g < 0.0) == (g0 < 0.0) {
                t0 = t;
                g0 = g;
                if side == -1 {
                    g1 *= 0.5;
                }
                side = -1;
            } else {
                t1 = t;
                g1 = g;
                if side == 1 {
                    g0 *= 0.5;
                }
                side = 1;
            }
            let slope = gradient(engine, pa + d * t, &e).map_or(0.0, |n| n.dot(d));
            let newton = t - g / slope;
            t = if slope.abs() > 1e-3 * d.norm() && newton > t0 && newton < t1 {
                newton
            } else {
                let s = (t0 * g1 - t1 * g0) / (g1 - g0);
                if s.is_finite() && s > t0 && s < t1 {
                    s
                } else {
                    0.5 * (t0 + t1)
                }
            };
        }
    }
    let v = pa + d * t.clamp(t_min, 1.0 - t_min);
    let e = last.unwrap_or_else(|| {
        out.evals += 1;
        field.eval_local(v, band, local)
    });
    // Fallback normal: along the edge, from the inside end to the outside end.
    let outward = if fa < 0.0 { d } else { -d }.normalized();
    let (normal, patch, atom) = attributes(engine, v, &e, outward);
    out.positions.push(v.to_array().map(|c| c as f32));
    out.normals.push(normal.to_array().map(|c| c as f32));
    out.atoms.push(atom);
    out.patches.push(patch);
}

/// Exact gradient of the field at `x` from its evaluation (`None` in the
/// solvent or beyond the cutoff).
fn gradient(engine: &Engine, x: Vec3, e: &NearEval) -> Option<Vec3> {
    let a = &engine.field.arr;
    let w = match (engine.field.kind, e.feature) {
        (_, Feature::Solvent | Feature::Beyond) => return None,
        (SurfaceKind::Ses, _) => e.closest - x,
        (_, Feature::Sphere(i)) => x - a.centers[i as usize],
        _ => return None,
    };
    let len = w.norm();
    (len > 1e-12).then(|| w / len)
}

/// Normal, patch kind and nearest atom (input index) of a vertex `v` from the
/// evaluation of a point next to it (Design §8.2 step 4).
fn attributes(engine: &Engine, v: Vec3, e: &NearEval, outward: Vec3) -> (Vec3, PatchKind, u32) {
    let field = &engine.field;
    let a = &field.arr;
    let unit = |w: Vec3| {
        let len = w.norm();
        if len > 1e-9 { w / len } else { outward }
    };
    let nearest = |atoms: &[u32]| {
        let mut best = (f64::INFINITY, atoms[0]);
        for &k in atoms {
            let d = v.dist(a.centers[k as usize]) - a.radii[k as usize];
            if d < best.0 {
                best = (d, k);
            }
        }
        a.orig[best.1 as usize]
    };
    match (field.kind, e.feature) {
        (SurfaceKind::Ses, Feature::Sphere(i)) => {
            (unit(e.closest - v), PatchKind::Convex, a.orig[i as usize])
        }
        (SurfaceKind::Ses, Feature::Circle(c)) => {
            let c = &a.circles[c as usize];
            (unit(e.closest - v), PatchKind::Saddle, nearest(&[c.i, c.j]))
        }
        (SurfaceKind::Ses, Feature::Vertex(k)) => (
            unit(e.closest - v),
            PatchKind::Concave,
            nearest(&a.vertices[k as usize].atoms),
        ),
        (_, Feature::Sphere(i)) => (
            unit(v - a.centers[i as usize]),
            PatchKind::Convex,
            a.orig[i as usize],
        ),
        _ => {
            let (_, i) = field.sphere_eval(v, field.probe + engine.band + 8.0);
            (
                outward,
                PatchKind::Convex,
                i.map_or(0, |i| a.orig[i as usize]),
            )
        }
    }
}

/// Concatenates brick meshes (chunks of `chunk` bricks, in brick order) and
/// resolves triangle references to global vertex indices. Each chunk is
/// freed once copied, so the parts and the result do not coexist in full.
///
/// # Errors
/// [`Error::ResourceLimit`] beyond `u32` vertex indices; [`Error::Internal`]
/// for an unresolvable reference (a bug).
pub(crate) fn assemble(bricks: &[[i32; 3]], chunk: usize, parts: Vec<ChunkMesh>) -> Result<Mesh> {
    debug_assert_eq!(parts.len(), bricks.len().div_ceil(chunk.max(1)));
    // Global index of the first vertex of every brick.
    let mut base = Vec::with_capacity(bricks.len());
    let mut total = 0usize;
    for p in &parts {
        base.extend(
            (0..p.vertex_end.len()).map(|l| total + ChunkMesh::range(&p.vertex_end, l).start),
        );
        total += p.mesh.edges.len();
    }
    if total > u32::MAX as usize {
        return Err(Error::ResourceLimit {
            what: format!("mesh with {total} vertices (u32 indices)"),
            limit: u64::from(u32::MAX),
        });
    }
    let index: FxHashMap<u64, u32> = bricks
        .iter()
        .enumerate()
        .map(|(i, k)| (pack(*k), i as u32))
        .collect();
    let edges_of = |bi: usize| parts[bi / chunk].edges(bi % chunk);
    let resolved = par::map(parts.len(), |c| -> Result<Vec<[u32; 3]>> {
        let part = &parts[c];
        let mut tris = Vec::with_capacity(part.mesh.tris.len());
        for l in 0..part.tri_end.len() {
            let bi = c * chunk + l;
            let key = bricks[bi];
            let mut cache = [None::<usize>; 8];
            cache[0] = Some(bi);
            let mut target_of = |code: usize| -> Result<usize> {
                if let Some(t) = cache[code] {
                    return Ok(t);
                }
                let nk = [0, 1, 2].map(|a| key[a] + ((code >> a) & 1) as i32);
                let t = index.get(&pack(nk)).map(|&t| t as usize).ok_or_else(|| {
                    Error::Internal(format!(
                        "triangle of brick {key:?} references edge of missing brick {nk:?}"
                    ))
                })?;
                cache[code] = Some(t);
                Ok(t)
            };
            for tri in &part.mesh.tris[ChunkMesh::range(&part.tri_end, l)] {
                let mut out = [0u32; 3];
                for (o, &r) in out.iter_mut().zip(tri) {
                    let target = target_of((r >> CODE_SHIFT) as usize)?;
                    let id = r & ID_MASK;
                    let pos = edges_of(target).binary_search(&id).map_err(|_| {
                        Error::Internal(format!(
                            "brick {key:?} references edge {id} without vertex in brick {:?}",
                            bricks[target]
                        ))
                    })?;
                    *o = (base[target] + pos) as u32;
                }
                tris.push(out);
            }
        }
        Ok(tris)
    });
    drop(base);
    let mut mesh = Mesh {
        positions: Vec::with_capacity(total),
        normals: Vec::with_capacity(total),
        triangles: Vec::with_capacity(
            resolved
                .iter()
                .map(|r| r.as_ref().map_or(0, Vec::len))
                .sum(),
        ),
        vertex_atom: Vec::with_capacity(total),
        vertex_patch: Vec::with_capacity(total),
        vertex_component: Vec::new(),
    };
    for (p, tris) in parts.into_iter().zip(resolved) {
        let p = p.mesh;
        mesh.positions.extend_from_slice(&p.positions);
        mesh.normals.extend_from_slice(&p.normals);
        mesh.vertex_atom.extend_from_slice(&p.atoms);
        mesh.vertex_patch.extend_from_slice(&p.patches);
        mesh.triangles.extend(tris?);
    }
    Ok(mesh)
}
