//! Exact `f64` evaluation of the surface field at arbitrary points (Design
//! §2.3, §7.6): the public [`SesField`] and the evaluator used for brick
//! classification and, later, vertex refinement.

use super::SurfaceKind;
use crate::control::Control;
use crate::error::{Error, Result};
use crate::geom::{Arrangement, CellGrid, EPS, Sphere, Status, Vec3};

/// What carries the closest point of a field sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Patch {
    /// The point is in the solvent region where the probe centre can go
    /// (SES) or outside all spheres (SAS, vdW, no sphere nearby).
    Solvent,
    /// Convex patch on an atom sphere (input atom index).
    Convex {
        /// Input index of the atom.
        atom: usize,
    },
    /// Saddle (toroidal) patch between two atoms (input indices).
    Saddle {
        /// Input indices of the two atoms.
        atoms: [usize; 2],
    },
    /// Concave patch of a probe touching three atoms (input indices).
    Concave {
        /// Input indices of the three atoms.
        atoms: [usize; 3],
    },
}

/// A field value with its closest surface feature.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FieldSample {
    /// Field value: negative inside the molecule, 0 on the surface. For the
    /// SES it is `r_p − d(x, P)`; for SAS/vdW `min_i(|x − c_i| − R_i)`.
    pub value: f64,
    /// SES: closest point of the probe-accessible region `P`; SAS/vdW:
    /// closest point on the nearest sphere.
    pub closest: [f64; 3],
    /// Outward unit normal of the level set through the point (`None` where
    /// undefined, e.g. in the solvent region of the SES).
    pub normal: Option<[f64; 3]>,
    /// Feature carrying the closest point.
    pub patch: Patch,
}

/// Internal feature of a point evaluation (internal indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Feature {
    /// Point is probe-accessible.
    Solvent,
    /// Nothing within the cutoff.
    Beyond,
    Sphere(u32),
    Circle(u32),
    Vertex(u32),
}

/// Distance to `P` (clamped at the cutoff) and the closest point.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PointEval {
    pub(crate) distance: f64,
    pub(crate) closest: Vec3,
    pub(crate) feature: Feature,
}

/// Candidates for points of one ball (global internal indices), from
/// [`SesField::gather_local`].
#[derive(Debug, Default)]
pub(crate) struct LocalCandidates {
    /// SES: spheres that can contain a point of the ball; SAS/vdW: spheres
    /// within reach, nearest first.
    inside: Vec<u32>,
    /// Non-contained spheres within reach (possible blockers), ascending.
    near: Vec<u32>,
    /// Class S candidates, nearest first, with their blockers among `near`
    /// (`s_blk[s_off[k]..s_off[k + 1]]`).
    spheres: Vec<u32>,
    s_off: Vec<u32>,
    s_blk: Vec<u32>,
    /// Class C candidates, nearest first, with their blockers among `near`.
    circles: Vec<u32>,
    c_off: Vec<u32>,
    c_blk: Vec<u32>,
    vertices: Vec<u32>,
    order: Vec<(f64, u32)>,
}

impl LocalCandidates {
    fn clear(&mut self) {
        self.inside.clear();
        self.near.clear();
        self.spheres.clear();
        self.s_off.clear();
        self.s_off.push(0);
        self.s_blk.clear();
        self.circles.clear();
        self.c_off.clear();
        self.c_off.push(0);
        self.c_blk.clear();
        self.vertices.clear();
        self.order.clear();
    }

    /// Appends the members of `row` that are in `near` to `out`.
    fn filter_blockers(near: &[u32], row: &[u32], out: &mut Vec<u32>) {
        out.extend(row.iter().filter(|k| near.binary_search(k).is_ok()));
    }
}

/// Nearest first; ties by index (deterministic).
fn sort_nearest(v: &mut [(f64, u32)]) {
    v.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
}

/// Result of [`SesField::eval_local`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct NearEval {
    pub(crate) value: f64,
    pub(crate) feature: Feature,
    /// SES: closest point of `P` (the probe centre for surface points);
    /// SAS/vdW: the evaluated point itself.
    pub(crate) closest: Vec3,
}

/// The exact surface field of a sphere set: SES (Connolly), SAS or vdW.
///
/// ```
/// use sesmesh::{SesField, Sphere, SurfaceKind};
///
/// let atoms = [Sphere::new([0.0, 0.0, 0.0], 1.7), Sphere::new([3.0, 0.0, 0.0], 1.7)];
/// let field = SesField::new(&atoms, 1.4, SurfaceKind::Ses)?;
/// // On the convex cap of the first atom the SES is the atom sphere itself.
/// assert!(field.value([-1.7, 0.0, 0.0]).abs() < 1e-12);
/// // Between the atoms the toroidal saddle fills the groove: this point is
/// // outside both atom spheres (1.80 Å from each centre) yet inside the SES.
/// assert!(field.value([1.5, 1.0, 0.0]) < 0.0);
/// // Further out, the probe can reach it: outside the SES.
/// assert!(field.value([1.5, 1.5, 0.0]) > 0.0);
/// # Ok::<(), sesmesh::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct SesField {
    pub(crate) arr: Arrangement,
    pub(crate) kind: SurfaceKind,
    pub(crate) probe: f64,
    pub(crate) circle_cells: CellGrid,
    pub(crate) vertex_cells: CellGrid,
    pub(crate) rho_max: f64,
    /// Bounding-box diagonal plus margins: an upper bound for any distance
    /// the evaluator needs.
    extent: f64,
}

impl SesField {
    /// Builds the field. For the SES a probe radius of 0 gives the vdW
    /// surface. Spheres with radius 0 are ignored.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for non-finite or negative values.
    pub fn new(spheres: &[Sphere], probe_radius: f64, kind: SurfaceKind) -> Result<Self> {
        Self::new_with(spheres, probe_radius, kind, &Control::new())
    }

    /// [`SesField::new`] with cancellation.
    pub(crate) fn new_with(
        spheres: &[Sphere],
        probe_radius: f64,
        kind: SurfaceKind,
        control: &Control<'_>,
    ) -> Result<Self> {
        if !(probe_radius.is_finite() && probe_radius >= 0.0) {
            return Err(Error::InvalidInput(format!(
                "probe radius must be finite and >= 0, got {probe_radius}"
            )));
        }
        let kind = if kind == SurfaceKind::Ses && probe_radius == 0.0 {
            SurfaceKind::Vdw
        } else {
            kind
        };
        let mut arr = match kind {
            SurfaceKind::Ses => Arrangement::build_with(spheres, probe_radius, true, control)?,
            SurfaceKind::Sas => Arrangement::build_with(spheres, probe_radius, false, control)?,
            SurfaceKind::Vdw => Arrangement::build_with(spheres, 0.0, false, control)?,
        };
        // The evaluator needs neither the arcs nor the neighbour rows of
        // spheres that are not exposed (every reader filters on
        // `Status::Exposed` first): release them (issue #21).
        arr.arcs = Vec::new();
        let status = &arr.status;
        arr.neighbors.retain_rows(|i| status[i] == Status::Exposed);
        let cell = (2.0 * arr.max_expanded).max(1e-6);
        let circle_cells = CellGrid::new(
            &arr.circles.iter().map(|c| c.center).collect::<Vec<_>>(),
            cell,
        );
        let vertex_cells = CellGrid::new(
            &arr.vertices.iter().map(|v| v.position).collect::<Vec<_>>(),
            cell,
        );
        let rho_max = arr.circles.iter().map(|c| c.radius).fold(0.0, f64::max);
        let extent = match arr.centers.first() {
            None => 0.0,
            Some(&first) => {
                let (lo, hi) = arr
                    .centers
                    .iter()
                    .fold((first, first), |(lo, hi), &c| (lo.min(c), hi.max(c)));
                (hi - lo).norm() + 4.0 * arr.max_expanded + 4.0 * probe_radius + 1.0
            }
        };
        Ok(Self {
            arr,
            kind,
            probe: probe_radius,
            circle_cells,
            vertex_cells,
            rho_max,
            extent,
        })
    }

    /// The surface kind (SES with probe radius 0 reports [`SurfaceKind::Vdw`]).
    pub fn kind(&self) -> SurfaceKind {
        self.kind
    }

    /// Probe radius in Å.
    pub fn probe_radius(&self) -> f64 {
        self.probe
    }

    /// Field value at `point` (exact, see [`FieldSample::value`]).
    pub fn value(&self, point: [f64; 3]) -> f64 {
        self.sample(point).value
    }

    /// Field value, closest feature and normal at `point`.
    pub fn sample(&self, point: [f64; 3]) -> FieldSample {
        let x = Vec3::from_array(point);
        let orig = |i: u32| self.arr.orig[i as usize] as usize;
        if self.arr.centers.is_empty() {
            return FieldSample {
                value: if self.kind == SurfaceKind::Ses {
                    self.probe
                } else {
                    f64::INFINITY
                },
                closest: point,
                normal: None,
                patch: Patch::Solvent,
            };
        }
        match self.kind {
            SurfaceKind::Ses => {
                let mut cutoff = self.probe + 4.0;
                let e = loop {
                    let e = self.ses_eval(x, cutoff);
                    if e.feature != Feature::Beyond || cutoff > self.extent {
                        break e;
                    }
                    cutoff *= 2.0;
                };
                let patch = match e.feature {
                    Feature::Solvent | Feature::Beyond => Patch::Solvent,
                    Feature::Sphere(i) => Patch::Convex { atom: orig(i) },
                    Feature::Circle(c) => {
                        let c = &self.arr.circles[c as usize];
                        Patch::Saddle {
                            atoms: [orig(c.i), orig(c.j)],
                        }
                    }
                    Feature::Vertex(v) => Patch::Concave {
                        atoms: self.arr.vertices[v as usize].atoms.map(orig),
                    },
                };
                let to_p = e.closest - x;
                let normal = (patch != Patch::Solvent && to_p.norm() > 1e-12)
                    .then(|| to_p.normalized().to_array());
                FieldSample {
                    value: self.probe - e.distance,
                    closest: e.closest.to_array(),
                    normal,
                    patch,
                }
            }
            SurfaceKind::Sas | SurfaceKind::Vdw => {
                let mut cutoff = 4.0;
                let (value, i) = loop {
                    let (v, i) = self.sphere_eval(x, cutoff);
                    if i.is_some() || cutoff > self.extent {
                        break (v, i);
                    }
                    cutoff *= 2.0;
                };
                let Some(i) = i else {
                    return FieldSample {
                        value,
                        closest: point,
                        normal: None,
                        patch: Patch::Solvent,
                    };
                };
                let (c, r) = (self.arr.centers[i as usize], self.arr.expanded[i as usize]);
                let dir = x - c;
                let len = dir.norm();
                let unit = if len > 1e-12 { dir / len } else { Vec3::X };
                FieldSample {
                    value,
                    closest: (c + unit * r).to_array(),
                    normal: (len > 1e-12).then(|| unit.to_array()),
                    patch: Patch::Convex { atom: orig(i) },
                }
            }
        }
    }

    /// `true` if `x` is inside some (non-contained) expanded sphere.
    fn inside_sas(&self, x: Vec3) -> bool {
        let a = &self.arr;
        let mut inside = false;
        a.cells.for_each_near(x, a.max_expanded, |k| {
            let k = k as usize;
            if !inside && a.status[k] != Status::Contained && x.dist(a.centers[k]) < a.expanded[k] {
                inside = true;
            }
        });
        inside
    }

    /// SES distance `d(x, P)` clamped at `cutoff` (Design §2.3).
    pub(crate) fn ses_eval(&self, x: Vec3, cutoff: f64) -> PointEval {
        let a = &self.arr;
        if !self.inside_sas(x) {
            return PointEval {
                distance: 0.0,
                closest: x,
                feature: Feature::Solvent,
            };
        }
        let mut best = PointEval {
            distance: cutoff,
            closest: x,
            feature: Feature::Beyond,
        };
        a.cells.for_each_near(x, cutoff + a.max_expanded, |i| {
            if a.status[i as usize] == Status::Exposed {
                self.try_sphere(x, i, &mut best);
            }
        });
        self.circle_cells
            .for_each_near(x, cutoff + self.rho_max, |ci| {
                self.try_circle(x, ci, &mut best);
            });
        self.vertex_cells
            .for_each_near(x, cutoff, |vi| self.try_vertex(x, vi, &mut best));
        best
    }

    /// Class S candidate: radial projection of `x` on exposed sphere `i`.
    #[inline]
    fn try_sphere(&self, x: Vec3, i: u32, best: &mut PointEval) {
        self.try_sphere_with(x, i, self.arr.neighbors.row(i as usize), best);
    }

    /// [`SesField::try_sphere`] with an explicit blocker list (a superset of
    /// the spheres that can contain the candidate point).
    #[inline]
    fn try_sphere_with(&self, x: Vec3, i: u32, blockers: &[u32], best: &mut PointEval) {
        let a = &self.arr;
        let iu = i as usize;
        let v = x - a.centers[iu];
        let len = v.norm();
        let lb = (len - a.expanded[iu]).abs();
        if lb >= best.distance || len < 1e-12 {
            return;
        }
        let q = a.centers[iu] + v * (a.expanded[iu] / len);
        let free = blockers
            .iter()
            .all(|&k| q.dist(a.centers[k as usize]) >= a.expanded[k as usize] - EPS);
        if free {
            *best = PointEval {
                distance: lb,
                closest: q,
                feature: Feature::Sphere(i),
            };
        }
    }

    /// Class C candidate: closest point of circle `ci`.
    #[inline]
    fn try_circle(&self, x: Vec3, ci: u32, best: &mut PointEval) {
        self.try_circle_with(x, ci, self.arr.circle_blockers.row(ci as usize), best);
    }

    /// [`SesField::try_circle`] with an explicit blocker list.
    #[inline]
    fn try_circle_with(&self, x: Vec3, ci: u32, blockers: &[u32], best: &mut PointEval) {
        let a = &self.arr;
        let c = &a.circles[ci as usize];
        let y = x - c.center;
        let along = y.dot(c.axis);
        let w = y - c.axis * along;
        let wl = w.norm();
        let dist = (along * along + (wl - c.radius) * (wl - c.radius)).sqrt();
        if dist >= best.distance {
            return;
        }
        let dir = if wl > 1e-12 { w / wl } else { c.e1 };
        let q = c.center + dir * c.radius;
        let free = blockers
            .iter()
            .all(|&k| q.dist(a.centers[k as usize]) >= a.expanded[k as usize] - EPS);
        if free {
            *best = PointEval {
                distance: dist,
                closest: q,
                feature: Feature::Circle(ci),
            };
        }
    }

    /// Class V candidate: SAS vertex `vi`.
    #[inline]
    fn try_vertex(&self, x: Vec3, vi: u32, best: &mut PointEval) {
        let p = self.arr.vertices[vi as usize].position;
        let d = x.dist(p);
        if d < best.distance {
            *best = PointEval {
                distance: d,
                closest: p,
                feature: Feature::Vertex(vi),
            };
        }
    }

    /// Collects into `out` every candidate that can matter for the field
    /// clamped at this `band` (SES: distance cutoff `r_p + band`) at any
    /// point within `rho` of
    /// `b` (the same completeness argument as the kernel's gathering,
    /// Design §7.4), nearest first.
    pub(crate) fn gather_local(&self, b: Vec3, rho: f64, band: f64, out: &mut LocalCandidates) {
        let a = &self.arr;
        out.clear();
        match self.kind {
            SurfaceKind::Ses => {
                // A closest point q of a point x of the ball is within
                // `cutoff` of x, hence within `reach` of b: a sphere that can
                // block q satisfies |b − c| < R + reach.
                let reach = self.probe + band + rho;
                a.cells.for_each_near(b, reach + a.max_expanded, |k| {
                    let ku = k as usize;
                    if a.status[ku] == Status::Contained {
                        return;
                    }
                    let d = b.dist(a.centers[ku]);
                    if d < a.expanded[ku] + reach {
                        out.near.push(k);
                    }
                    // Can contain a point of the ball (SAS inside test).
                    if d < a.expanded[ku] + rho {
                        out.inside.push(k);
                    }
                    let lb = (d - a.expanded[ku]).abs();
                    if a.status[ku] == Status::Exposed && lb <= reach {
                        out.order.push((lb, k));
                    }
                });
                out.near.sort_unstable();
                sort_nearest(&mut out.order);
                for &(_, k) in &out.order {
                    out.spheres.push(k);
                    LocalCandidates::filter_blockers(
                        &out.near,
                        a.neighbors.row(k as usize),
                        &mut out.s_blk,
                    );
                    out.s_off.push(out.s_blk.len() as u32);
                }
                out.order.clear();
                self.circle_cells
                    .for_each_near(b, reach + self.rho_max, |ci| {
                        let c = &a.circles[ci as usize];
                        let y = b - c.center;
                        let along = y.dot(c.axis);
                        let wl = (y - c.axis * along).norm();
                        let dist = (along * along + (wl - c.radius) * (wl - c.radius)).sqrt();
                        if dist <= reach {
                            out.order.push((dist, ci));
                        }
                    });
                sort_nearest(&mut out.order);
                for &(_, ci) in &out.order {
                    out.circles.push(ci);
                    LocalCandidates::filter_blockers(
                        &out.near,
                        a.circle_blockers.row(ci as usize),
                        &mut out.c_blk,
                    );
                    out.c_off.push(out.c_blk.len() as u32);
                }
                out.order.clear();
                self.vertex_cells.for_each_near(b, reach, |vi| {
                    let d = b.dist(a.vertices[vi as usize].position);
                    if d <= reach {
                        out.order.push((d, vi));
                    }
                });
                sort_nearest(&mut out.order);
                out.vertices.extend(out.order.iter().map(|o| o.1));
            }
            SurfaceKind::Sas | SurfaceKind::Vdw => {
                let reach = band + rho;
                a.cells.for_each_near(b, reach + a.max_expanded, |k| {
                    let ku = k as usize;
                    if a.status[ku] != Status::Contained
                        && b.dist(a.centers[ku]) - a.expanded[ku] < reach
                    {
                        out.order.push((b.dist(a.centers[ku]) - a.expanded[ku], k));
                    }
                });
                sort_nearest(&mut out.order);
                out.inside.extend(out.order.iter().map(|o| o.1));
            }
        }
        out.order.clear();
    }

    /// Field value near the surface (exact where `|F| < band`) with the
    /// feature carrying the closest point and that point, from candidates
    /// gathered by [`SesField::gather_local`] for a ball containing `x` (same
    /// `band`); used to place and annotate mesh vertices (Design §8.2).
    pub(crate) fn eval_local(&self, x: Vec3, band: f64, l: &LocalCandidates) -> NearEval {
        let a = &self.arr;
        match self.kind {
            SurfaceKind::Ses => {
                let inside = l
                    .inside
                    .iter()
                    .any(|&k| x.dist(a.centers[k as usize]) < a.expanded[k as usize]);
                if !inside {
                    return NearEval {
                        value: self.probe,
                        feature: Feature::Solvent,
                        closest: x,
                    };
                }
                let mut best = PointEval {
                    distance: self.probe + band,
                    closest: x,
                    feature: Feature::Beyond,
                };
                for (k, &i) in l.spheres.iter().enumerate() {
                    let blk = &l.s_blk[l.s_off[k] as usize..l.s_off[k + 1] as usize];
                    self.try_sphere_with(x, i, blk, &mut best);
                }
                for (k, &ci) in l.circles.iter().enumerate() {
                    let blk = &l.c_blk[l.c_off[k] as usize..l.c_off[k + 1] as usize];
                    self.try_circle_with(x, ci, blk, &mut best);
                }
                for &vi in &l.vertices {
                    self.try_vertex(x, vi, &mut best);
                }
                NearEval {
                    value: self.probe - best.distance,
                    feature: best.feature,
                    closest: best.closest,
                }
            }
            SurfaceKind::Sas | SurfaceKind::Vdw => {
                let mut best = (band, None);
                for &k in &l.inside {
                    let v = x.dist(a.centers[k as usize]) - a.expanded[k as usize];
                    if v < best.0 {
                        best = (v, Some(k));
                    }
                }
                NearEval {
                    value: best.0,
                    feature: best.1.map_or(Feature::Beyond, Feature::Sphere),
                    closest: x,
                }
            }
        }
    }

    /// `min_i(|x − c_i| − R_i)` over spheres closer than `cutoff`, clamped at
    /// `cutoff`, with the minimising sphere.
    pub(crate) fn sphere_eval(&self, x: Vec3, cutoff: f64) -> (f64, Option<u32>) {
        let a = &self.arr;
        let mut best = (cutoff, None);
        a.cells.for_each_near(x, cutoff + a.max_expanded, |k| {
            let ku = k as usize;
            if a.status[ku] == Status::Contained {
                return;
            }
            let v = x.dist(a.centers[ku]) - a.expanded[ku];
            if v < best.0 {
                best = (v, Some(k));
            }
        });
        best
    }

    /// Field value clamped to at most `band` (SES: distance clamped at
    /// `probe + band`), the form sampled on the lattice (Design §7.2).
    pub(crate) fn clamped_value(&self, x: Vec3, band: f64) -> f64 {
        match self.kind {
            SurfaceKind::Ses => self.probe - self.ses_eval(x, self.probe + band).distance,
            SurfaceKind::Sas | SurfaceKind::Vdw => self.sphere_eval(x, band).0,
        }
    }
}
