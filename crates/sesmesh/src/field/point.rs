//! Exact `f64` evaluation of the surface field at arbitrary points (Design
//! §2.3, §7.6): the public [`SesField`] and the evaluator used for brick
//! classification and, later, vertex refinement.

use super::SurfaceKind;
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
        let arr = match kind {
            SurfaceKind::Ses => Arrangement::build(spheres, probe_radius)?,
            SurfaceKind::Sas => Arrangement::build_spheres(spheres, probe_radius)?,
            SurfaceKind::Vdw => Arrangement::build_spheres(spheres, 0.0)?,
        };
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
            let iu = i as usize;
            if a.status[iu] != Status::Exposed {
                return;
            }
            let v = x - a.centers[iu];
            let len = v.norm();
            let lb = (len - a.expanded[iu]).abs();
            if lb >= best.distance || len < 1e-12 {
                return;
            }
            let q = a.centers[iu] + v * (a.expanded[iu] / len);
            let free = a
                .neighbors
                .row(iu)
                .iter()
                .all(|&k| q.dist(a.centers[k as usize]) >= a.expanded[k as usize] - EPS);
            if free {
                best = PointEval {
                    distance: lb,
                    closest: q,
                    feature: Feature::Sphere(i),
                };
            }
        });
        self.circle_cells
            .for_each_near(x, cutoff + self.rho_max, |ci| {
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
                let free = a
                    .circle_blockers
                    .row(ci as usize)
                    .iter()
                    .all(|&k| q.dist(a.centers[k as usize]) >= a.expanded[k as usize] - EPS);
                if free {
                    best = PointEval {
                        distance: dist,
                        closest: q,
                        feature: Feature::Circle(ci),
                    };
                }
            });
        self.vertex_cells.for_each_near(x, cutoff, |vi| {
            let p = a.vertices[vi as usize].position;
            let d = x.dist(p);
            if d < best.distance {
                best = PointEval {
                    distance: d,
                    closest: p,
                    feature: Feature::Vertex(vi),
                };
            }
        });
        best
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
