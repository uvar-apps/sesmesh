// sesmesh — Solvent Excluded Surface meshes for molecules
// Copyright (c) 2026 Ugo Varetto
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Brute-force reference for the SES field and the SAS arrangement.
//!
//! Implements Design §2 directly: with expanded radii `R_i = r_i + r_p` and
//! the probe-accessible region `P = {x : |x - c_i| >= R_i for all i}`,
//! `F(x) = r_p - d(x, P)`. For `x` outside `P`, `d(x, P)` is the minimum over
//! three candidate classes — radial projections onto spheres, projections onto
//! pairwise intersection circles, and triple intersection points — each valid
//! only if it is not strictly inside any other expanded sphere. Validity is
//! always tested against **every** sphere; the only shortcut is enumerating
//! circles and triples from intersecting pairs, which is exact.
//!
//! Triple points use the classic three-sphere trilateration, a different
//! construction from the library's circle/sphere intersection.

use crate::V3;
use crate::v::{add, cross, dist, dot, mul, norm, normalize, sub};

/// Tolerance (Å) of the "strictly inside another sphere" test.
pub const TOL: f64 = 1e-9;

/// What the closest point of `P` lies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {
    /// The query point itself is probe-accessible (`d = 0`).
    Accessible,
    /// Radial projection onto expanded sphere `i` (convex SES patch).
    Sphere(usize),
    /// Projection onto the intersection circle of spheres `i < j` (saddle).
    Circle(usize, usize),
    /// Triple intersection point of spheres `i < j < k` (concave patch).
    Vertex([usize; 3]),
}

/// Result of a field evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// `F(x) = r_p - d(x, P)`: negative inside the molecule, 0 on the SES.
    pub value: f64,
    /// `d(x, P)`.
    pub distance: f64,
    /// Closest point of `P` (equal to `x` when accessible).
    pub closest: V3,
    /// Feature carrying the closest point.
    pub feature: Feature,
}

/// Intersection circle of two expanded spheres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    /// First sphere (smaller index).
    pub i: usize,
    /// Second sphere.
    pub j: usize,
    /// Centre.
    pub center: V3,
    /// Unit normal of the circle plane, from `i` towards `j`.
    pub axis: V3,
    /// Radius.
    pub radius: f64,
}

impl Circle {
    /// Point of the circle at angle `theta` in an arbitrary fixed frame.
    pub fn point(&self, theta: f64) -> V3 {
        let t = if self.axis[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let e1 = normalize(cross(self.axis, t));
        let e2 = cross(self.axis, e1);
        add(
            self.center,
            add(
                mul(e1, self.radius * theta.cos()),
                mul(e2, self.radius * theta.sin()),
            ),
        )
    }
}

/// An exposed triple intersection point (a probe position touching three
/// atoms).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    /// The three spheres, ascending.
    pub atoms: [usize; 3],
    /// Position.
    pub position: V3,
    /// Smallest `|q - c_l| - R_l` over all other spheres (how far from being
    /// buried); tiny values flag near-degenerate configurations.
    pub margin: f64,
}

/// Brute-force oracle over a fixed set of spheres.
#[derive(Debug, Clone)]
pub struct Oracle {
    centers: Vec<V3>,
    radii: Vec<f64>,
    expanded: Vec<f64>,
    probe: f64,
    circles: Vec<Circle>,
    vertices: Vec<Vertex>,
}

impl Oracle {
    /// Builds the oracle. Spheres with radius `<= 0` are ignored.
    ///
    /// # Panics
    /// If the lengths differ or the probe radius is negative.
    pub fn new(centers: &[V3], radii: &[f64], probe: f64) -> Self {
        assert_eq!(centers.len(), radii.len());
        assert!(probe >= 0.0);
        let expanded: Vec<f64> = radii
            .iter()
            .map(|&r| if r > 0.0 { r + probe } else { 0.0 })
            .collect();
        let n = centers.len();
        let mut oracle = Self {
            centers: centers.to_vec(),
            radii: radii.to_vec(),
            expanded,
            probe,
            circles: Vec::new(),
            vertices: Vec::new(),
        };
        // Exact pair filter: circles exist for intersecting, non-nested pairs.
        let mut adjacent = vec![false; n * n];
        for i in 0..n {
            for j in i + 1..n {
                if let Some(c) = oracle.circle(i, j) {
                    adjacent[i * n + j] = true;
                    adjacent[j * n + i] = true;
                    oracle.circles.push(c);
                }
            }
        }
        for i in 0..n {
            for j in i + 1..n {
                if !adjacent[i * n + j] {
                    continue;
                }
                for k in j + 1..n {
                    if !(adjacent[i * n + k] && adjacent[j * n + k]) {
                        continue;
                    }
                    for q in oracle.trilaterate(i, j, k) {
                        let margin = oracle.margin(q, &[i, j, k]);
                        if margin >= -TOL {
                            oracle.vertices.push(Vertex {
                                atoms: [i, j, k],
                                position: q,
                                margin,
                            });
                        }
                    }
                }
            }
        }
        oracle
    }

    /// Probe radius.
    pub fn probe(&self) -> f64 {
        self.probe
    }

    /// Expanded radius `R_i` (0 for ignored spheres).
    pub fn expanded_radius(&self, i: usize) -> f64 {
        self.expanded[i]
    }

    /// All intersection circles (exposed or not).
    pub fn circles(&self) -> &[Circle] {
        &self.circles
    }

    /// All exposed triple points.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    fn active(&self, i: usize) -> bool {
        self.radii[i] > 0.0
    }

    fn circle(&self, i: usize, j: usize) -> Option<Circle> {
        if !self.active(i) || !self.active(j) {
            return None;
        }
        let (ri, rj) = (self.expanded[i], self.expanded[j]);
        let d = dist(self.centers[i], self.centers[j]);
        if d >= ri + rj || d <= (ri - rj).abs() {
            return None;
        }
        let axis = mul(sub(self.centers[j], self.centers[i]), 1.0 / d);
        let t = (d * d + ri * ri - rj * rj) / (2.0 * d);
        let rho2 = ri * ri - t * t;
        (rho2 > 0.0).then(|| Circle {
            i,
            j,
            center: add(self.centers[i], mul(axis, t)),
            axis,
            radius: rho2.sqrt(),
        })
    }

    /// The (up to) two intersection points of three expanded spheres.
    pub fn trilaterate(&self, i: usize, j: usize, k: usize) -> Vec<V3> {
        let (p1, p2, p3) = (self.centers[i], self.centers[j], self.centers[k]);
        let (r1, r2, r3) = (self.expanded[i], self.expanded[j], self.expanded[k]);
        let d = dist(p1, p2);
        let ex = mul(sub(p2, p1), 1.0 / d);
        let ii = dot(ex, sub(p3, p1));
        let ey_raw = sub(sub(p3, p1), mul(ex, ii));
        let jn = norm(ey_raw);
        if jn < 1e-12 {
            return Vec::new(); // collinear centres: no isolated triple points
        }
        let ey = mul(ey_raw, 1.0 / jn);
        let ez = cross(ex, ey);
        let jj = dot(ey, sub(p3, p1));
        let x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
        let y = (r1 * r1 - r3 * r3 + ii * ii + jj * jj) / (2.0 * jj) - ii * x / jj;
        let z2 = r1 * r1 - x * x - y * y;
        if z2 <= 0.0 {
            return Vec::new();
        }
        let z = z2.sqrt();
        let base = add(p1, add(mul(ex, x), mul(ey, y)));
        vec![add(base, mul(ez, z)), sub(base, mul(ez, z))]
    }

    /// Smallest `|q - c_l| - R_l` over active spheres not in `exclude`
    /// (`+inf` if none): negative means `q` is inside some sphere.
    pub fn margin(&self, q: V3, exclude: &[usize]) -> f64 {
        (0..self.centers.len())
            .filter(|l| self.active(*l) && !exclude.contains(l))
            .map(|l| dist(q, self.centers[l]) - self.expanded[l])
            .fold(f64::INFINITY, f64::min)
    }

    /// `true` if `q` is probe-accessible (not strictly inside any expanded
    /// sphere, with tolerance [`TOL`]).
    pub fn is_accessible(&self, q: V3) -> bool {
        self.margin(q, &[]) >= -TOL
    }

    /// The SES field and the closest point of `P` (Design §2.2–2.3).
    pub fn ses(&self, x: V3) -> Sample {
        if self.margin(x, &[]) >= 0.0 {
            return Sample {
                value: self.probe,
                distance: 0.0,
                closest: x,
                feature: Feature::Accessible,
            };
        }
        let mut best = Sample {
            value: f64::NEG_INFINITY,
            distance: f64::INFINITY,
            closest: x,
            feature: Feature::Accessible,
        };
        let offer = |q: V3, feature: Feature, best: &mut Sample| {
            let d = dist(x, q);
            if d < best.distance {
                *best = Sample {
                    value: self.probe - d,
                    distance: d,
                    closest: q,
                    feature,
                };
            }
        };
        for i in (0..self.centers.len()).filter(|&i| self.active(i)) {
            let v = sub(x, self.centers[i]);
            let len = norm(v);
            if len < 1e-12 {
                continue;
            }
            let q = add(self.centers[i], mul(v, self.expanded[i] / len));
            if self.margin(q, &[i]) >= -TOL {
                offer(q, Feature::Sphere(i), &mut best);
            }
        }
        for c in &self.circles {
            let y = sub(x, c.center);
            let w = sub(y, mul(c.axis, dot(y, c.axis)));
            let wl = norm(w);
            if wl < 1e-12 {
                continue; // equidistant circle: minimum is attained at a vertex
            }
            let q = add(c.center, mul(w, c.radius / wl));
            if self.margin(q, &[c.i, c.j]) >= -TOL {
                offer(q, Feature::Circle(c.i, c.j), &mut best);
            }
        }
        for v in &self.vertices {
            offer(v.position, Feature::Vertex(v.atoms), &mut best);
        }
        best
    }

    /// Outward unit normal direction of the SES field at `x` (`(p - x)/|p - x|`
    /// with `p` the closest point of `P`), or `None` when `x` is accessible.
    pub fn ses_gradient(&self, x: V3) -> Option<V3> {
        let s = self.ses(x);
        (s.feature != Feature::Accessible).then(|| normalize(sub(s.closest, x)))
    }

    /// Signed distance-like SAS field `min_i(|x - c_i| - R_i)`.
    pub fn sas(&self, x: V3) -> f64 {
        self.margin(x, &[])
    }

    /// Signed distance-like vdW field `min_i(|x - c_i| - r_i)`.
    pub fn vdw(&self, x: V3) -> f64 {
        (0..self.centers.len())
            .filter(|&i| self.active(i))
            .map(|i| dist(x, self.centers[i]) - self.radii[i])
            .fold(f64::INFINITY, f64::min)
    }
}

/// Points of a Fibonacci lattice on a sphere (`n` points).
pub fn fibonacci_sphere(center: V3, radius: f64, n: usize) -> Vec<V3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|k| {
            let z = 1.0 - 2.0 * (k as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).sqrt();
            let phi = golden * k as f64;
            add(center, mul([r * phi.cos(), r * phi.sin(), z], radius))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clusters::{Rng, random_chain};

    #[test]
    fn single_sphere_field_is_vdw_distance() {
        let o = Oracle::new(&[[1.0, 2.0, 3.0]], &[1.7], 1.4);
        for x in [[1.0, 2.0, 3.5], [2.5, 2.0, 3.0], [1.0, 4.0, 3.0]] {
            let s = o.ses(x);
            let expected = dist(x, [1.0, 2.0, 3.0]) - 1.7;
            assert!((s.value - expected).abs() < 1e-12, "{x:?}");
        }
        assert_eq!(o.ses([10.0, 0.0, 0.0]).feature, Feature::Accessible);
    }

    #[test]
    fn two_sphere_saddle_point() {
        // Equal spheres r = 1.7 at distance 3; the probe-centre circle has
        // radius rho = sqrt(3.1^2 - 1.5^2). The midpoint's closest point of P
        // is on that circle, at distance rho.
        let o = Oracle::new(&[[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], &[1.7, 1.7], 1.4);
        let rho = (3.1f64 * 3.1 - 1.5 * 1.5).sqrt();
        let s = o.ses([0.0, 0.3, 0.0]);
        assert_eq!(s.feature, Feature::Circle(0, 1));
        assert!((s.distance - (rho - 0.3)).abs() < 1e-12);
        assert_eq!(o.circles().len(), 1);
        assert!(o.vertices().is_empty());
    }

    #[test]
    fn closest_points_agree_with_dense_sampling() {
        // Independent check of the three-candidate theory: sample exposed
        // points densely on every expanded sphere; the sampled minimum
        // distance can never beat the oracle and must approach it.
        let mut rng = Rng::new(7);
        for _ in 0..6 {
            let c = random_chain(&mut rng, 6);
            let o = Oracle::new(&c.centers, &c.radii, c.probe);
            let mut samples = Vec::new();
            for (i, &ci) in c.centers.iter().enumerate() {
                for q in fibonacci_sphere(ci, o.expanded_radius(i), 40_000) {
                    if o.margin(q, &[i]) >= 0.0 {
                        samples.push(q);
                    }
                }
            }
            for _ in 0..40 {
                let k = rng.below(c.centers.len());
                let x = add(
                    c.centers[k],
                    [
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(-2.0, 2.0),
                        rng.uniform(-2.0, 2.0),
                    ],
                );
                let s = o.ses(x);
                if s.feature == Feature::Accessible {
                    continue;
                }
                let sampled = samples
                    .iter()
                    .map(|&q| dist(x, q))
                    .fold(f64::INFINITY, f64::min);
                assert!(
                    sampled >= s.distance - 1e-9,
                    "sampled {sampled} < oracle {}",
                    s.distance
                );
                assert!(
                    sampled - s.distance < 0.06,
                    "sampled {sampled} vs oracle {}",
                    s.distance
                );
            }
        }
    }
}
