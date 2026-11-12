//! Development-only support crate for sesmesh tests and benchmarks.
//!
//! * [`oracle`] — brute-force evaluation of the SES/SAS/vdW fields and of the
//!   SAS arrangement (circles, triple points), with no acceleration beyond
//!   exact pair filtering. Ground truth for the optimized library code.
//! * [`analytic`] — closed-form and high-precision reference values for
//!   single- and two-sphere surfaces.
//! * [`clusters`] — deterministic random and symmetric sphere clusters, and
//!   `proptest` strategies.
//! * [`obj`] — a minimal OBJ reader for round-trip tests.
//!
//! The crate deliberately does not depend on `sesmesh`: it works on plain
//! `[f64; 3]` centres and radii, so `sesmesh`'s own unit tests can use it
//! without type duplication. Nothing here is part of the public API.

pub mod analytic;
pub mod clusters;
pub mod obj;
pub mod oracle;

/// A point or vector in Å.
pub type V3 = [f64; 3];

pub(crate) mod v {
    use super::V3;

    pub(crate) fn sub(a: V3, b: V3) -> V3 {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    pub(crate) fn add(a: V3, b: V3) -> V3 {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    pub(crate) fn mul(a: V3, s: f64) -> V3 {
        [a[0] * s, a[1] * s, a[2] * s]
    }
    pub(crate) fn dot(a: V3, b: V3) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    pub(crate) fn cross(a: V3, b: V3) -> V3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }
    pub(crate) fn norm(a: V3) -> f64 {
        dot(a, a).sqrt()
    }
    pub(crate) fn dist(a: V3, b: V3) -> f64 {
        norm(sub(a, b))
    }
    pub(crate) fn normalize(a: V3) -> V3 {
        mul(a, 1.0 / norm(a))
    }
}
