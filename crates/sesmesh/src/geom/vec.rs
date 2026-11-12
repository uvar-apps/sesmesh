//! Minimal internal 3-vector (never exposed in the public API, Design §1).

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A 3-vector of `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Vec3 {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Vec3 {
    pub(crate) const X: Vec3 = Vec3::new(1.0, 0.0, 0.0);
    pub(crate) const Y: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub(crate) const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    pub(crate) const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub(crate) const fn from_array(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    pub(crate) const fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub(crate) fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub(crate) fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub(crate) fn norm2(self) -> f64 {
        self.dot(self)
    }

    pub(crate) fn norm(self) -> f64 {
        self.norm2().sqrt()
    }

    pub(crate) fn dist(self, o: Vec3) -> f64 {
        (self - o).norm()
    }

    pub(crate) fn normalized(self) -> Vec3 {
        self / self.norm()
    }

    pub(crate) fn min(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    pub(crate) fn max(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    pub(crate) fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// A deterministic orthonormal pair `(e1, e2)` perpendicular to the unit
/// vector `u`, with `(e1, e2, u)` right-handed.
pub(crate) fn orthonormal_basis(u: Vec3) -> (Vec3, Vec3) {
    let (ax, ay, az) = (u.x.abs(), u.y.abs(), u.z.abs());
    let helper = if ax <= ay && ax <= az {
        Vec3::X
    } else if ay <= az {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let e1 = helper.cross(u).normalized();
    let e2 = u.cross(e1);
    (e1, e2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_is_orthonormal_and_right_handed() {
        for u in [
            Vec3::X,
            Vec3::new(0.3, -0.4, 0.866),
            Vec3::new(-1.0, 1.0, 1e-9),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let u = u.normalized();
            let (e1, e2) = orthonormal_basis(u);
            for (a, b) in [(e1, e2), (e1, u), (e2, u)] {
                assert!(a.dot(b).abs() < 1e-12);
            }
            assert!((e1.norm() - 1.0).abs() < 1e-12 && (e2.norm() - 1.0).abs() < 1e-12);
            assert!((e1.cross(e2) - u).norm() < 1e-12);
        }
    }

    #[test]
    fn arithmetic() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-2.0, 0.5, 4.0);
        assert_eq!((a + b) - b, a);
        assert_eq!(a * 2.0 / 2.0, a);
        assert_eq!(-a + a, Vec3::default());
        assert_eq!(a.cross(b).dot(a), 0.0);
        assert_eq!(a.min(b), Vec3::new(-2.0, 0.5, 3.0));
        assert_eq!(Vec3::from_array(a.to_array()), a);
    }
}
